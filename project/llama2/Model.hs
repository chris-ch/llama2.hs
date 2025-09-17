module Model where

import qualified Prelude as P
import Clash.Prelude
import qualified Clash.Sized.Vector as CV
import qualified Data.Binary.Get as BG
import qualified Data.Vector.Unboxed as V
import qualified Foreign as F
import GHC.IO (unsafePerformIO)
import Data.ByteString (useAsCString)
import qualified GHC.TypeNats
import Data.Functor (($>))
import qualified Clash.Explicit.SimIO as SimIO
import Data.Foldable (for_)

import Helpers ( NumAttentionHeads
  , NumLayers
  , SeqLen
  , HeadDimension
  , ModelDim
  , HiddenDim
  , VocabSize
  , Token(..)
  , TransformerLayerComponent(..), TransformerDecoderComponent (..)
  , SingleHeadComponent(..)
  , MultiHeadAttentionComponent(..)
  , RotaryEncodingComponent(..)
  , FeedForwardNetworkComponent(..), EmbeddingComponent (..),
  runSingleHeadQKV, applyRotaryToHead, StepCount (..),
  computeFeedForward, transformerLogits, argMax, embed,
  drawSample,
   CArray2D (..), computeMultiHeadAttention
  )

-- ============================================================================
-- Types and Cache Management
-- ============================================================================

type CacheDepth = NumLayers GHC.TypeNats.* NumAttentionHeads GHC.TypeNats.* SeqLen GHC.TypeNats.* HeadDimension
type CacheAddr  = Index CacheDepth

-- Pure cache addressing
cacheAddr
  :: Index NumLayers
  -> Index NumAttentionHeads
  -> Index SeqLen
  -> Index HeadDimension
  -> Index CacheDepth
cacheAddr l h s d =
  let
    headSize = natToNum @HeadDimension
    seqSize  = natToNum @SeqLen * headSize
    layerSize = natToNum @NumAttentionHeads * seqSize
    addr =
      fromIntegral d
      + fromIntegral s * headSize
      + fromIntegral h * seqSize
      + fromIntegral l * layerSize
  in addr

-- Simplified cache structure - just the RAM functions
data AttentionCache dom = AttentionCache
  { keyCache :: Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float
  , valueCache :: Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float
  }

-- Initialize caches with block RAMs
initAttentionCache
  :: forall dom. HiddenClockResetEnable dom
  => AttentionCache dom
initAttentionCache = AttentionCache
  { keyCache = blockRam (replicate (SNat @CacheDepth) 0)
  , valueCache = blockRam (replicate (SNat @CacheDepth) 0)
  }

-- ============================================================================
-- Cache Access Functions (Handle Signals)
-- ============================================================================

-- Read a complete sequence from cache for one head
readCachedSequence
  :: forall dom
   . (KnownNat SeqLen, KnownNat HeadDimension)
  => (Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float)
  -> Index NumLayers
  -> Index NumAttentionHeads
  -> Int  -- current sequence position
  -> Signal dom (Vec SeqLen (Vec HeadDimension Float))
readCachedSequence ram layerIdx headIdx seqPos =
  let -- Create a mask for valid sequence positions
      isValidPos :: Index SeqLen -> Bool
      isValidPos i = fromIntegral i < seqPos

      -- Read all positions, but we'll mask invalid ones later
      readRow :: Index SeqLen -> Signal dom (Vec HeadDimension Float)
      readRow s = traverse (\d ->
        ram (pure $ cacheAddr layerIdx headIdx s d) (pure Nothing)
        ) indicesI

      -- Read all sequence positions
      allRows :: Signal dom (Vec SeqLen (Vec HeadDimension Float))
      allRows = traverse readRow indicesI

      -- Mask invalid positions with zeros
      zeroVec = repeat 0 :: Vec HeadDimension Float

  in fmap (imap (\i row -> if isValidPos i then row else zeroVec)
     ) allRows

-- Write current key/value to cache
writeToCacheRow
  :: forall dom
   . (Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float)
  -> Index NumLayers
  -> Index NumAttentionHeads
  -> Int  -- current sequence position
  -> Vec HeadDimension Float
  -> Signal dom ()  -- side effect
writeToCacheRow ram layerIdx headIdx seqPos vec =
  let seqIdx = fromIntegral seqPos :: Index SeqLen
      writeOps = imap (\d val ->
        ram (pure $ cacheAddr layerIdx headIdx seqIdx d)
            (pure $ Just (cacheAddr layerIdx headIdx seqIdx d, val)) ) vec
  in sequenceA writeOps $> ()

-- ============================================================================
-- Main Transformer Logic
-- ============================================================================

-- Single transformer layer with proper cache management
transformerLayer
  :: forall dom
   . ( Monad (Signal dom))
  =>TransformerLayerComponent
  -> AttentionCache dom
  -> Index NumLayers
  -> Int  -- current sequence position
  -> Signal dom (Vec ModelDim Float)  -- input token
  -> Signal dom (Vec ModelDim Float)  -- output token
transformerLayer layer cache layerIdx seqPos inputSig = do
  let mha = multiHeadAttention layer
      ffn = feedforwardNetwork layer
      keyRam = keyCache cache
      valueRam = valueCache cache

  inputSig >>= \input -> do
    -- Generate Q, K, V with rotary encoding for current input
    let currentQKV = map (\hIdx ->
          let headComp = heads mha !! hIdx
              (q, k, v) = runSingleHeadQKV headComp input
              -- Apply rotary encoding here
              (q_rot, k_rot) = applyRotaryToHead headComp (StepCount (fromIntegral seqPos)) (q, k)
          in (q_rot, k_rot, v)
          ) indicesI

        queries = map (\(q, _, _) -> q) currentQKV

    -- Write ROTARY-ENCODED K, V to cache
    for_ (imap (,) currentQKV) $ \(hIdx, (k, v, _)) ->
        writeToCacheRow keyRam   layerIdx hIdx seqPos k *>
        writeToCacheRow valueRam layerIdx hIdx seqPos v

    -- Read cached sequences INCLUDING the just-written current timestep
    keysPerHead <- traverse (\hIdx ->
      readCachedSequence keyRam layerIdx hIdx (seqPos + 1) -- +1 to include current
      ) indicesI

    valuesPerHead <- traverse (\hIdx ->
      readCachedSequence valueRam layerIdx hIdx (seqPos + 1)
      ) indicesI

    -- Compute attention and feed-forward (pure computations)
    let -- Apply RMS normalization to input for attention
        tokenAfterAttention = computeMultiHeadAttention mha input queries keysPerHead valuesPerHead
        -- Apply feed-forward network with its own normalization and residual
        ffnOutput = computeFeedForward ffn tokenAfterAttention
        -- Final output (second residual connection handled in computeFeedForward)
        layerOutput = ffnOutput

    return layerOutput

-- Full transformer with all layers
transformer
  :: forall dom .
     (Monad (Signal dom))
  => TransformerDecoderComponent                 -- ^ decoder (holds embedding, layers, logits)
  -> Vec NumLayers (AttentionCache dom)          -- ^ per‑layer caches
  -> Int                                         -- ^ current sequence position (step)
  -> Token                                       -- ^ input token
  -> Float                                       -- ^ temperature (0 → deterministic)
  -> Int                                         -- ^ RNG seed for sampling
  -> [Token]                                     -- ^ prompt tokens (may be empty)
  -> Signal dom Token                            -- ^ produced token
transformer decoder caches seqPos inputToken temperature seed promptTokens = do
  let embedding = modelEmbedding decoder
      layers    = modelLayers decoder

      -- Embed the incoming token
      inputVector :: Vec ModelDim Float
      inputVector = embed (vocabulary embedding) inputToken

      ------------------------------------------------------------------
      -- Recursive processing of all layers
      ------------------------------------------------------------------
      processLayers
        :: Vec ModelDim Float          -- ^ current hidden vector
        -> Int                         -- ^ layer index we are about to run
        -> Signal dom (Vec ModelDim Float)
      processLayers vec layerIdx
        | layerIdx >= length layers = return vec
        | otherwise = do
            let layer = layers !! layerIdx
                cache = caches   !! layerIdx
            out <- transformerLayer layer cache (fromIntegral layerIdx) seqPos (pure vec)
            processLayers out (layerIdx + 1)

  -- Run through every layer
  finalHidden <- processLayers inputVector 0

  ------------------------------------------------------------------
  -- Logits & token selection
  ------------------------------------------------------------------
  let rawLogits :: Vec VocabSize Float
      rawLogits = transformerLogits decoder finalHidden

      -- Helper to pick the next token from a probability vector
      pickToken :: Vec VocabSize Float -> Token
      pickToken probs
        | temperature == 0.0 = fromIntegral $ argMax probs
        | otherwise =
            let scaled = map (/ temperature) probs
            in Token $ drawSample seed scaled

  -- Prompt‑token shortcut: if we are still inside the prompt length,
  -- just emit the corresponding prompt token.
  let nextToken
        | seqPos < P.length promptTokens = promptTokens P.!! seqPos
        | otherwise                    = pickToken rawLogits

  return nextToken

-- ============================================================================
-- File Parsing
-- ============================================================================

readVector :: Int -> BG.Get (V.Vector Float)
readVector count = do
  byteData <- BG.getByteString (count * 4)
  return $! unsafePerformIO $ do
    useAsCString byteData $ \ptr -> do
      let floatPtr = F.castPtr ptr :: F.Ptr Float
      V.generateM count (F.peekElemOff floatPtr)

readVec1D :: forall n. KnownNat n => BG.Get (Vec n Float)
readVec1D = do
    let total = snatToNum (SNat :: SNat n)
    vec <- readVector total
    return $ CV.unsafeFromList (V.toList vec)

readVec2D :: forall n m. (KnownNat n, KnownNat m) => BG.Get (Vec n (Vec m Float))
readVec2D = do
    let n = snatToNum (SNat :: SNat n)
        m = snatToNum (SNat :: SNat m)
        total = n * m
    vec <- readVector total
    let floatList = V.toList vec
        chunks = chunksOf m floatList
        vecs = P.map CV.unsafeFromList chunks
    return $ CV.unsafeFromList vecs
  where
    chunksOf :: Int -> [a] -> [[a]]
    chunksOf _ [] = []
    chunksOf k xs = P.take k xs : chunksOf k (P.drop k xs)

readVec3D :: forall n m p. (KnownNat n, KnownNat m, KnownNat p) => BG.Get (Vec n (Vec m (Vec p Float)))
readVec3D = do
    let n = snatToNum (SNat :: SNat n)
        m = snatToNum (SNat :: SNat m)
        p = snatToNum (SNat :: SNat p)
        total = n * m * p
    vec <- readVector total
    let floatList = V.toList vec
        innerChunks = chunksOf p floatList
        innerVecs = P.map CV.unsafeFromList innerChunks
        middleChunks = chunksOf m innerVecs
        middleVecs = P.map CV.unsafeFromList middleChunks
    return $ CV.unsafeFromList middleVecs
  where
    chunksOf :: Int -> [a] -> [[a]]
    chunksOf _ [] = []
    chunksOf k xs = P.take k xs : chunksOf k (P.drop k xs)

parseModelConfigFile :: BG.Get TransformerDecoderComponent
parseModelConfigFile = do
  _ <- BG.getInt32le
  _ <- BG.getInt32le
  _ <- BG.getInt32le
  _ <- BG.getInt32le
  _ <- BG.getInt32le
  _ <- BG.getInt32le
  _ <- BG.getInt32le
  tokenEmbeddingTable' <- readVec2D @VocabSize
  rmsAttWeight' <- readVec2D @NumLayers
  wq' <- readVec3D @NumLayers @HeadDimension @ModelDim
  wk' <- readVec3D @NumLayers @HeadDimension @ModelDim
  wv' <- readVec3D @NumLayers @HeadDimension @ModelDim
  wo' <- readVec3D @NumLayers @ModelDim @ModelDim
  rmsFfnWeight' <- readVec2D @NumLayers
  w1' <- readVec3D @NumLayers @HiddenDim @ModelDim
  w2' <- readVec3D @NumLayers @ModelDim @HiddenDim
  w3' <- readVec3D @NumLayers @HiddenDim @ModelDim
  rmsFinalWeight' <- readVec1D @ModelDim
  freqCisReal' <- readVec2D @SeqLen
  freqCisImag' <- readVec2D @SeqLen

  let embedding = EmbeddingComponent
        { vocabulary = CArray2D tokenEmbeddingTable'
        , rmsFinalWeight = rmsFinalWeight'
        }

      sha hIdx = SingleHeadComponent
        { wqHead = CArray2D $ wq' !! toInteger hIdx
        , wkHead = CArray2D $ wk' !! toInteger hIdx
        , wvHead = CArray2D $ wv' !! toInteger hIdx
        , rotary = RotaryEncodingComponent
            { freqCos = CArray2D freqCisReal'
            , freqSin = CArray2D freqCisImag'
            }
        }

      layer lIdx = TransformerLayerComponent
        { multiHeadAttention = MultiHeadAttentionComponent
            { heads = map sha (indicesI :: Vec NumAttentionHeads (Index NumAttentionHeads))
            , mWo = CArray2D $ wo' !! lIdx
            , rmsAtt = rmsAttWeight' !! lIdx
            }
        , feedforwardNetwork = FeedForwardNetworkComponent
            { fW1 = CArray2D $ w1' !! toInteger lIdx
            , fW2 = CArray2D $ w2' !! toInteger lIdx
            , fW3 = CArray2D $ w3' !! toInteger lIdx
            , fRMSFfn = rmsFfnWeight' !! lIdx
            }
        }

      decoder = TransformerDecoderComponent
        { modelEmbedding = embedding
        , modelLayers = map layer (indicesI :: Vec NumLayers (Index NumLayers))
        }

  return decoder
