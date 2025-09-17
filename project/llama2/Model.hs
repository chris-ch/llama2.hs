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
   CArray2D (..), computeMultiHeadAttention, seqLen
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
  -> Signal dom Int  -- current sequence position
  -> Signal dom (Vec SeqLen (Vec HeadDimension Float))
readCachedSequence ram layerIdx headIdx seqPos =
  let -- Read all positions unconditionally
      readRow :: Index SeqLen -> Signal dom (Vec HeadDimension Float)
      readRow s = traverse (\d ->
        ram (pure $ cacheAddr layerIdx headIdx s d) (pure Nothing)
        ) indicesI

      -- Read all sequence positions
      allRows :: Signal dom (Vec SeqLen (Vec HeadDimension Float))
      allRows = traverse readRow indicesI

      -- Zero vector for masking invalid positions
      zeroVec = repeat 0 :: Vec HeadDimension Float

      -- Apply masking based on current sequence position
      maskRows :: Int -> Vec SeqLen (Vec HeadDimension Float) -> Vec SeqLen (Vec HeadDimension Float)
      maskRows currentPos = imap (\i row ->
        if fromIntegral i < currentPos then row else zeroVec
        )

  in liftA2 maskRows seqPos allRows

-- Write current key/value to cache
writeToCacheRow
  :: forall dom
   . (Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float)
  -> Index NumLayers
  -> Index NumAttentionHeads
  -> Signal dom Int
  -> Vec HeadDimension Float
  -> Signal dom ()
writeToCacheRow ram layerIdx headIdx seqPos vec =
  let
      writeOps :: Vec HeadDimension (Signal dom Float)
      writeOps = imap (\d val ->
          ram (fmap (\s -> cacheAddr layerIdx headIdx (fromIntegral s) d) seqPos)
              (fmap (\s -> Just (cacheAddr layerIdx headIdx (fromIntegral s) d, val)) seqPos)
        ) vec
  in sequenceA writeOps $> ()

-- ============================================================================
-- Main Transformer Logic
-- ============================================================================

-- | Single transformer layer (synthesizable)
transformerLayer
  :: forall dom
   . TransformerLayerComponent
  -> AttentionCache dom
  -> Index NumLayers            -- ^ layer index (static)
  -> Signal dom Int             -- ^ current sequence position
  -> Signal dom (Vec ModelDim Float)  -- ^ input token
  -> Signal dom (Vec ModelDim Float)  -- ^ output token
transformerLayer layer cache layerIdx seqPosSig inputSig =
  outputSig
  where
    mha = multiHeadAttention layer
    ffn = feedforwardNetwork layer
    keyRam = keyCache cache
    valueRam = valueCache cache

    -- Step 1: compute Q, K, V for each head
    qkvSig :: Signal dom (Vec NumAttentionHeads (Vec HeadDimension Float, Vec HeadDimension Float, Vec HeadDimension Float))
    qkvSig = liftA2 (\seqPos input ->
                      map (\hIdx ->
                        let headComp = heads mha !! hIdx
                            (q, k, v) = runSingleHeadQKV headComp input
                            (q_rot, k_rot) = applyRotaryToHead headComp (StepCount (fromIntegral seqPos)) (q, k)
                        in (q_rot, k_rot, v)
                      ) indicesI
                  ) seqPosSig inputSig

    -- Step 2: write K,V to caches
{-     _cacheWriteSig :: Signal dom ()
    _cacheWriteSig = fmap (\qkvVec ->
      for_ (imap (,) qkvVec) $ \(hIdx, (k, v, _)) ->
        writeToCacheRow keyRam layerIdx hIdx seqPosSig k *>
        writeToCacheRow valueRam layerIdx hIdx seqPosSig v
      ) qkvSig -}

    newSeqPos = seqPosSig + 1

    -- Step 3: read cached sequences including current timestep
    keysPerHeadSig :: Signal dom (Vec NumAttentionHeads (Vec SeqLen (Vec HeadDimension Float)))
    keysPerHeadSig = traverse (\hIdx ->
        readCachedSequence keyRam layerIdx hIdx newSeqPos
      ) indicesI

    valuesPerHeadSig :: Signal dom (Vec NumAttentionHeads (Vec SeqLen (Vec HeadDimension Float)))
    valuesPerHeadSig = traverse (\hIdx ->
      readCachedSequence valueRam layerIdx hIdx newSeqPos
      ) indicesI

    -- Step 4: compute attention and feed-forward
    outputSig :: Signal dom (Vec ModelDim Float)
    outputSig =
      (\input qkv keys values ->
         let queries = map (\(q, _, _) -> q) qkv
             tokenAfterAttention = computeMultiHeadAttention mha input queries keys values
             ffnOutput = computeFeedForward ffn tokenAfterAttention
         in ffnOutput
      ) <$> inputSig <*> qkvSig <*> keysPerHeadSig <*> valuesPerHeadSig

-- Full transformer with all layers
transformer
  :: forall dom
   . HiddenClockResetEnable dom
  =>  TransformerDecoderComponent
  -> Vec NumLayers (AttentionCache dom)
  -> Signal dom Int              -- ^ current sequence position (step)
  -> Signal dom Token            -- ^ input token
  -> Signal dom Float            -- ^ temperature
  -> Signal dom Int              -- ^ RNG seed for sampling
  -> Signal dom (Vec SeqLen Token) -- ^ prompt tokens (fixed length)
  -> Signal dom Token            -- ^ produced token
transformer decoder caches seqPosSig tokenSig temperatureSig seedSig promptTokensSig =
  let
    -- ------------------------------------------------------------------
    -- Embedding
    -- ------------------------------------------------------------------
    embedding = modelEmbedding decoder
    layers    = modelLayers decoder

    -- Embed token -> Vec ModelDim Float, as a Signal
    inputVecSig :: Signal dom (Vec ModelDim Float)
    inputVecSig = embed (vocabulary embedding) <$> tokenSig

    -- ------------------------------------------------------------------
    -- Chain all transformer layers
    -- Each layer: Signal (Vec ModelDim Float) -> Signal (Vec ModelDim Float)
    -- ------------------------------------------------------------------
    chainLayers
      :: Signal dom (Vec ModelDim Float)
      -> Vec NumLayers TransformerLayerComponent
      -> Vec NumLayers (AttentionCache dom)
      -> Signal dom (Vec ModelDim Float)
    chainLayers sigIn layersVec cachesVec =
      foldl (\acc (layer, cache, idx) ->
               transformerLayer layer cache idx seqPosSig acc)
            sigIn
            (zip3 layersVec cachesVec (indicesI @NumLayers))

    hiddenSig :: Signal dom (Vec ModelDim Float)
    hiddenSig = chainLayers inputVecSig layers caches

    -- ------------------------------------------------------------------
    -- Logits
    -- ------------------------------------------------------------------
    rawLogitsSig :: Signal dom (Vec VocabSize Float)
    rawLogitsSig = transformerLogits decoder <$> hiddenSig

    -- ------------------------------------------------------------------
    -- Sampling / Argmax
    -- ------------------------------------------------------------------
    pickToken :: Vec VocabSize Float -> Float -> Int -> Token
    pickToken probs temp seed
      | temp == 0.0 = Token (argMax probs)
      | otherwise  =
          let scaled = map (/ temp) probs
          in Token (drawSample seed scaled)

    pickSig :: Signal dom Token
    pickSig = pickToken <$> rawLogitsSig <*> temperatureSig <*> seedSig

    -- ------------------------------------------------------------------
    -- Prompt tokens
    -- ------------------------------------------------------------------
    promptLenSig :: Signal dom Int
    promptLenSig = pure seqLen

    -- Select the prompt token at seqPos
    promptTokenSig :: Signal dom Token
    promptTokenSig =
      (!!) <$> promptTokensSig <*> seqPosSig

    -- ------------------------------------------------------------------
    -- Output token: if we’re still in prompt, echo prompt; else pick new
    -- ------------------------------------------------------------------
    outTokenSig :: Signal dom Token
    outTokenSig =
      mux ((<) <$> seqPosSig <*> promptLenSig)
          promptTokenSig
          pickSig

  in outTokenSig

-- | Top entity: takes inputs as signals and produces the next token.
--   The model is fixed as a parameter.
topEntity
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> Signal dom Int                  -- ^ sequence position
  -> Signal dom Token                -- ^ input token
  -> Signal dom Float                -- ^ temperature
  -> Signal dom Int                  -- ^ RNG seed
  -> Signal dom (Vec SeqLen Token)   -- ^ prompt tokens
  -> Signal dom Token
topEntity decoder = transformer decoder caches
  where
    -- Initialize caches for all layers
    caches :: Vec NumLayers (AttentionCache dom)
    caches = repeat initAttentionCache

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
