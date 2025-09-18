module Model (topEntity) where

import Clash.Prelude

import qualified GHC.TypeNats
import Data.Functor (($>))

import Helpers ( NumAttentionHeads
  , NumLayers
  , SeqLen
  , HeadDimension
  , ModelDim
  , VocabSize
  , Token(..)
  , TransformerLayerComponent(..), TransformerDecoderComponent (..)
  , MultiHeadAttentionComponent(..)
  , EmbeddingComponent (..),
  runSingleHeadQKV, applyRotaryToHead, StepCount (..),
  computeFeedForward, transformerLogits, argMax, embed,
  drawSample, computeMultiHeadAttention, seqLen
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
   . TransformerDecoderComponent
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
