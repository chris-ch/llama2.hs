module Model (topEntity) where

import Clash.Prelude
import qualified GHC.TypeNats
import Data.Functor (($>))
import Helpers
  ( NumQueryHeads, NumKeyValueHeads, NumLayers, SeqLen, HeadDimension, ModelDim, VocabSize
  , Token, TransformerLayerComponent(..), TransformerDecoderComponent (..)
  , MultiHeadAttentionComponent(..), EmbeddingComponent (..)
  , runSingleHeadQKV, applyRotaryToHead, StepCount (..)
  , computeFeedForward, transformerLogits, argMax, embed
  , drawSample, computeMultiHeadAttention, seqLen, liftA4
  )
import GHC.Stack (HasCallStack)

-- ============================================================================
-- Multi-Cycle State Machine
-- ============================================================================

data CycleStage =
    Cycle1_ReadCache
  | Cycle2_ComputeQKV
  | Cycle3_ComputeAttn
  | Cycle4_WriteCache
  | Cycle5_ComputeFFN
  deriving (Show, Eq, Enum, Bounded)

instance NFDataX CycleStage where
  rnfX :: CycleStage -> ()
  rnfX x = seq x ()
  hasUndefined :: CycleStage -> Bool
  hasUndefined _ = False
  ensureSpine :: CycleStage -> CycleStage
  ensureSpine x = x
  deepErrorX :: HasCallStack => String -> CycleStage
  deepErrorX = errorX

data ProcessingState = ProcessingState
  { psStage      :: CycleStage
  , psLayer      :: Index NumLayers
  , psSeqPos     :: Int
  , psTokenReady :: Bool
  } deriving (Show, Generic, NFDataX)

initialProcessingState :: ProcessingState
initialProcessingState = ProcessingState { psStage = Cycle1_ReadCache, psLayer = 0, psSeqPos = 0, psTokenReady = False }

-- ============================================================================
-- Intermediate data storage
-- ============================================================================

data IntermediateData = IntermediateData
  { idInputVec    :: Vec ModelDim Float
  , idQueries     :: Vec NumQueryHeads (Vec HeadDimension Float)
  , idKeys        :: Vec NumKeyValueHeads (Vec HeadDimension Float)
  , idValues      :: Vec NumKeyValueHeads (Vec HeadDimension Float)
  , idCachedKeys  :: Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float))
  , idCachedVals  :: Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float))
  , idAttnOutput  :: Vec ModelDim Float
  , idFFNOutput   :: Vec ModelDim Float
  } deriving (Show, Generic, NFDataX)

initialIntermediateData :: IntermediateData
initialIntermediateData = IntermediateData
  { idInputVec   = repeat 0
  , idQueries    = repeat (repeat 0)
  , idKeys       = repeat (repeat 0)
  , idValues     = repeat (repeat 0)
  , idCachedKeys = repeat (repeat (repeat 0))
  , idCachedVals = repeat (repeat (repeat 0))
  , idAttnOutput = repeat 0
  , idFFNOutput  = repeat 0
  }

-- ============================================================================
-- Memory Interface
-- ============================================================================

type CacheDepth = NumLayers GHC.TypeNats.* NumKeyValueHeads GHC.TypeNats.* SeqLen GHC.TypeNats.* HeadDimension
type CacheAddr  = Index CacheDepth

cacheAddr :: Index NumLayers -> Index NumKeyValueHeads -> Index SeqLen -> Index HeadDimension -> CacheAddr
cacheAddr l h s d =
  let headSize  = natToNum @HeadDimension
      seqSize   = natToNum @SeqLen * headSize
      layerSize = natToNum @NumKeyValueHeads * seqSize
      addr = fromIntegral d + fromIntegral s * headSize + fromIntegral h * seqSize + fromIntegral l * layerSize
  in addr

data AttentionCache dom = AttentionCache
  { keyCache   :: Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float
  , valueCache :: Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float
  }

initAttentionCache :: forall dom. HiddenClockResetEnable dom => AttentionCache dom
initAttentionCache = AttentionCache
  { keyCache = blockRam (replicate (SNat @CacheDepth) 0)
  , valueCache = blockRam (replicate (SNat @CacheDepth) 0)
  }

-- ============================================================================
-- Cache Access Functions
-- ============================================================================

readCachedSequence
  :: forall dom
   . HiddenClockResetEnable dom
  => AttentionCache dom
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom Int
  -> Signal dom (Vec SeqLen (Vec HeadDimension Float))
readCachedSequence cache layerIdx headIdx seqPosSig =
  let keyRam = keyCache cache
      readAddresses = map (\s -> map (pure . cacheAddr layerIdx headIdx s) indicesI) indicesI
      readValues = map (map (\addr ->
        keyRam addr (pure Nothing)
        )) readAddresses
      rawSequence = traverse sequenceA readValues
      maskedSequence = liftA2 maskSeq seqPosSig rawSequence
      maskSeq pos = imap (\i row ->
        if fromIntegral i < pos then row else repeat 0
        )
  in maskedSequence

writeToCacheSequence
  :: forall dom
   . HiddenClockResetEnable dom
  => AttentionCache dom
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom Int
  -> Signal dom (Vec HeadDimension Float, Vec HeadDimension Float)
  -> Signal dom ()
writeToCacheSequence cache layerIdx headIdx seqPosSig kvSig =
  let keyRam = keyCache cache
      valueRam = valueCache cache
      keyWrites = liftA2 (\pos (k, _) ->
        imap (\d val ->
          let addr = cacheAddr layerIdx headIdx (fromIntegral pos) d
          in keyRam (pure addr) (pure $ Just (addr, val))
        ) k
        ) seqPosSig kvSig
      valueWrites = liftA2 (\pos (_, v) ->
        imap (\d val ->
          let addr = cacheAddr layerIdx headIdx (fromIntegral pos) d
          in valueRam (pure addr) (pure $ Just (addr, val))
        ) v
        ) seqPosSig kvSig
  in liftA2 (\_ _ -> ()) keyWrites valueWrites

-- ============================================================================
-- Multi-Cycle Transformer Layer
-- ============================================================================

multiCycleTransformerLayer
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayerComponent
  -> AttentionCache dom
  -> Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom IntermediateData
  -> (Signal dom ProcessingState, Signal dom IntermediateData)
multiCycleTransformerLayer layer cache layerIdx stateSig dataSig =
  (nextStateSig, nextDataSig)
 where
  mha = multiHeadAttention layer
  ffn = feedforwardNetwork layer

  nextStateSig = fmap advanceState stateSig
  advanceState state =
    case psStage state of
      Cycle1_ReadCache  -> state { psStage = Cycle2_ComputeQKV }
      Cycle2_ComputeQKV -> state { psStage = Cycle3_ComputeAttn }
      Cycle3_ComputeAttn -> state { psStage = Cycle4_WriteCache }
      Cycle4_WriteCache -> state { psStage = Cycle5_ComputeFFN }
      Cycle5_ComputeFFN ->
        if psLayer state == maxBound
        then state { psStage = Cycle1_ReadCache
                   , psLayer = 0
                   , psSeqPos = psSeqPos state + 1
                   , psTokenReady = True }
        else state { psStage = Cycle1_ReadCache, psLayer = psLayer state + 1 }

  -- Cache reads for all heads
  cachedKeysPerHead :: Vec NumKeyValueHeads (Signal dom (Vec SeqLen (Vec HeadDimension Float)))
  cachedKeysPerHead =
    map (\hIdx -> readCachedSequence cache layerIdx hIdx (psSeqPos <$> stateSig)) indicesI

  cachedValsPerHead :: Vec NumKeyValueHeads (Signal dom (Vec SeqLen (Vec HeadDimension Float)))
  cachedValsPerHead =
    map (\hIdx -> readCachedSequence cache layerIdx hIdx (psSeqPos <$> stateSig)) indicesI

  -- Bundle all heads into one signal of Vec-of-Vecs
  cachedKeysSig :: Signal dom (Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float)))
  cachedKeysSig = sequenceA cachedKeysPerHead

  cachedValsSig :: Signal dom (Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float)))
  cachedValsSig = sequenceA cachedValsPerHead

  -- Write to cache on Cycle4
  writeOpsSig :: Signal dom ()
  writeOpsSig =
    let kvSig = liftA2 zip (fmap idKeys dataSig) (fmap idValues dataSig)
    in bundle (traverse (imap (\hIdx kv ->
        writeToCacheSequence cache layerIdx hIdx (fmap psSeqPos stateSig) (pure kv)
        )) kvSig)
       $> ()

  -- Combine everything in processCycle
  baseNextDataSig = liftA4 processCycle stateSig dataSig cachedKeysSig cachedValsSig

  -- Tie the writeOpsSig in so it’s not dropped
  nextDataSig = liftA2 const baseNextDataSig writeOpsSig

  processCycle
    :: ProcessingState
    -> IntermediateData
    -> Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float))
    -> Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float))
    -> IntermediateData
  processCycle state idata keysNow valsNow =
    case psStage state of
      Cycle1_ReadCache ->
        idata { idCachedKeys = keysNow
              , idCachedVals = valsNow }
      Cycle2_ComputeQKV ->
        let input = idInputVec idata
            -- Compute queries for all NumAttentionHeads
            queries = imap (\hIdx _ ->
              let headComp = heads mha !! hIdx
                  (q, _, _) = runSingleHeadQKV headComp input
                  (q_rot, _) = applyRotaryToHead headComp (StepCount $ fromIntegral $ psSeqPos state) (q, repeat 0)
              in q_rot
              ) indicesI
            -- Compute keys and values for NumKeyValueHeads
            headsPerGroup = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads
            keysAndValues = imap (\hIdx _ ->
              let qIdx = hIdx * headsPerGroup
                  headComp = heads mha !! qIdx
                  (_, k, v) = runSingleHeadQKV headComp input
                  (_, k_rot) = applyRotaryToHead headComp (StepCount $ fromIntegral $ psSeqPos state) (repeat 0, k)
              in (k_rot, v)
              ) indicesI
            (keys, values) = unzip keysAndValues
        in idata { idQueries = queries, idKeys = keys, idValues = values }
      Cycle3_ComputeAttn ->
        let attnOut = computeMultiHeadAttention mha (idInputVec idata) (idQueries idata) keysNow valsNow
        in idata { idAttnOutput = attnOut }
      Cycle4_WriteCache ->
        idata  -- writes happen via writeOpsSig side effect
      Cycle5_ComputeFFN ->
        let ffnOut = computeFeedForward ffn (idAttnOutput idata)
        in idata { idFFNOutput = ffnOut }

-- ============================================================================
-- Multi-Cycle Full Transformer
-- ============================================================================

multiCycleTransformer
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> Vec NumLayers (AttentionCache dom)
  -> Signal dom Token
  -> Signal dom Float
  -> Signal dom Int
  -> Signal dom Token
multiCycleTransformer decoder caches tokenSig temperatureSig seedSig =
  outputTokenSig
 where
  embedding = modelEmbedding decoder
  layers    = modelLayers decoder
  inputVecSig = embed (vocabulary embedding) <$> tokenSig

  procStateSig = register initialProcessingState (pure initialProcessingState)
  intermDataSig = register initialIntermediateData $
    fmap (\input -> initialIntermediateData { idInputVec = input }) inputVecSig

  (finalStateSig, finalDataSig) =
    foldl
      (\(stateSig, dataSig) (layer, cache, layerIdx) ->
        multiCycleTransformerLayer layer cache layerIdx stateSig dataSig)
      (procStateSig, intermDataSig)
      (zip3 layers caches indicesI)

  finalOutputVecSig :: Signal dom (Vec ModelDim Float)
  finalOutputVecSig =
    register (repeat 0) (idFFNOutput <$> finalDataSig)

  logitsSig = transformerLogits decoder <$> finalOutputVecSig

  pickToken probs temp seed
    | temp == 0.0 = argMax probs
    | otherwise   = drawSample seed (map (/ temp) probs)

  sampledTokenSig = pickToken <$> logitsSig <*> temperatureSig <*> seedSig
  outputTokenSig  = sampledTokenSig

-- ============================================================================
-- Top Entity
-- ============================================================================

topEntity
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> Signal dom Int
  -> Signal dom Token
  -> Signal dom Float
  -> Signal dom Int
  -> Signal dom Token
topEntity decoder _ = multiCycleTransformer decoder caches
 where
  caches = repeat initAttentionCache
