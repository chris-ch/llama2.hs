module Model (topEntity, multiCycleTransformer, initAttentionCache, ProcessingState(..)) where

import Clash.Prelude
import qualified Prelude as P
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
-- Cache Access (sequential, single-element per cycle)
-- ============================================================================

-- Read one element per cycle from a given BRAM function, and incrementally
-- fill a buffered Vec SeqLen (Vec HeadDimension Float). The BRAM read has
-- one-cycle latency, so we track previous indices to write the result back.
readCachedSequenceWith
  :: forall dom
   . HiddenClockResetEnable dom
  => (Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float) -- ^ BRAM port (keyCache or valueCache)
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom Int                                        -- ^ pos (causal horizon)
  -> Signal dom (Vec SeqLen (Vec HeadDimension Float))     -- ^ buffer filled over time
readCachedSequenceWith ram l h posSig = bufSig
 where
  -- 2D counters over (t,d)
  nextTD :: (Index SeqLen, Index HeadDimension) -> (Index SeqLen, Index HeadDimension)
  nextTD (t, d) =
    if d == maxBound
      then (if t == maxBound then 0 else succ t, 0)
      else (t, succ d)

  -- drive address
  tdSig :: Signal dom (Index SeqLen, Index HeadDimension)
  tdSig = register (0,0) (nextTD <$> tdSig)

  tSig = fst <$> tdSig
  dSig = snd <$> tdSig

  -- BRAM address for current indices
  addrSig = cacheAddr l h <$> tSig <*> dSig

  -- issue read; no write here
  qSig :: Signal dom Float
  qSig = ram addrSig (pure Nothing)

  -- due to BRAM latency, capture previous indices
  tPrevSig = register 0 tSig
  dPrevSig = register 0 dSig
  posPrevSig = register 0 posSig

  -- local buffer register, updated 1 element/cycle
  bufInit :: Vec SeqLen (Vec HeadDimension Float)
  bufInit = repeat (repeat 0)

  bufNext
    :: Signal dom (Vec SeqLen (Vec HeadDimension Float))
    -> Signal dom (Vec SeqLen (Vec HeadDimension Float))
  bufNext bufS =
    liftA5
      (\buf tPrev dPrev _ q ->
        -- Do not mask here. Keep the cache values as-is; apply causal mask at logits.
        let
          oldRow  = buf !! tPrev
          newRow  = replace dPrev q oldRow
         in replace tPrev newRow buf)
      bufS tPrevSig dPrevSig posPrevSig qSig

  bufSig = register bufInit (bufNext bufSig)

-- Key / Value wrappers
readCachedKeys
  :: forall dom. HiddenClockResetEnable dom
  => AttentionCache dom
  -> Index NumLayers -> Index NumKeyValueHeads -> Signal dom Int
  -> Signal dom (Vec SeqLen (Vec HeadDimension Float))
readCachedKeys cache = readCachedSequenceWith (keyCache cache)

readCachedVals
  :: forall dom. HiddenClockResetEnable dom
  => AttentionCache dom
  -> Index NumLayers -> Index NumKeyValueHeads -> Signal dom Int
  -> Signal dom (Vec SeqLen (Vec HeadDimension Float))
readCachedVals cache = readCachedSequenceWith (valueCache cache)

-- Sequential writer: one element of K and one element of V per cycle.
-- Gated by explicit 'enSig' so you can assert it only in Cycle4_WriteCache
-- of the active layer. Writes to 'seqPosSig' row, sweeping 'd' across time.
writeToCacheSequence
  :: forall dom
   . HiddenClockResetEnable dom
  => AttentionCache dom
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom Int                                   -- ^ sequence position (Int, clamped)
  -> Signal dom (Vec HeadDimension Float, Vec HeadDimension Float) -- ^ current (K,V) vectors
  -> Signal dom Bool                                  -- ^ enable pulse/level
  -> Signal dom ()                                    -- ^ dummy sink
writeToCacheSequence cache l h seqPosSig kvSig enSig =
  sinkSig
 where
  keyRam   = keyCache cache
  valueRam = valueCache cache

  -- dimension counter, increments only when enabled
  dCnt :: Signal dom (Index HeadDimension)
  dCnt = register 0 nextD
  nextD = mux enSig
              (P.fmap (\d -> if d == maxBound then 0 else succ d) dCnt)
              dCnt

  -- clamp sequence position to [0 .. SeqLen-1]
  clamp :: Int -> Int -> Int -> Int
  clamp lo hi x = max lo (min hi x)

  seqIdxSig :: Signal dom (Index SeqLen)
  seqIdxSig = fmap (toEnum . clamp 0 (natToNum @SeqLen - 1)) seqPosSig

  -- select the K,V element for current dimension
  kElemSig :: Signal dom Float
  kElemSig = (\(k,_ ) d -> k !! d) <$> kvSig <*> dCnt

  vElemSig :: Signal dom Float
  vElemSig = (\(_ ,v) d -> v !! d) <$> kvSig <*> dCnt

  -- compute linear addresses
  kAddrSig = cacheAddr l h <$> seqIdxSig <*> dCnt
  vAddrSig = kAddrSig  -- same addressing expression for value cache

  -- build write enables
  kWriteSig :: Signal dom (Maybe (CacheAddr, Float))
  kWriteSig = mux enSig (Just <$> bundle (kAddrSig, kElemSig)) (pure Nothing)

  vWriteSig :: Signal dom (Maybe (CacheAddr, Float))
  vWriteSig = mux enSig (Just <$> bundle (vAddrSig, vElemSig)) (pure Nothing)

  -- perform writes; tie off read addresses (dummy) to keep the ports alive
  _kQ = keyRam kAddrSig kWriteSig
  _vQ = valueRam vAddrSig vWriteSig

  -- force usage to avoid being optimized away
  sinkSig = () <$ (_kQ + _vQ)

liftA5 :: Applicative g => (a -> b -> c -> d -> e -> f) -> g a -> g b -> g c -> g d -> g e -> g f
liftA5 f fa fb fc fd fe = f <$> fa <*> fb <*> fc <*> fd <*> fe

-- ============================================================================
-- Multi-Cycle Transformer Layer
-- ============================================================================

multiCycleTransformerLayer
  :: forall dom
  . HiddenClockResetEnable dom => TransformerLayerComponent
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
  nextStateSig = fmap nextState stateSig

  -- Cache reads for all heads
  cachedKeysPerHead :: Vec NumKeyValueHeads (Signal dom (Vec SeqLen (Vec HeadDimension Float)))
  cachedKeysPerHead =
    map (\hIdx -> readCachedKeys cache layerIdx hIdx (psSeqPos <$> stateSig)) indicesI

  cachedValsPerHead :: Vec NumKeyValueHeads (Signal dom (Vec SeqLen (Vec HeadDimension Float)))
  cachedValsPerHead =
    map (\hIdx -> readCachedVals cache layerIdx hIdx (psSeqPos <$> stateSig)) indicesI

  -- Bundle all heads into one signal of Vec-of-Vecs
  cachedKeysSig :: Signal dom (Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float)))
  cachedKeysSig = sequenceA cachedKeysPerHead

  cachedValsSig :: Signal dom (Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float)))
  cachedValsSig = sequenceA cachedValsPerHead

  writeEnableSig :: Signal dom Bool
  writeEnableSig = liftA2 (\st lidx -> psStage st == Cycle4_WriteCache && psLayer st == lidx)
                          stateSig (pure layerIdx)

  -- Per-head KV signals and write effects, sequenced so they are kept alive
  kvAllHeadsSig :: Signal dom (Vec NumKeyValueHeads (Vec HeadDimension Float, Vec HeadDimension Float))
  kvAllHeadsSig = liftA2 (zipWith (,)) (idKeys <$> dataSig) (idValues <$> dataSig)

  writeOps :: Vec NumKeyValueHeads (Signal dom ())
  writeOps =
    map (\hIdx ->
           writeToCacheSequence cache layerIdx hIdx
             (psSeqPos <$> stateSig)
             (((!!) <$> kvAllHeadsSig) <*> pure hIdx)
             writeEnableSig)
        indicesI

  writeOpsSig :: Signal dom ()
  writeOpsSig = () <$ sequenceA writeOps

  -- Combine everything in processCycle
  baseNextDataSig = liftA4 processCycle stateSig dataSig cachedKeysSig cachedValsSig

  -- Wire in the write operations to ensure they happen
  nextDataSig = liftA2 P.const baseNextDataSig writeOpsSig

  processCycle
    :: ProcessingState
    -> IntermediateData
    -> Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float))
    -> Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float))
    -> IntermediateData
  processCycle state idata keysNow valsNow =
    -- Only process when this layer is active
    if psLayer state /= layerIdx
    then idata  -- Pass through unchanged if not processing this layer
    else
      case psStage state of
        Cycle1_ReadCache ->
          idata { idCachedKeys = keysNow, idCachedVals = valsNow }
        Cycle2_ComputeQKV ->
            -- Compute keys and values for NumKeyValueHeads
            let
              input = idInputVec idata
              -- Compute queries for all NumQueryHeads
              queries = imap (\hIdx _ ->
                let headComp = heads mha !! hIdx
                    (q, _, _) = runSingleHeadQKV headComp input
                    (q_rot, _) = applyRotaryToHead headComp (StepCount $ P.fromIntegral $ psSeqPos state) (q, repeat 0)
                in q_rot
                ) indicesI

              -- Compute keys and values for NumKeyValueHeads
              headsPerGroupI :: Int
              headsPerGroupI = natToNum @NumQueryHeads `P.div` natToNum @NumKeyValueHeads

              toIndexClamped :: forall n. KnownNat n => Int -> Index n
              toIndexClamped i =
                let hi = natToNum @n - 1
                    j  = max 0 (min hi i)
                in toEnum j
              keysAndValues = imap (\hIdx _ ->
                let qIdx :: Index NumQueryHeads
                    qIdx = toIndexClamped @NumQueryHeads (fromEnum hIdx * headsPerGroupI)
                    headComp = heads mha !! qIdx
                    (_, k, v) = runSingleHeadQKV headComp input
                    (_, k_rot) = applyRotaryToHead headComp (StepCount $ P.fromIntegral $ psSeqPos state) (repeat 0, k)
                in (k_rot, v)
                ) indicesI
              (keys, values) = unzip keysAndValues
          in idata { idQueries = queries, idKeys = keys, idValues = values }
        Cycle3_ComputeAttn ->
          -- Include the just-computed (K,V) at the current position 'pos' by
          -- overlaying them on top of the streamed cache before attention.
          let
            -- clamp Int pos to Index SeqLen
            clamp :: Int -> Int -> Int -> Int
            clamp lo hi x = max lo (min hi x)
            posIdx :: Index SeqLen
            posIdx = toEnum (clamp 0 (natToNum @SeqLen - 1) (psSeqPos state))

            -- Current K,V per KV head from Cycle2_ComputeQKV
            curK :: Vec NumKeyValueHeads (Vec HeadDimension Float)
            curK = idKeys idata
            curV :: Vec NumKeyValueHeads (Vec HeadDimension Float)
            curV = idValues idata

            -- Overlay row 'posIdx' with current K,V for each KV head
            keysForAttn :: Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float))
            keysForAttn = imap (\h kCache -> replace posIdx (curK !! h) kCache) keysNow

            valsForAttn :: Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float))
            valsForAttn = imap (\h vCache -> replace posIdx (curV !! h) vCache) valsNow

            attnOut = computeMultiHeadAttention
                        mha
                        (idInputVec idata)
                        (idQueries idata)
                        keysForAttn
                        valsForAttn
                        (psSeqPos state)
          in idata { idAttnOutput = attnOut }
        Cycle5_ComputeFFN ->
          let ffnOut = computeFeedForward ffn (idAttnOutput idata)
          in idata { idFFNOutput = ffnOut }
        Cycle4_WriteCache ->
          idata

-- ============================================================================
-- Multi-Cycle Full Transformer
-- ============================================================================

-- Single state transition function
nextState :: ProcessingState -> ProcessingState
nextState st = case psStage st of
  Cycle1_ReadCache   -> st { psStage = Cycle2_ComputeQKV }
  Cycle2_ComputeQKV  -> st { psStage = Cycle3_ComputeAttn }
  Cycle3_ComputeAttn -> st { psStage = Cycle4_WriteCache }
  Cycle4_WriteCache  -> st { psStage = Cycle5_ComputeFFN }
  Cycle5_ComputeFFN ->
    if psLayer st == maxBound
      then st { psStage = Cycle1_ReadCache
              , psLayer = 0
              , psSeqPos = psSeqPos st + 1
              , psTokenReady = True }
      else st { psStage = Cycle1_ReadCache
              , psLayer = psLayer st + 1
              , psTokenReady = False }

multiCycleTransformer
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> Vec NumLayers (AttentionCache dom)
  -> Signal dom Token
  -> Signal dom Float
  -> Signal dom Int
  -> (Signal dom Token, Signal dom Bool)
multiCycleTransformer decoder caches tokenSig temperatureSig seedSig =
  (outputTokenSig, readyFlagSig)
 where
  embedding = modelEmbedding decoder
  layers    = modelLayers decoder
  inputVecSig = embed (vocabulary embedding) <$> tokenSig

  -- controller state: advance exactly once per cycle
  procStateSig   = register initialProcessingState (nextState <$> procStateSig)
  currentDataSig = register initialIntermediateData nextDataSig

  -- Input loading only at the start of a token
  inputLoadedSig = liftA3 (\state currentData inputVec ->
    if psStage state == Cycle1_ReadCache && psLayer state == 0
    then currentData { idInputVec = inputVec }
    else currentData) procStateSig currentDataSig inputVecSig

  -- Process through all layers; layers do NOT modify the state
  (_, nextDataSig) =
    foldl
      (\(_, dataSig) (layer, cache, layerIdx) ->
         let (_ignored, newDataSig) = multiCycleTransformerLayer layer cache layerIdx procStateSig dataSig
             conditionalDataSig = liftA3 (\state oldData newData ->
               if psLayer state == layerIdx then newData else oldData)
               procStateSig dataSig newDataSig
         in ((), conditionalDataSig))
      ((), inputLoadedSig)
      (zip3 layers caches indicesI)

  -- Use the single advanced state (procStateSig -> nextState step) for outputs
  nextStateSig = nextState <$> procStateSig

  logitsSig = liftA2 (\state finalData ->
    if psTokenReady state
    then transformerLogits decoder (idFFNOutput finalData)
    else repeat 0 ) nextStateSig nextDataSig

  pickToken probs temp seed
    | temp == 0.0 = argMax probs
    | otherwise   = drawSample seed (map (/ temp) probs)

  readyFlagSig = psTokenReady <$> nextStateSig
  outputTokenSig = pickToken <$> logitsSig <*> temperatureSig <*> seedSig

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
  -> (Signal dom Token, Signal dom Bool)
topEntity decoder _ = multiCycleTransformer decoder (repeat initAttentionCache)
