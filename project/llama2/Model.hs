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
  , drawSample, computeMultiHeadAttention, seqLen, liftA4, liftA5
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
  , psSeqPos     :: Index SeqLen
  , psTokenReady :: Bool
  } deriving (Show, Generic, NFDataX)

initialProcessingState :: ProcessingState
initialProcessingState = ProcessingState
  { psStage = Cycle1_ReadCache
  , psLayer = 0
  , psSeqPos = 0
  , psTokenReady = False
  }

-- Single state transition function (one step)
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
              , psSeqPos = succ (psSeqPos st)  -- wraps as Index
              , psTokenReady = True }
      else st { psStage = Cycle1_ReadCache
              , psLayer = succ (psLayer st)
              , psTokenReady = False }

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
  -> Signal dom (Index SeqLen)                               -- ^ pos (causal horizon) - not used to mask here
  -> Signal dom (Vec SeqLen (Vec HeadDimension Float))       -- ^ buffer filled over time
readCachedSequenceWith ram l h _posSig = bufSig
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

  -- local buffer register, updated 1 element/cycle
  bufInit :: Vec SeqLen (Vec HeadDimension Float)
  bufInit = repeat (repeat 0)

  bufNext
    :: Signal dom (Vec SeqLen (Vec HeadDimension Float))
    -> Signal dom (Vec SeqLen (Vec HeadDimension Float))
  bufNext bufS = liftA4 (\buf tPrev dPrev q ->
        let oldRow  = buf !! tPrev
            newRow  = replace dPrev q oldRow
        in replace tPrev newRow buf)
      bufS tPrevSig dPrevSig qSig

  bufSig = register bufInit (bufNext bufSig)

-- Key / Value wrappers
readCachedKeys
  :: forall dom. HiddenClockResetEnable dom
  => AttentionCache dom
  -> Index NumLayers -> Index NumKeyValueHeads -> Signal dom (Index SeqLen)
  -> Signal dom (Vec SeqLen (Vec HeadDimension Float))
readCachedKeys cache = readCachedSequenceWith (keyCache cache)

readCachedVals
  :: forall dom. HiddenClockResetEnable dom
  => AttentionCache dom
  -> Index NumLayers -> Index NumKeyValueHeads -> Signal dom (Index SeqLen)
  -> Signal dom (Vec SeqLen (Vec HeadDimension Float))
readCachedVals cache = readCachedSequenceWith (valueCache cache)

-- Sequential writer: one element of K and one element of V per cycle.
-- Enable is asserted for the whole stage; done pulses on last element.
writeToCacheSequence
  :: forall dom
   . HiddenClockResetEnable dom
  => AttentionCache dom
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom (Index SeqLen)                                   -- ^ sequence position
  -> Signal dom (Vec HeadDimension Float, Vec HeadDimension Float) -- ^ current (K,V) vectors
  -> Signal dom Bool                                             -- ^ enable (high across stage)
  -> Signal dom Bool                                             -- ^ done (1-cycle pulse)
writeToCacheSequence cache l h seqPosSig kvSig enSig =
  doneSig
 where
  keyRam   = keyCache cache
  valueRam = valueCache cache

  -- dimension counter; increments only when enabled, resets to 0 otherwise
  dCnt :: Signal dom (Index HeadDimension)
  dCnt = register 0 nextD
  nextD = mux enSig
              (P.fmap (\d -> if d == maxBound then 0 else succ d) dCnt)
              (pure 0)

  -- last element this cycle?
  lastElemSig :: Signal dom Bool
  lastElemSig = (== maxBound) <$> dCnt

  -- done when enabled and writing the last element
  doneSig :: Signal dom Bool
  doneSig = (&&) <$> enSig <*> lastElemSig

  -- select the K,V element for current dimension
  kElemSig :: Signal dom Float
  kElemSig = (\(k,_ ) d -> k !! d) <$> kvSig <*> dCnt

  vElemSig :: Signal dom Float
  vElemSig = (\(_ ,v) d -> v !! d) <$> kvSig <*> dCnt

  -- compute linear addresses
  kAddrSig = cacheAddr l h <$> seqPosSig <*> dCnt
  vAddrSig = kAddrSig  -- same addressing expression for value cache

  -- build write enables
  kWriteSig :: Signal dom (Maybe (CacheAddr, Float))
  kWriteSig = mux enSig (Just <$> bundle (kAddrSig, kElemSig)) (pure Nothing)

  vWriteSig :: Signal dom (Maybe (CacheAddr, Float))
  vWriteSig = mux enSig (Just <$> bundle (vAddrSig, vElemSig)) (pure Nothing)

  -- perform writes; tie off reads
  _kQ = keyRam   kAddrSig kWriteSig
  _vQ = valueRam vAddrSig vWriteSig

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
  -> (Signal dom IntermediateData, Signal dom Bool)  -- (nextData, writeDone)
multiCycleTransformerLayer layer cache layerIdx stateSig dataSig =
  (nextDataSig, writeDoneThisLayerSig)
 where
  mha = multiHeadAttention layer
  ffn = feedforwardNetwork layer

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
  writeEnableSig =
    (\st -> psStage st == Cycle4_WriteCache && psLayer st == layerIdx) <$> stateSig

  -- Per-head KV signals and write effects
  kvAllHeadsSig :: Signal dom (Vec NumKeyValueHeads (Vec HeadDimension Float, Vec HeadDimension Float))
  kvAllHeadsSig = liftA2 (zipWith (,)) (idKeys <$> dataSig) (idValues <$> dataSig)

  writeDonePerHead :: Vec NumKeyValueHeads (Signal dom Bool)
  writeDonePerHead =
    map (\hIdx ->
           writeToCacheSequence cache layerIdx hIdx
             (psSeqPos <$> stateSig)
             (((!!) <$> kvAllHeadsSig) <*> pure hIdx)
             writeEnableSig)
        indicesI

  -- all heads complete this row
  writeDoneAllHeads :: Signal dom Bool
  writeDoneAllHeads = fmap and (sequenceA writeDonePerHead)

  -- only the active layer reports done; others remain False
  writeDoneThisLayerSig :: Signal dom Bool
  writeDoneThisLayerSig =
    (\st doneH -> (psStage st == Cycle4_WriteCache && psLayer st == layerIdx) && doneH)
      <$> stateSig <*> writeDoneAllHeads

  -- Combine everything in processCycle
  baseNextDataSig = liftA4 processCycle stateSig dataSig cachedKeysSig cachedValsSig
  nextDataSig     = baseNextDataSig

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
          let
            input = idInputVec idata
            -- Compute queries for all NumQueryHeads
            queries = imap (\hIdx _ ->
              let headComp = heads mha !! hIdx
                  (q, _, _) = runSingleHeadQKV headComp input
                  (q_rot, _) = applyRotaryToHead headComp (StepCount $ fromIntegral $ psSeqPos state) (q, repeat 0)
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
                  (_, k_rot) = applyRotaryToHead headComp (StepCount $ fromIntegral $ psSeqPos state) (repeat 0, k)
              in (k_rot, v)
              ) indicesI

            (keys, values) = unzip keysAndValues
          in idata { idQueries = queries, idKeys = keys, idValues = values }

        Cycle3_ComputeAttn ->
          let
            posIdx :: Index SeqLen
            posIdx = psSeqPos state

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
                        (psSeqPos state)  -- Helpers expects Int
          in idata { idAttnOutput = attnOut }

        Cycle5_ComputeFFN ->
          let ffnOut = computeFeedForward ffn (idAttnOutput idata)
          in idata { idFFNOutput = ffnOut }

        Cycle4_WriteCache ->
          idata

-- ============================================================================
-- Multi-Cycle Full Transformer (advance on stageDone)
-- ============================================================================

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
  (outputTokenSig, readyPulseSig)
 where
  embedding    = modelEmbedding decoder
  layers       = modelLayers decoder
  inputVecSig  = embed (vocabulary embedding) <$> tokenSig

  -- Global controller: advance only when current stageDone is asserted
  advance st done = if done then nextState st else st
  procStateSig = register initialProcessingState (advance <$> procStateSig <*> stageDoneSig)

  -- Intermediate data register
  currentDataSig = register initialIntermediateData nextDataSig

  -- Load input at token start
  inputLoadedSig = liftA3 (\st cur inp ->
                      if psStage st == Cycle1_ReadCache && psLayer st == 0
                        then cur { idInputVec = inp } else cur)
                    procStateSig currentDataSig inputVecSig

  -- Run layers; gather writeDone pulses
  foldStep
    :: (Signal dom IntermediateData, Vec NumLayers (Signal dom Bool))
    -> (TransformerLayerComponent, AttentionCache dom, Index NumLayers)
    -> (Signal dom IntermediateData, Vec NumLayers (Signal dom Bool))
  foldStep (dSig, dones) (layer, cache, lidx) =
    let (newD, writeDoneL) = multiCycleTransformerLayer layer cache lidx procStateSig dSig
        dSel = liftA3 (\st old new -> if psLayer st == lidx then new else old) procStateSig dSig newD
        dones' = replace lidx writeDoneL dones
    in (dSel, dones')

  (nextDataSig, writeDoneVec) =
    foldl foldStep (inputLoadedSig, repeat (pure False)) (zip3 layers caches indicesI)

  -- Stage done signals
  -- Stage done signals
  writeDoneAny :: Signal dom Bool
  writeDoneAny = fmap or (sequenceA writeDoneVec)

  -- For now, other stages complete in a single cycle
  readDoneSig  = pure True
  qkvDoneSig   = pure True
  attnDoneSig  = pure True
  ffnDoneSig   = pure True

  -- Helper: current stage as a Signal
  stgSig :: Signal dom CycleStage
  stgSig = psStage <$> procStateSig

  -- Helper: lift (==) to Signal domain
  is :: CycleStage -> Signal dom Bool
  is c = liftA2 (==) stgSig (pure c)

  -- Select done for the current stage
  stageDoneSig :: Signal dom Bool
  stageDoneSig =
    mux (is Cycle1_ReadCache)   readDoneSig  $
    mux (is Cycle2_ComputeQKV)  qkvDoneSig   $
    mux (is Cycle3_ComputeAttn) attnDoneSig  $
    mux (is Cycle4_WriteCache)  writeDoneAny $
    mux (is Cycle5_ComputeFFN)  ffnDoneSig   $
    pure False

  -- Ready pulse at last layer FFN completion (rising edge)
  isLastFFN  = liftA2 (\st _ -> psStage st == Cycle5_ComputeFFN && psLayer st == maxBound)
                      procStateSig (pure ())
  readyPulseSig = liftA2 (\now prev -> now && not prev) isLastFFN (register False isLastFFN)

  logitsNow = transformerLogits decoder . idFFNOutput <$> nextDataSig
  latchedLogits = regEn (repeat 0) readyPulseSig logitsNow

  pick probs temp seed
    | temp == 0.0 = argMax probs
    | otherwise   = drawSample seed (map (/ temp) probs)

  outputTokenSig = regEn 0 readyPulseSig (pick <$> latchedLogits <*> temperatureSig <*> seedSig)

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
