module Model (topEntity, multiCycleTransformer, initAttentionCache, ProcessingState(..)) where

import Clash.Prelude
import qualified Prelude as P
import qualified GHC.TypeNats
import Helpers
  ( NumQueryHeads, NumKeyValueHeads, NumLayers, SeqLen, HeadDimension, ModelDim
  , Token, TransformerLayerComponent(..), TransformerDecoderComponent (..)
  , MultiHeadAttentionComponent(..), EmbeddingComponent (..)
  , runSingleHeadQKV, applyRotaryToHead, StepCount (..)
  , computeFeedForward, transformerLogits, argMax, embed
  , sampleFromProbs, liftA4, rmsNorm, softmax, matrixVectorMult, xorshift32, Temperature, Seed
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
  } deriving (Show, Generic, NFDataX)

-- One-head streaming attention core with online softmax accumulation.
-- Reads K/V from BRAM (1-cycle latency), bypasses current-row K/V, and
-- produces the head output vector when done pulses.

data AttnPhase = PhaseDot | PhaseAcc | PhaseFinalize
  deriving (Generic, NFDataX, Eq, Show)

streamHeadAttention
  :: HiddenClockResetEnable dom
  => AttentionCache dom
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom Bool
  -> Signal dom (Index SeqLen)
  -> Signal dom (Vec HeadDimension Float)
  -> Signal dom (Vec HeadDimension Float)
  -> Signal dom (Vec HeadDimension Float)
  -> ( Signal dom (Vec HeadDimension Float)
     , Signal dom Bool
     , Signal dom Bool )
streamHeadAttention cache l h startSig posSig qSig curKSig curVSig =
  (oOut, busySig, doneSig)
 where
  kQ = keyCache   cache kAddr (pure Nothing)
  vQ = valueCache cache vAddr (pure Nothing)

  startPrev  = register False startSig
  startPulse = liftA2 (\n p -> n && not p) startSig startPrev

  busySig = s where
    s    = register False next
    next = mux startPulse (pure True)
         $ mux doneSig    (pure False) s

  phase    = register PhaseDot phaseNext
  isDot    = (== PhaseDot)      <$> phase
  isAcc    = (== PhaseAcc)      <$> phase
  isFin    = (== PhaseFinalize) <$> phase

  en   = startPulse .||. busySig
  tCnt = regEn 0 en tNext
  dCnt = regEn 0 en dNext

  isLastD = (== maxBound) <$> dCnt
  isLastT = (==) <$> tCnt <*> posSig

  dPrev = register 0 dCnt

  kAddr = cacheAddr l h <$> tCnt <*> dCnt
  vAddr = cacheAddr l h <$> tCnt <*> dCnt

  kAligned = register 0 kQ
  vAligned = register 0 vQ

  useKBypass = (&&) <$> isDot <*> isLastT
  useVBypass = (&&) <$> isAcc <*> isLastT

  kBypass = (!!) <$> curKSig <*> dPrev
  vBypass = (!!) <$> curVSig <*> dPrev

  kElem = mux useKBypass kBypass kAligned
  vElem = mux useVBypass vBypass vAligned

  qElem = (!!) <$> qSig <*> dPrev

  partialDot = (+) <$> dotAcc <*> ((*) <$> qElem <*> kElem)

  dotAcc = regEn 0 en dotAccNext
  dotAccNext =
    mux isDot
      (mux isLastD (pure 0) partialDot)
      dotAcc

  dotBoundary = (&&) <$> isDot <*> isLastD
  scoreThisT  = regEn 0 dotBoundary partialDot

  -- online softmax accumulators
  mAcc   = regEn (- (1 / 0)) en mNext
  sAcc   = regEn 0      en sNext

  oNum   = regEn (repeat (0 :: Float) :: Vec HeadDimension Float) en oNumNext
  oOut   = regEn (repeat (0 :: Float) :: Vec HeadDimension Float) en oOutNext

  mNew     = max <$> mAcc <*> scoreThisT
  scaleOld = exp <$> ((-) <$> mAcc      <*> mNew)
  scaleNew = exp <$> ((-) <$> scoreThisT <*> mNew)
  sNew     = (+) <$> ((* ) <$> sAcc <*> scaleOld) <*> scaleNew

  mNext = mux dotBoundary mNew mAcc
  sNext = mux dotBoundary sNew sAcc

  oElemPrev = (!!) <$> oNum <*> dPrev
  oElemUpd  = (\o v a b -> o * b + v * a) <$> oElemPrev <*> vElem <*> scaleNew <*> scaleOld

  oNumNext =
    mux isAcc
      (replace <$> dPrev <*> oElemUpd <*> oNum)
      oNum

  oOutDivElem = (/ ) <$> oElemPrev <*> sNew
  oOutNext =
    mux isFin
      (replace <$> dPrev <*> oOutDivElem <*> oOut)
      oOut

  phaseNext =
    liftA3
      (\ph dLast tLast ->
         case ph of
           PhaseDot      -> if dLast then PhaseAcc else PhaseDot
           PhaseAcc      -> if dLast then (if tLast then PhaseFinalize else PhaseDot) else PhaseAcc
           PhaseFinalize -> if dLast then PhaseDot else PhaseFinalize)
      phase isLastD isLastT

  dNext =
    liftA3
      (\ph d dLast ->
         case ph of
           PhaseDot      -> if dLast then 0 else succ d
           PhaseAcc      -> if dLast then 0 else succ d
           PhaseFinalize -> if dLast then 0 else succ d)
      phase dCnt isLastD

  tNext =
    liftA4
      (\ph t dLast tLast ->
         case ph of
           PhaseDot      -> t
           PhaseAcc      -> if dLast then (if tLast then 0 else succ t) else t
           PhaseFinalize -> t)
      phase tCnt isLastD isLastT

  doneSig = register False $ (&&) <$> isFin <*> isLastD

initialProcessingState :: ProcessingState
initialProcessingState = ProcessingState
  { psStage = Cycle1_ReadCache
  , psLayer = 0
  , psSeqPos = 0
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
              }
      else st { psStage = Cycle1_ReadCache
              , psLayer = succ (psLayer st)
              }

-- ============================================================================
-- Intermediate data storage
-- ============================================================================

data IntermediateData = IntermediateData
  { idInputVec   :: Vec ModelDim Float
  , idQueries    :: Vec NumQueryHeads (Vec HeadDimension Float)
  , idKeys       :: Vec NumKeyValueHeads (Vec HeadDimension Float)
  , idValues     :: Vec NumKeyValueHeads (Vec HeadDimension Float)
  , idAttnOutput :: Vec ModelDim Float
  , idFFNOutput  :: Vec ModelDim Float
  } deriving (Show, Generic, NFDataX)

initialIntermediateData :: IntermediateData
initialIntermediateData = IntermediateData
  { idInputVec   = repeat 0
  , idQueries    = repeat (repeat 0)
  , idKeys       = repeat (repeat 0)
  , idValues     = repeat (repeat 0)
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

-- Micro-FSM for Cycle3: iterate t = 0..pos (one per cycle).
-- done pulses one cycle after the last t was issued.
cycle3AttnMicro
  :: HiddenClockResetEnable dom
  => Signal dom Bool                 -- start
  -> Signal dom (Index SeqLen)       -- pos
  -> ( Signal dom (Index SeqLen)     -- t counter (valid while busy)
     , Signal dom Bool               -- busy
     , Signal dom Bool               -- done pulse (1-cycle)
     )
cycle3AttnMicro startSig posSig = (tCnt, busySig, doneSig)
 where
  startPrev  = register False startSig
  startPulse = liftA2 (\now prev -> now && not prev) startSig startPrev

  busySig = sig where
    sig = register False next
    next = liftA3 (\b sp dn -> (b && not dn) || sp) sig startPulse doneSig

  tCnt = regEn 0 (startPulse .||. busySig) tNext

  atLast = (==) <$> tCnt <*> posSig

  tNext = mux busySig
             (liftA2 (\t lastT -> if lastT then 0 else succ t) tCnt atLast)
             tCnt

  issuedLast = (&&) <$> busySig <*> atLast
  doneSig    = register False issuedLast

-- ============================================================================
-- Multi-Cycle Transformer Layer
-- ============================================================================

multiCycleTransformerLayer
  :: forall dom
   . (HiddenClockResetEnable dom)
  => TransformerLayerComponent
  -> AttentionCache dom
  -> Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom IntermediateData
  -> ( Signal dom IntermediateData
     , Signal dom Bool     -- writeDone (Cycle4)
     , Signal dom Bool     -- attnDone  (Cycle3)
     , Signal dom IntermediateData) -- commitCycle3 (gated write-back)
multiCycleTransformerLayer layer cache layerIdx stateSig dataSig =
  (nextDataSig, writeDoneThisLayerSig, attnDoneThisLayerSig, commitCycle3)
 where
  mha = multiHeadAttention layer
  ffn = feedforwardNetwork layer

  -- Stage enables (only when this layer is active)
  stgEq s = liftA2 (\st _ -> psStage st == s && psLayer st == layerIdx) stateSig (pure ())
  attStart = stgEq Cycle3_ComputeAttn
  wrStart  = stgEq Cycle4_WriteCache

  posSig = psSeqPos <$> stateSig

  -- Attention micro-FSM (unchanged)
  (_tCnt, _attBusy, attnDoneThisLayerSig) = cycle3AttnMicro attStart posSig

  -- Streaming attention, one instance per query head
  -- kvIdx = qIdx `div` headsPerGroup
  headsPerGroupI :: Int
  headsPerGroupI = natToNum @NumQueryHeads `P.div` natToNum @NumKeyValueHeads

  kvIdxOf :: Index NumQueryHeads -> Index NumKeyValueHeads
  kvIdxOf q =
    let qi  = fromEnum q
        idx = qi `P.div` headsPerGroupI
        hi  = natToNum @NumKeyValueHeads - 1
    in toEnum (max 0 (min hi idx))

  -- For each query head:
  --   start = attStart
  --   pos   = psSeqPos
  --   q     = (idQueries <$> dataSig) !! qIdx
  --   curK  = (idKeys    <$> dataSig) !! kvIdx
  --   curV  = (idValues  <$> dataSig) !! kvIdx
  headsOutSigsVec :: Vec NumQueryHeads (Signal dom (Vec HeadDimension Float))
  headsOutSigsVec =
    imap
      (\qIdx _ ->
         let kvIx  = kvIdxOf qIdx
             qS    = ((!! qIdx) P.. idQueries P.<$> dataSig)
             curKS = ((!! kvIx) P.. idKeys P.<$> dataSig)
             curVS = ((!! kvIx) P.. idValues P.<$> dataSig)
             (headOut, _busyH, _doneH) =
               streamHeadAttention cache layerIdx kvIx attStart posSig qS curKS curVS
         in headOut
      )
      indicesI
  -- Vec NumQueryHeads (Signal (Vec ModelDim Float))
  perHeadProjectedVec :: Vec NumQueryHeads (Signal dom (Vec ModelDim Float))
  perHeadProjectedVec =
    zipWith (\wo h -> matrixVectorMult wo <$> h) (mWo mha) headsOutSigsVec

  -- Signal (Vec NumQueryHeads (Vec ModelDim Float))
  perHeadProjectedSig :: Signal dom (Vec NumQueryHeads (Vec ModelDim Float))
  perHeadProjectedSig = sequenceA perHeadProjectedVec

  -- Signal (Vec ModelDim Float)
  attnSumSig :: Signal dom (Vec ModelDim Float)
  attnSumSig = fmap (foldl1 (zipWith (+))) perHeadProjectedSig

  -- Residual: x + sum_h W_O h_out
  -- Signal (Vec ModelDim Float)
  attnOutSig :: Signal dom (Vec ModelDim Float)
  attnOutSig = (zipWith (+) P.. idInputVec P.<$> dataSig) <*> attnSumSig

  -- Commit idAttnOutput only when this layer’s attention finishes
  nextDataSig :: Signal dom IntermediateData
  nextDataSig =
    liftA4
      (\st cur attOut done ->
         if psLayer st == layerIdx && psStage st == Cycle3_ComputeAttn && done
           then cur { idAttnOutput = attOut }
           else cur)
      stateSig baseNextDataSig attnOutSig attnDoneThisLayerSig

  -- Writes to KV cache
  writeEnableSig = wrStart
  kvAllHeadsSig  = liftA2 zip (idKeys <$> dataSig) (idValues <$> dataSig)
  writeDonePerHead =
    map (\hIdx ->
           writeToCacheSequence cache layerIdx hIdx
             posSig
             (((!!) <$> kvAllHeadsSig) <*> pure hIdx)
             writeEnableSig)
        indicesI
  writeDoneThisLayerSig = (\st d -> psStage st == Cycle4_WriteCache && psLayer st == layerIdx && d)
                          <$> stateSig <*> fmap and (sequenceA writeDonePerHead)

  commitCycle3 :: Signal dom IntermediateData
  commitCycle3 =
    liftA4
      (\st cur attOut done ->
        if psLayer st == layerIdx && psStage st == Cycle3_ComputeAttn && done
          then cur { idAttnOutput = attOut }
          else cur)
      stateSig dataSig attnOutSig attnDoneThisLayerSig
  
  -- Cycle1 no-op; QKV/FFN unchanged;
  baseNextDataSig :: Signal dom IntermediateData
  baseNextDataSig = liftA2 processCycle stateSig dataSig

  processCycle :: ProcessingState -> IntermediateData -> IntermediateData
  processCycle st idata
    | psLayer st /= layerIdx = idata
    | otherwise =
        case psStage st of
          Cycle1_ReadCache -> idata
          Cycle2_ComputeQKV ->
            let
              xHat = rmsNorm (idInputVec idata) (rmsAtt mha)
              queries = imap (\hIdx _ ->
                let headComp = heads mha !! hIdx
                    (q, _, _) = runSingleHeadQKV headComp xHat
                    (q_rot, _) = applyRotaryToHead headComp
                                    (StepCount $ fromIntegral $ psSeqPos st)
                                    (q, repeat 0)
                in q_rot) indicesI
              headsPerGroupI' :: Int
              headsPerGroupI' = natToNum @NumQueryHeads `P.div` natToNum @NumKeyValueHeads
              toIndexClamped :: forall n. KnownNat n => Int -> Index n
              toIndexClamped i =
                let hi = natToNum @n - 1
                    j  = max 0 (min hi i)
                in toEnum j
              keysAndValues =
                imap (\hIdx _ ->
                  let qIdx :: Index NumQueryHeads
                      qIdx = toIndexClamped @NumQueryHeads (fromEnum hIdx * headsPerGroupI')
                      headComp = heads mha !! qIdx
                      (_, k, v) = runSingleHeadQKV headComp xHat
                      (_, k_rot) = applyRotaryToHead headComp
                                    (StepCount $ fromIntegral $ psSeqPos st)
                                    (repeat 0, k)
                  in (k_rot, v)) indicesI
              (keys, values) = unzip keysAndValues
            in idata { idQueries = queries, idKeys = keys, idValues = values }
          Cycle3_ComputeAttn -> idata
          Cycle4_WriteCache  -> idata
          Cycle5_ComputeFFN  ->
            let ffnOut = computeFeedForward ffn (idAttnOutput idata)
            in idata { idFFNOutput = ffnOut }

-- ============================================================================
-- Multi-Cycle Full Transformer (advance on stageDone)
-- ============================================================================

multiCycleTransformer
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> Vec NumLayers (AttentionCache dom)
  -> Signal dom Token
  -> Signal dom Temperature
  -> Signal dom Seed
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

  -- Load the layer input at Cycle1:
  -- - layer 0 gets the token embedding
  -- - layers >0 get the previous layer's FFN output
  inputLoadedSig =
    liftA3
      (\st cur inp ->
         if psStage st == Cycle1_ReadCache
           then if psLayer st == 0
                  -- layer 0: use embedding
                  then cur { idInputVec = inp }
                  -- layer >0: feed forward previous layer's output
                  else cur { idInputVec = idFFNOutput cur }
           else cur)
      procStateSig currentDataSig inputVecSig

  -- Run layers; gather per-stage done pulses
  foldStep
    :: ( Signal dom IntermediateData
      , Vec NumLayers (Signal dom Bool)
      , Vec NumLayers (Signal dom Bool))
    -> (TransformerLayerComponent, AttentionCache dom, Index NumLayers)
    -> ( Signal dom IntermediateData
      , Vec NumLayers (Signal dom Bool)
      , Vec NumLayers (Signal dom Bool))
  foldStep (dSig, wD, aD) (layer, cache, lidx) =
    let (newD, wDoneL, aDoneL, commitC3) =
          multiCycleTransformerLayer layer cache lidx procStateSig dSig

        dSel = liftA4
          (\st old new commit ->
             if psLayer st == lidx
               then if psStage st == Cycle3_ComputeAttn
                      then commit   -- gated write-back on attnDone
                      else new
               else old)
          procStateSig dSig newD commitC3
    in ( dSel
       , replace lidx wDoneL wD
       , replace lidx aDoneL aD)

  (nextDataSig, writeDoneVec, attnDoneVec) =
    foldl foldStep (inputLoadedSig, repeat (pure False), repeat (pure False))
                   (zip3 layers caches indicesI)

  writeDoneAny = fmap or (sequenceA writeDoneVec)
  attnDoneAny  = fmap or (sequenceA attnDoneVec)

  -- Stage done selection
  stgSig = psStage <$> procStateSig
  is c   = (== c) <$> stgSig

  stageDoneSig =
    mux (is Cycle1_ReadCache)   (pure True)  $
    mux (is Cycle2_ComputeQKV)  (pure True)  $
    mux (is Cycle3_ComputeAttn) attnDoneAny  $
    mux (is Cycle4_WriteCache)  writeDoneAny $
    mux (is Cycle5_ComputeFFN)  (pure True)  $
    pure False

  -- Ready pulse at last layer FFN completion (rising edge)
  isLastFFN  = liftA2 (\st _ -> psStage st == Cycle5_ComputeFFN && psLayer st == maxBound)
                      procStateSig (pure ())
  readyPulseSig = liftA2 (\now prev -> now && not prev) isLastFFN (register False isLastFFN)

  -- existing logits
  logitsNow      = transformerLogits decoder . idFFNOutput <$> nextDataSig
  latchedLogits  = regEn (repeat 0) readyPulseSig logitsNow

  -- PRNG state: seed is mixed in on the first pulse; otherwise advance each pulse
  firstPulseSig :: Signal dom Bool
  firstPulseSig = regEn True readyPulseSig (pure False)

  mixedSeedSig :: Signal dom (Unsigned 32)
  mixedSeedSig = (`xor` 0x9E3779B9) <$> seedSig

  prngStateSig :: Signal dom (Unsigned 32)
  prngStateSig =
    let nextVal = mux firstPulseSig (xorshift32 <$> mixedSeedSig)
                                    (xorshift32 <$> prngStateSig)
    in regEn 2463534242 readyPulseSig nextVal  -- non-zero default

  -- Convert to Float in [0,1). Use top 24 bits as mantissa fraction.
  uniform01Sig :: Signal dom Float
  uniform01Sig = (/ 16777216.0) . fromIntegral . (`shiftR` 8) <$> prngStateSig

  -- Sampling on pulse:
  sampledTokenOnPulse :: Signal dom (Unsigned 32)
  sampledTokenOnPulse =
    liftA3
      (\temperature logs u ->
         if temperature <= 0.0
           then argMax logs
           else
             let probs = softmax temperature logs
             in sampleFromProbs u probs
      )
      temperatureSig latchedLogits uniform01Sig

  outputTokenSig = regEn 0 readyPulseSig sampledTokenOnPulse

-- ============================================================================
-- Top Entity
-- ============================================================================

topEntity
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> Signal dom Token
  -> Signal dom Temperature
  -> Signal dom Seed
  -> (Signal dom Token, Signal dom Bool)
topEntity decoder = multiCycleTransformer decoder (repeat initAttentionCache)
