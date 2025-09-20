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
import Data.Maybe (isJust)

-- ============================================================================
-- Banked KV RAMs (one bank per KV head). Each bank has two read/write ports.
-- For your config (NumQueryHeads / NumKeyValueHeads = 2), two Q heads share a bank.
-- ============================================================================

-- Turn (addr, maybe write (addr,data)) into a RamOp stream
toOp
  :: NFDataX a => Signal dom (Index n)
  -> Signal dom (Maybe (Index n, a))
  -> Signal dom (RamOp n a)
toOp addrSig wrSig =
  mux (isJust <$> wrSig) (uncurry RamWrite . fromJustX <$> wrSig) (RamRead <$> addrSig)

-- Bank geometry
type BankDepth = SeqLen GHC.TypeNats.* HeadDimension
type BankAddr  = Index BankDepth

bankAddr :: Index SeqLen -> Index HeadDimension -> BankAddr
bankAddr s d = toEnum (fromIntegral d + fromIntegral s * natToNum @HeadDimension)

-- Per-bank K/V dual-port “runner” (instantiate RAM once, feed both ports)
type TdprRunner dom n a =
       ( Signal dom (Index n)               -- Port A address
       , Signal dom (Maybe (Index n, a)) )  -- Port A write data (optional)
    -> ( Signal dom (Index n)               -- Port B address
       , Signal dom (Maybe (Index n, a)) )  -- Port B write data (optional)
    -> ( Signal dom a                      -- qA – read data from port A
       , Signal dom a )                    -- qB – read data from port B

mkTdprRunner :: (HiddenClockResetEnable dom, KnownNat n, NFDataX a) => TdprRunner dom n a
mkTdprRunner (addrA, wrA) (addrB, wrB) = trueDualPortBlockRam (toOp addrA wrA) (toOp addrB wrB)

data KvBank dom = KvBank
  { runK :: TdprRunner dom BankDepth Float
  , runV :: TdprRunner dom BankDepth Float
  }

mkKvBank :: HiddenClockResetEnable dom => KvBank dom
mkKvBank = KvBank { runK = mkTdprRunner, runV = mkTdprRunner }

newtype KVRamOwner dom = KVRamOwner
  { kvBanks :: Vec NumKeyValueHeads (KvBank dom)
  }

mkKVRamOwner :: HiddenClockResetEnable dom => KVRamOwner dom
mkKVRamOwner = KVRamOwner { kvBanks = map (const mkKvBank) indicesI }

-- Writer sequencer for one bank: generates (addr, write) streams and a done pulse.
writeSequencer
  :: HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom (Index SeqLen)
  -> Signal dom (Vec HeadDimension Float, Vec HeadDimension Float)
  -> ( Signal dom BankAddr
     , Signal dom (Maybe (BankAddr, Float))  -- K write
     , Signal dom (Maybe (BankAddr, Float))  -- V write
     , Signal dom Bool)                      -- done (1-cycle on last d)
writeSequencer enSig seqPosSig kvSig = (addr, kWr, vWr, doneSig)
 where
  dCnt   = register 0 nextD
  nextD  = mux enSig (P.fmap (\d -> if d == maxBound then 0 else succ d) dCnt) (pure 0)
  lastD  = (== maxBound) <$> dCnt
  doneSig = (&&) <$> enSig <*> lastD

  kElem  = (\(k,_) d -> k !! d) <$> kvSig <*> dCnt
  vElem  = (\(_,v) d -> v !! d) <$> kvSig <*> dCnt
  addr   = bankAddr <$> seqPosSig <*> dCnt

  kWr    = mux enSig (Just <$> bundle (addr, kElem)) (pure Nothing)
  vWr    = mux enSig (Just <$> bundle (addr, vElem)) (pure Nothing)

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

-- One-head streaming attention with externalized memory interface.
-- Caller must connect kAddr/vAddr to a RAM and feed kQ/vQ back (synchronous, 1-cycle).
streamHeadAttentionAddrIO
  :: HiddenClockResetEnable dom
  => Signal dom Bool                         -- start
  -> Signal dom (Index SeqLen)               -- pos
  -> Signal dom (Vec HeadDimension Float)    -- q
  -> Signal dom (Vec HeadDimension Float)    -- curK (bypass)
  -> Signal dom (Vec HeadDimension Float)    -- curV (bypass)
  -> Signal dom Float                        -- kQ (from RAM)
  -> Signal dom Float                        -- vQ (from RAM)
  -> ( Signal dom BankAddr                   -- addr (use for both K and V)
     , Signal dom (Vec HeadDimension Float)  -- head output
     , Signal dom Bool                       -- busy
     , Signal dom Bool )                     -- done (1-cycle)
streamHeadAttentionAddrIO startSig posSig qSig curKSig curVSig kQin vQin =
  (addr, oOut, busySig, doneSig)
 where
  -- Phases and enables (unchanged)
  phase    = register PhaseDot phaseNext
  isDot    = (== PhaseDot)      <$> phase
  isAcc    = (== PhaseAcc)      <$> phase
  isFin    = (== PhaseFinalize) <$> phase

  startPrev  = register False startSig
  startPulse = liftA2 (\n p -> n && not p) startSig startPrev
  en         = startPulse .||. busySig

  tCnt = regEn 0 en tNext
  dCnt = regEn 0 en dNext

  isLastD = (== maxBound) <$> dCnt
  isLastT = (==) <$> tCnt <*> posSig
  dPrev   = register 0 dCnt

  -- External RAM addressing
  addr = bankAddr <$> tCnt <*> dCnt

  -- Synchronous read alignment
  kAligned = register 0 kQin
  vAligned = register 0 vQin

  -- Bypass at t == pos (current row)
  useKBypass = (&&) <$> isDot <*> isLastT
  useVBypass = (&&) <$> isAcc <*> isLastT
  kBypass = (!!) <$> curKSig <*> dPrev
  vBypass = (!!) <$> curVSig <*> dPrev
  kElem = mux useKBypass kBypass kAligned
  vElem = mux useVBypass vBypass vAligned
  qElem = (!!) <$> qSig     <*> dPrev

  -- Dot-product accumulation and online softmax (unchanged)
  invSqrtHd :: Float
  invSqrtHd = 1.0 / sqrt (snatToNum (SNat @HeadDimension) :: Float)

  dotAcc = regEn 0 en dotAccNext
  partialDot = (+) <$> dotAcc <*> ((*) <$> qElem <*> kElem)
  dotBoundary = (&&) <$> isDot <*> isLastD

  dotAccNext =
    mux startPulse (pure 0) $
    mux isDot (mux isLastD (pure 0) partialDot) dotAcc

  scoreThisT  = regEn 0 dotBoundary (partialDot * pure invSqrtHd)

  mAcc = regEn (-(1/0)) en mNext
  sAcc = regEn 0        en sNext
  oNum = regEn (repeat (0 :: Float) :: Vec HeadDimension Float) en oNumNext
  oOut = regEn (repeat (0 :: Float) :: Vec HeadDimension Float) en oOutNext

  mNew     = max <$> mAcc <*> scoreThisT
  scaleOld = exp <$> ((-) <$> mAcc      <*> mNew)
  scaleNew = exp <$> ((-) <$> scoreThisT <*> mNew)
  sNew     = (+) <$> ((* ) <$> sAcc <*> scaleOld) <*> scaleNew

  oElemPrev   = (!!) <$> oNum <*> dPrev
  oElemUpd    = (\o v a b -> o * b + v * a) <$> oElemPrev <*> vElem <*> scaleNew <*> scaleOld
  oOutDivElem = (/ ) <$> oElemPrev <*> sNew

  mNext =
    mux startPulse (pure (-(1/0))) $
    mux dotBoundary mNew mAcc

  sNext =
    mux startPulse (pure 0) $
    mux dotBoundary sNew sAcc

  oNumNext =
    mux startPulse (pure (repeat 0)) $
    mux isAcc (replace <$> dPrev <*> oElemUpd <*> oNum) oNum

  oOutNext =
    mux startPulse (pure (repeat 0)) $
    mux isFin (replace <$> dPrev <*> oOutDivElem <*> oOut) oOut

  busySig = s where
    s    = register False next
    next = mux startPulse (pure True) $ mux doneSig (pure False) s

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
      then st { psStage  = Cycle1_ReadCache
              , psLayer  = 0
              , psSeqPos = if psSeqPos st == maxBound then 0 else succ (psSeqPos st)
              }
      else st { psStage  = Cycle1_ReadCache
              , psLayer  = succ (psLayer st)
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

data AttentionCache dom = AttentionCache
  { keyCache   :: Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float
  , valueCache :: Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float
  }

initAttentionCache :: forall dom. HiddenClockResetEnable dom => AttentionCache dom
initAttentionCache = AttentionCache
  { keyCache = blockRam (replicate (SNat @CacheDepth) 0)
  , valueCache = blockRam (replicate (SNat @CacheDepth) 0)
  }

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

getQ :: Signal dom IntermediateData -> Index NumQueryHeads -> Signal dom (Vec HeadDimension Float)
getQ dataSig h = fmap (\idata -> idQueries idata !! h) dataSig

getK :: Signal dom IntermediateData -> Index NumKeyValueHeads -> Signal dom (Vec HeadDimension Float)
getK dataSig kv = fmap (\idata -> idKeys idata !! kv) dataSig

getV :: Signal dom IntermediateData -> Index NumKeyValueHeads -> Signal dom (Vec HeadDimension Float)
getV dataSig kv = fmap (\idata -> idValues idata !! kv) dataSig


multiCycleTransformerLayer
  :: HiddenClockResetEnable dom
  => TransformerLayerComponent
  -> KVRamOwner dom
  -> Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom IntermediateData
  -> ( Signal dom IntermediateData
     , Signal dom Bool     -- writeDone (Cycle4)
     , Signal dom Bool     -- attnDone  (Cycle3)
     , Signal dom IntermediateData) -- commitCycle3 (gated write-back)
multiCycleTransformerLayer layer owner layerIdx stateSig dataSig =
  (nextDataSig, writeDoneThisLayerSig, attnDoneThisLayerSig, commitCycle3)
 where
  mha = multiHeadAttention layer
  ffn = feedforwardNetwork layer

  -- Stage enables (only when this layer is active)
  stgEq s = liftA2 (\st _ -> psStage st == s && psLayer st == layerIdx) stateSig (pure ())
  attStart = stgEq Cycle3_ComputeAttn

  posSig = psSeqPos <$> stateSig

  -- Attention micro-FSM (unchanged)
  (_tCnt, _attBusy, _) = cycle3AttnMicro attStart posSig

  doneAllHeadsSig = fmap P.and (sequenceA headDoneSigsVec)

  doneAllPrev = register False doneAllHeadsSig
  attnDoneThisLayerSig = liftA2 (\n p -> n && not p) doneAllHeadsSig doneAllPrev

  banksThisLayer = kvBanks owner

  -- Streaming attention, one instance per query head

  isC3 = stgEq Cycle3_ComputeAttn
  isC4 = stgEq Cycle4_WriteCache

  -- For each KV bank, wire two heads (A and B) in Cycle3, or the writer on port B in Cycle4.
  -- Collect per-head outputs in a Vec NumQueryHeads of Signals
  -- Vec of per-head outputs and per-head done flags

  (headsOutSigsVec, headDoneSigsVec) =
    let
      initOut :: Vec NumQueryHeads (Signal dom (Vec HeadDimension Float))
      initOut  = repeat (pure (repeat 0))
      initDone :: Vec NumQueryHeads (Signal dom Bool)
      initDone = repeat (pure (False :: Bool))
      headsPerGroupI :: Int
      headsPerGroupI = natToNum @NumQueryHeads `P.div` natToNum @NumKeyValueHeads
      qHi :: Int
      qHi = natToNum @NumQueryHeads - 1

      fillOneBank (outAcc, doneAcc) kv =
        let
          bank  = banksThisLayer !! kv
          kRun  = runK bank
          vRun  = runV bank

          base  = fromEnum kv * headsPerGroupI
          q0i   = toEnum (min qHi base)         :: Index NumQueryHeads
          hasQ1 = base + 1 <= qHi
          q1i   = if hasQ1 then toEnum (base + 1) else q0i

          q0S  = getQ dataSig q0i
          kCur = getK dataSig kv
          vCur = getV dataSig kv
          q1S  = if hasQ1 then getQ dataSig q1i else pure (repeat 0)

          (addr0, out0, _busy0, done0) = streamHeadAttentionAddrIO attStart posSig q0S kCur vCur kQ0 vQ0
          (addr1, out1, _busy1, done1) = streamHeadAttentionAddrIO attStart posSig q1S kCur vCur kQ1 vQ1

          kvPairSig                   = liftA2 (,) kCur vCur
          (wrAddr, kWr, vWr, _wrDone) = writeSequencer isC4 posSig kvPairSig

          addrA = addr0
          wrA   = pure Nothing
          addrB = mux isC3 addr1 wrAddr
          wrKB  = mux isC3 (pure Nothing) kWr
          wrVB  = mux isC3 (pure Nothing) vWr

          (kQA, kQB) = kRun (addrA, wrA) (addrB, wrKB)
          (vQA, vQB) = vRun (addrA, wrA) (addrB, wrVB)

          kQ0 = kQA; vQ0 = vQA
          kQ1 = kQB; vQ1 = vQB

          outAcc0  = replace q0i out0 outAcc
          doneAcc0 = replace q0i done0 doneAcc
          outAcc1  = if hasQ1 then replace q1i out1 outAcc0 else outAcc0
          doneAcc1 = if hasQ1 then replace q1i done1 doneAcc0 else doneAcc0
        in (outAcc1, doneAcc1)
    in P.foldl fillOneBank (initOut, initDone) indicesI

  -- Vec NumQueryHeads (Signal (Vec ModelDim Float))
  perHeadProjectedVec =
    zipWith (\wo h -> matrixVectorMult wo <$> h) (mWo mha) headsOutSigsVec

  -- Signal (Vec NumQueryHeads (Vec ModelDim Float))
  perHeadProjectedSig = sequenceA perHeadProjectedVec

  -- Signal (Vec ModelDim Float)
  attnSumSig = fmap (foldl1 (zipWith (+))) perHeadProjectedSig

  -- Residual: x + sum_h W_O h_out
  -- Signal (Vec ModelDim Float)
  attnOutSig = (zipWith (+) P.. idInputVec P.<$> dataSig) <*> attnSumSig

  -- Commit idAttnOutput only when this layer’s attention finishes
  nextDataSig =
    liftA4
      (\st cur attOut done ->
         if psLayer st == layerIdx && psStage st == Cycle3_ComputeAttn && done
           then cur { idAttnOutput = attOut }
           else cur)
      stateSig baseNextDataSig attnOutSig attnDoneThisLayerSig

  -- Collect wrDone from the same writeSequencer used to drive the RAMs
  wrDoneVec = imap (\kv _ ->
                      let kvPair = liftA2 (,) (getK dataSig kv) (getV dataSig kv)
                          (_,_,_,wd) = writeSequencer isC4 posSig kvPair
                      in wd) (repeat ())
                      
  writeDoneThisLayerSig =
    (\st d -> psStage st == Cycle4_WriteCache && psLayer st == layerIdx && d)
      <$> stateSig <*> fmap or (sequenceA wrDoneVec)

  commitCycle3 =
    liftA4
      (\st cur attOut done ->
        if psLayer st == layerIdx && psStage st == Cycle3_ComputeAttn && done
          then cur { idAttnOutput = attOut }
          else cur)
      stateSig dataSig attnOutSig attnDoneThisLayerSig

  -- Cycle1 no-op; QKV/FFN unchanged;
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
  -> Vec NumLayers (KVRamOwner dom)
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
    -> (TransformerLayerComponent, KVRamOwner dom, Index NumLayers)
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
         if temperature <= 0.0 then argMax logs
         else let probs = softmax temperature logs in sampleFromProbs u probs)
      temperatureSig logitsNow uniform01Sig

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
topEntity decoder = multiCycleTransformer decoder (repeat mkKVRamOwner)
