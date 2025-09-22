module Model.Attention
  ( AttnPhase(..)
  , streamHeadAttentionAddrIO
  ) where

import Clash.Prelude

import Helpers ( HeadDimension, SeqLen, liftA4, liftA5 )
import Model.Cache (bankAddr)
import Model.Types (BankAddr)

-- One-head streaming attention with online softmax, aligned to 1-cycle BRAM latency.
-- Phases:
--   Dot:  compute score_t = (q · k_t) / sqrt(d) for t = 0..pos
--   Acc:  update running m,s and numerator oNum with scaleOld/scaleNew for this t
--   Fin:  emit head output o = oNum / s for all d
--
-- We use a 1-cycle warm-up at the start of Dot and Acc so that BRAM outputs
-- (kQin/vQin) align with the element index we accumulate. Bypass of current-row
-- K/V is aligned using (tPrev,dPrev).
data AttnPhase = PhaseDot | PhaseAcc | PhaseFinalize
  deriving (Generic, NFDataX, Eq, Show)

streamHeadAttentionAddrIO
  :: HiddenClockResetEnable dom
  => Signal dom Bool                         -- start
  -> Signal dom (Index SeqLen)               -- pos
  -> Signal dom (Vec HeadDimension Float)    -- q
  -> Signal dom (Vec HeadDimension Float)    -- curK (bypass at t==pos)
  -> Signal dom (Vec HeadDimension Float)    -- curV (bypass at t==pos)
  -> Signal dom Float                        -- kQ (from RAM, 1-cycle)
  -> Signal dom Float                        -- vQ (from RAM, 1-cycle)
  -> ( Signal dom BankAddr                   -- addr (use for both K and V)
     , Signal dom (Vec HeadDimension Float)  -- head output
     , Signal dom Bool                       -- busy
     , Signal dom Bool )                     -- done (1-cycle pulse)
streamHeadAttentionAddrIO startSig posSig qSig curKSig curVSig kQin vQin =
  (addr, oOut, busySig, donePulse)
 where
  -- phase control + warm-up (align 1-cycle BRAM)
  phase    = register PhaseDot phaseNext
  isDot    = (== PhaseDot)      <$> phase
  isAcc    = (== PhaseAcc)      <$> phase
  isFin    = (== PhaseFinalize) <$> phase

  startPrev  = register False startSig
  startPulse = liftA2 (\n p -> n && not p) startSig startPrev

  warm     = regEn False en warmNext
  warmNext = mux startPulse (pure False) $
             mux (isDot .||. isAcc) (pure True) warm -- stay warm after first tick in Dot/Acc
  en = startPulse .||. busySig

  busySig = s where
    s    = register False next
    next = mux startPulse (pure True) $
           mux donePulse   (pure False) s

  -- counters
  tCnt  = regEn 0 en tNext
  dCnt  = regEn 0 en dNext
  tPrev = register 0 tCnt
  dPrev = register 0 dCnt
  isLastD_now = (== maxBound) <$> dCnt
  isLastT_now = (==) <$> tCnt <*> posSig

  -- addressing (issue address for current (tCnt,dCnt); data valid next cycle)
  addr = bankAddr <$> tCnt <*> dCnt

  -- element selection, aligned to previous address
  qElem_raw = (!!) <$> qSig   <*> dPrev
  kBypass   = (!!) <$> curKSig <*> dPrev
  vBypass   = (!!) <$> curVSig <*> dPrev

  useKBypass = liftA3 (\w pd pt -> w && pd && pt) warm (pure True) ((==) <$> tPrev <*> posSig)
  useVBypass = useKBypass -- same t test; used in Acc

  kElem = mux useKBypass kBypass kQin
  vElem = mux useVBypass vBypass vQin
  qElem = qElem_raw

  -- scaling
  invSqrtHd :: Float
  invSqrtHd = 1.0 / sqrt (snatToNum (SNat @HeadDimension) :: Float)

  -- online softmax accumulators across timesteps
  mAcc = regEn (-(1/0)) en mAccNext
  sAcc = regEn 0        en sAccNext

  -- per-t cached scales (written at end of Dot, used across all d in Acc)
  scaleOldReg = regEn 1 en scaleOldNext
  scaleNewReg = regEn 0 en scaleNewNext

  -- dot partial across d for current t
  dotAcc = regEn 0 en dotAccNext

  canAccumulate = warm .&&. isDot
  dotAccNext =
    mux startPulse (pure 0)
    $ mux isDot
        (mux canAccumulate (dotAcc + (qElem * kElem)) dotAcc)
        dotAcc

  -- boundary at end of Dot: when we just accumulated dPrev == maxBound (i.e., now dCnt==0 next cycle)
  dotBoundary =
    liftA3
      (\ph w dLast -> ph == PhaseDot && w && dLast)
      phase warm isLastD_now

  scoreThisT = (dotAcc + (qElem * kElem)) * pure invSqrtHd
  mNew       = max <$> mAcc <*> scoreThisT
  scaleOld   = exp <$> ((-) <$> mAcc      <*> mNew)
  scaleNew   = exp <$> ((-) <$> scoreThisT <*> mNew)
  sNew       = (+) <$> ((* ) <$> sAcc <*> scaleOld) <*> scaleNew

  -- latch m/s/scales at t boundary
  mAccNext      = mux startPulse (pure (-(1/0)))
                 $ mux dotBoundary mNew mAcc
  sAccNext      = mux startPulse (pure 0)
                 $ mux dotBoundary sNew sAcc
  scaleOldNext  = mux dotBoundary scaleOld scaleOldReg
  scaleNewNext  = mux dotBoundary scaleNew scaleNewReg

  -- numerator across d: oNum[d] = oNum[d]*scaleOld + v[d]*scaleNew
  oNum     = regEn (repeat 0 :: Vec HeadDimension Float) en oNumNext
  oElemPrev = (!!) <$> oNum <*> dPrev
  oElemUpd  = (\o v a b -> o * b + v * a) <$> oElemPrev <*> vElem <*> scaleNewReg <*> scaleOldReg

  canAccumulateAcc = warm .&&. isAcc
  oNumNext =
    mux startPulse (pure (repeat 0))
    $ mux isAcc
        (mux canAccumulateAcc (replace <$> dPrev <*> oElemUpd <*> oNum) oNum)
        oNum

  -- finalize: write oOut[d] = oNum[d] / sAcc
  oOut     = regEn (repeat 0 :: Vec HeadDimension Float) en oOutNext
  oOutDiv  = (/ ) <$> oElemPrev <*> sAcc
  canFinalize = warm .&&. isFin
  oOutNext =
    mux startPulse (pure (repeat 0))
    $ mux isFin
        (mux canFinalize (replace <$> dPrev <*> oOutDiv <*> oOut) oOut)
        oOut

  -- phase transitions and counters
  phaseNext =
    liftA5
      (\ph w dLast tLast dotBdy ->
        case ph of
          PhaseDot      -> if dotBdy then PhaseAcc else PhaseDot
          PhaseAcc      -> if w && dLast then (if tLast then PhaseFinalize else PhaseDot) else PhaseAcc
          PhaseFinalize -> if w && dLast then PhaseDot else PhaseFinalize)
      phase warm isLastD_now isLastT_now dotBoundary

  -- d counter: run through 0..max in Dot/Acc/Fin; reset at phase changes
  dNext =
    liftA4
      (\ph w dLast curr ->
        case ph of
          PhaseDot      -> if w && dLast then 0 else if w then succ curr else curr
          PhaseAcc      -> if w && dLast then 0 else if w then succ curr else curr
          PhaseFinalize -> if w && dLast then 0 else if w then succ curr else curr)
      phase warm isLastD_now dCnt

  -- t counter: advance once per finished Acc (after last d)
  tNext =
    liftA5
      (\ph w dLast tLast curr ->
        case ph of
          PhaseDot      -> curr
          PhaseAcc      -> if w && dLast then (if tLast then 0 else succ curr) else curr
          PhaseFinalize -> curr)
      phase warm isLastD_now isLastT_now tCnt

  -- done pulse at end of Finalize, after last element written
  donePulse = register False $ (&&) <$> (isFin .&&. warm) <*> isLastD_now