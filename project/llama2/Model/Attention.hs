module Model.Attention
  ( AttnPhase(..)
  , streamHeadAttentionAddrIO
  ) where

import Clash.Prelude

import Helpers ( HeadDimension, SeqLen, liftA4 )
import Model.Cache (bankAddr)
import Model.Types (BankAddr)

-- One-head streaming attention core with online softmax accumulation.
-- Reads K/V from BRAM (1-cycle latency), bypasses current-row K/V,
-- produces the head output vector when 'done' pulses.

data AttnPhase = PhaseDot | PhaseAcc | PhaseFinalize
  deriving (Generic, NFDataX, Eq, Show)

streamHeadAttentionAddrIO
  :: HiddenClockResetEnable dom
  => Signal dom Bool                         -- start
  -> Signal dom (Index SeqLen)               -- pos
  -> Signal dom (Vec HeadDimension Float)    -- q
  -> Signal dom (Vec HeadDimension Float)    -- curK (bypass)
  -> Signal dom (Vec HeadDimension Float)    -- curV (bypass)
  -> Signal dom Float                        -- kQ (from RAM, 1-cycle)
  -> Signal dom Float                        -- vQ (from RAM, 1-cycle)
  -> ( Signal dom BankAddr                   -- addr (use for both K and V)
     , Signal dom (Vec HeadDimension Float)  -- head output
     , Signal dom Bool                       -- busy
     , Signal dom Bool )                     -- done (1-cycle)
streamHeadAttentionAddrIO startSig posSig qSig curKSig curVSig kQin vQin =
  (addr, oOut, busySig, doneSig)
 where
  phase    = register PhaseDot phaseNext
  isDot    = (== PhaseDot)      <$> phase
  isAcc    = (== PhaseAcc)      <$> phase
  isFin    = (== PhaseFinalize) <$> phase
  startPrev  = register False startSig
  startPulse = liftA2 (\n p -> n && not p) startSig startPrev

  -- Enables
  busySig = s where
    s    = register False next
    next = mux startPulse (pure True) $ mux doneSig (pure False) s

  en = startPulse .||. busySig

  -- Counters
  tCnt = regEn 0 en tNext
  dCnt = regEn 0 en dNext
  dPrev = register 0 dCnt
  isLastD = (== maxBound) <$> dCnt
  isLastT = (==) <$> tCnt <*> posSig

  -- RAM addressing (synchronous read: data appears next cycle)
  addr = bankAddr <$> tCnt <*> dCnt

  -- Bypass at t==pos (current row), aligned with dPrev
  useKBypass = (&&) <$> isDot <*> isLastT
  useVBypass = (&&) <$> isAcc <*> isLastT
  kBypass    = (!!) <$> curKSig <*> dPrev
  vBypass    = (!!) <$> curVSig <*> dPrev

  -- Use RAM outputs directly (1-cycle) and align indexes with dPrev
  kElem = mux useKBypass kBypass kQin
  vElem = mux useVBypass vBypass vQin
  qElem = (!!) <$> qSig     <*> dPrev

  -- Scaled dot-product (online softmax)
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
