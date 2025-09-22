module Model.Attention
  ( AttnPhase(..)
  , streamHeadAttentionAddrIO
  ) where

import Clash.Prelude

import Helpers ( HeadDimension, SeqLen, liftA4, liftA5 )
import Model.Cache (computeBankAddress)
import Model.Types (BankAddress)

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
  => Signal dom Bool                         -- start signal
  -> Signal dom (Index SeqLen)               -- sequence position signal
  -> Signal dom (Vec HeadDimension Float)    -- query vector signal
  -> Signal dom (Vec HeadDimension Float)    -- current-key vector signal (bypass at t==pos)
  -> Signal dom (Vec HeadDimension Float)    -- current-value vector signal (bypass at t==pos)
  -> Signal dom Float                        -- key from RAM (1-cycle delayed)
  -> Signal dom Float                        -- value from RAM (1-cycle delayed)
  -> ( Signal dom BankAddress                -- bank address signal (use for both K and V)
     , Signal dom (Vec HeadDimension Float)  -- head output vector signal
     , Signal dom Bool                       -- busy signal
     , Signal dom Bool )                     -- done signal (1-cycle pulse)
streamHeadAttentionAddrIO startSignal sequencePositionSignal queryVectorSignal currentKeySignal currentValueSignal keyFromRamSignal valueFromRamSignal =
  (bankAddressSignal, outputVectorSignal, busySignal, donePulseSignal)
 where
  -- Phase control + warm-up (align 1-cycle BRAM)
  phaseSignal    = register PhaseDot nextPhaseSignal
  isDotPhase     = (== PhaseDot)      <$> phaseSignal
  isAccPhase     = (== PhaseAcc)      <$> phaseSignal
  isFinalizePhase= (== PhaseFinalize) <$> phaseSignal

  startPreviousSignal  = register False startSignal
  startPulseSignal     = liftA2 (\now prev -> now && not prev) startSignal startPreviousSignal

  warmupSignal     = regEn False enableSignal warmupNextSignal
  warmupNextSignal = mux startPulseSignal (pure False) $
                     mux (isDotPhase .||. isAccPhase) (pure True) warmupSignal -- stay warm after first tick in Dot/Acc
  enableSignal = startPulseSignal .||. busySignal

  busySignal = busyReg where
    busyReg    = register False nextBusy
    nextBusy   = mux startPulseSignal (pure True) $
                 mux donePulseSignal   (pure False) busyReg

  -- Counters
  timeCounterSignal  = regEn 0 enableSignal nextTimeCounterSignal
  dimensionCounterSignal  = regEn 0 enableSignal nextDimensionCounterSignal
  previousTimeCounterSignal = register 0 timeCounterSignal
  previousDimensionCounterSignal = register 0 dimensionCounterSignal
  isLastDimensionSignal = (== maxBound) <$> dimensionCounterSignal
  isLastTimeSignal      = (==) <$> timeCounterSignal <*> sequencePositionSignal

  -- Addressing (issue address for current (time, dimension); data valid next cycle)
  bankAddressSignal = computeBankAddress <$> timeCounterSignal <*> dimensionCounterSignal

  -- Element selection, aligned to previous address
  queryElementRawSignal = (!!) <$> queryVectorSignal    <*> previousDimensionCounterSignal
  keyBypassElementSignal= (!!) <$> currentKeySignal     <*> previousDimensionCounterSignal
  valueBypassElementSignal= (!!) <$> currentValueSignal <*> previousDimensionCounterSignal

  useKeyBypassSignal =
    liftA3 (\warm pd pt -> warm && pd && pt)
           warmupSignal (pure True) ((==) <$> previousTimeCounterSignal <*> sequencePositionSignal)
  useValueBypassSignal = useKeyBypassSignal

  keyElementSignal   = mux useKeyBypassSignal  keyBypassElementSignal  keyFromRamSignal
  valueElementSignal = mux useValueBypassSignal valueBypassElementSignal valueFromRamSignal
  queryElementSignal = queryElementRawSignal

  -- Scaling
  invSqrtHeadDim :: Float
  invSqrtHeadDim = 1.0 / sqrt (snatToNum (SNat @HeadDimension) :: Float)

  -- Online softmax accumulators across timesteps
  maxAccumulatorSignal = regEn (-(1/0)) enableSignal nextMaxAccumulatorSignal
  sumAccumulatorSignal = regEn 0        enableSignal nextSumAccumulatorSignal

  -- Per-t cached scales (written at end of Dot, used across all d in Acc)
  scaleOldRegisterSignal = regEn 1 enableSignal nextScaleOldSignal
  scaleNewRegisterSignal = regEn 0 enableSignal nextScaleNewSignal

  -- Dot partial across d for current t
  dotAccumulatorSignal = regEn 0 enableSignal nextDotAccumulatorSignal

  canAccumulateDotSignal = warmupSignal .&&. isDotPhase
  nextDotAccumulatorSignal =
    mux startPulseSignal (pure 0)
    $ mux isDotPhase
        (mux canAccumulateDotSignal (dotAccumulatorSignal + (queryElementSignal * keyElementSignal))
                                    dotAccumulatorSignal)
        dotAccumulatorSignal

  -- Boundary at end of Dot
  dotBoundarySignal =
    liftA3
      (\phase warm dLast -> phase == PhaseDot && warm && dLast)
      phaseSignal warmupSignal isLastDimensionSignal

  scoreThisTimeSignal = (dotAccumulatorSignal + (queryElementSignal * keyElementSignal)) * pure invSqrtHeadDim
  newMaxSignal       = max <$> maxAccumulatorSignal <*> scoreThisTimeSignal
  scaleOldSignal     = exp <$> ((-) <$> maxAccumulatorSignal      <*> newMaxSignal)
  scaleNewSignal     = exp <$> ((-) <$> scoreThisTimeSignal <*> newMaxSignal)
  newSumSignal       = (+) <$> ((* ) <$> sumAccumulatorSignal <*> scaleOldSignal) <*> scaleNewSignal

  -- Latch m/s/scales at t boundary
  nextMaxAccumulatorSignal =
    mux startPulseSignal (pure (-(1/0)))
    $ mux dotBoundarySignal newMaxSignal maxAccumulatorSignal
  nextSumAccumulatorSignal =
    mux startPulseSignal (pure 0)
    $ mux dotBoundarySignal newSumSignal sumAccumulatorSignal
  nextScaleOldSignal  = mux dotBoundarySignal scaleOldSignal scaleOldRegisterSignal
  nextScaleNewSignal  = mux dotBoundarySignal scaleNewSignal scaleNewRegisterSignal

  -- Numerator across d: oNum[d] = oNum[d]*scaleOld + v[d]*scaleNew
  numeratorVectorSignal = regEn (repeat 0 :: Vec HeadDimension Float) enableSignal nextNumeratorVectorSignal
  numeratorElementPrevSignal = (!!) <$> numeratorVectorSignal <*> previousDimensionCounterSignal
  numeratorElementUpdatedSignal =
    (\o v a b -> o * b + v * a) <$> numeratorElementPrevSignal
                                <*> valueElementSignal
                                <*> scaleNewRegisterSignal
                                <*> scaleOldRegisterSignal

  canAccumulateAccSignal = warmupSignal .&&. isAccPhase
  nextNumeratorVectorSignal =
    mux startPulseSignal (pure (repeat 0))
    $ mux isAccPhase
        (mux canAccumulateAccSignal (replace <$> previousDimensionCounterSignal
                                             <*> numeratorElementUpdatedSignal
                                             <*> numeratorVectorSignal)
                                    numeratorVectorSignal)
        numeratorVectorSignal

  -- Finalize: write output[d] = numerator[d] / sAcc
  outputVectorSignal     = regEn (repeat 0 :: Vec HeadDimension Float) enableSignal nextOutputVectorSignal
  outputElementDivSignal = (/ ) <$> numeratorElementPrevSignal <*> sumAccumulatorSignal
  canFinalizeSignal = warmupSignal .&&. isFinalizePhase
  nextOutputVectorSignal =
    mux startPulseSignal (pure (repeat 0))
    $ mux isFinalizePhase
        (mux canFinalizeSignal (replace <$> previousDimensionCounterSignal
                                        <*> outputElementDivSignal
                                        <*> outputVectorSignal)
                               outputVectorSignal)
        outputVectorSignal

  -- Phase transitions and counters
  nextPhaseSignal =
    liftA5
      (\phase warm dLast tLast dotBoundary ->
        case phase of
          PhaseDot      -> if dotBoundary then PhaseAcc else PhaseDot
          PhaseAcc      -> if warm && dLast then (if tLast then PhaseFinalize else PhaseDot) else PhaseAcc
          PhaseFinalize -> if warm && dLast then PhaseDot else PhaseFinalize)
      phaseSignal warmupSignal isLastDimensionSignal isLastTimeSignal dotBoundarySignal

  -- d counter: run through 0..max in Dot/Acc/Fin; reset at phase changes
  nextDimensionCounterSignal =
    liftA4
      (\phase warm dLast current ->
        case phase of
          PhaseDot      -> if warm && dLast then 0 else if warm then succ current else current
          PhaseAcc      -> if warm && dLast then 0 else if warm then succ current else current
          PhaseFinalize -> if warm && dLast then 0 else if warm then succ current else current)
      phaseSignal warmupSignal isLastDimensionSignal dimensionCounterSignal

  -- t counter: advance once per finished Acc (after last d)
  nextTimeCounterSignal =
    liftA5
      (\phase warm dLast tLast current ->
        case phase of
          PhaseDot      -> current
          PhaseAcc      -> if warm && dLast then (if tLast then 0 else succ current) else current
          PhaseFinalize -> current)
      phaseSignal warmupSignal isLastDimensionSignal isLastTimeSignal timeCounterSignal

  -- done pulse at end of Finalize, after last element written
  donePulseSignal =
    register False $ (&&) <$> (isFinalizePhase .&&. warmupSignal) <*> isLastDimensionSignal
