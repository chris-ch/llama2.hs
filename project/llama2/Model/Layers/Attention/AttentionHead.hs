module Model.Layers.Attention.AttentionHead (
  streamHeadAttentionAddrIO
) where

import Model.Core.Types (HeadDimension, SeqLen, BankAddress )
import Clash.Prelude
import Helpers (liftA5, liftA4)
import qualified Model.Memory.Addressing as Addressing

data AttnPhase = PhaseDot | PhaseAcc | PhaseFinalize
  deriving (Generic, NFDataX, Eq, Show)

-- Streaming single-head attention with online softmax and t==pos bypass for K,V.
-- BRAM is 1-cycle latent: the data we see this cycle corresponds to last cycle's address,
-- which we track with previousTimeCounterSignal/previousDimensionCounterSignal.
streamHeadAttentionAddrIO
  :: HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom (Index SeqLen)
  -> Signal dom (Vec HeadDimension Float)
  -> Signal dom (Vec HeadDimension Float)
  -> Signal dom (Vec HeadDimension Float)
  -> Signal dom Float
  -> Signal dom Float
  -> ( Signal dom BankAddress
     , Signal dom (Vec HeadDimension Float)
     , Signal dom Bool
     , Signal dom Bool )
streamHeadAttentionAddrIO startSignal sequencePositionSignal queryVectorSignal currentKeySignal currentValueSignal keyFromRamSignal valueFromRamSignal =
  (bankAddressSignal, outputVectorSignal, busySignal, donePulseSignal)
 where
  phaseSignal        = register PhaseDot nextPhaseSignal
  isDotPhase         = (== PhaseDot)      <$> phaseSignal
  isAccPhase         = (== PhaseAcc)      <$> phaseSignal
  isFinalizePhase    = (== PhaseFinalize) <$> phaseSignal

  startPrevSignal    = register False startSignal
  startPulseSignal   = liftA2 (\now prev -> now && not prev) startSignal startPrevSignal

  enableSignal       = startPulseSignal .||. busySignal
  busySignal         = busyReg where
    busyReg = register False nextBusy
    nextBusy =
      mux startPulseSignal (pure True) $
      mux donePulseSignal   (pure False) busyReg

  timeCounterSignal              = regEn 0 enableSignal nextTimeCounterSignal
  dimensionCounterSignal         = regEn 0 enableSignal nextDimensionCounterSignal
  previousTimeCounterSignal      = register 0 timeCounterSignal
  previousDimensionCounterSignal = register 0 dimensionCounterSignal

  isLastDimensionSignal = (== maxBound) <$> dimensionCounterSignal
  isLastTimeSignal      = (==) <$> timeCounterSignal <*> sequencePositionSignal

  bankAddressSignal = Addressing.computeBankAddress <$> timeCounterSignal <*> dimensionCounterSignal

  queryElementSignal       = (!!) <$> queryVectorSignal        <*> previousDimensionCounterSignal
  keyBypassElementSignal   = (!!) <$> currentKeySignal         <*> previousDimensionCounterSignal
  valueBypassElementSignal = (!!) <$> currentValueSignal       <*> previousDimensionCounterSignal

  -- Unconditional selection for the current time (no warmup involved)
  isReturnForCurrentTime   = (==) <$> previousTimeCounterSignal <*> sequencePositionSignal
  keyElementSignal         = mux isReturnForCurrentTime  keyBypassElementSignal   keyFromRamSignal
  valueElementSignal       = mux isReturnForCurrentTime  valueBypassElementSignal valueFromRamSignal

  warmupSignal = regEn False enableSignal warmupNext
  warmupNext   = mux startPulseSignal (pure False) $
                 mux (isDotPhase .||. isAccPhase .||. isFinalizePhase) (pure True) warmupSignal

  dotAccumulatorSignal = regEn 0 enableSignal nextDotAccumulatorSignal
  canAccumulateDot     = warmupSignal .&&. isDotPhase
  nextDotAccumulatorSignal =
    mux startPulseSignal (pure 0)
    $ mux isDotPhase
        (mux canAccumulateDot (dotAccumulatorSignal + (queryElementSignal * keyElementSignal))
                              dotAccumulatorSignal)
        dotAccumulatorSignal

  dotBoundarySignal =
    liftA3 (\ph warm dLast -> ph == PhaseDot && warm && dLast)
           phaseSignal warmupSignal isLastDimensionSignal

  invSqrtHeadDim :: Float
  invSqrtHeadDim = 1.0 / sqrt (snatToNum (SNat @HeadDimension) :: Float)

  scoreThisTimeSignal =
    (dotAccumulatorSignal + (queryElementSignal * keyElementSignal)) * pure invSqrtHeadDim

  maxAccumulatorSignal = regEn (-(1/0)) enableSignal nextMaxAccumulatorSignal
  sumAccumulatorSignal = regEn 0        enableSignal nextSumAccumulatorSignal

  newMaxSignal   = max <$> maxAccumulatorSignal <*> scoreThisTimeSignal
  scaleOldSignal = exp <$> ((-) <$> maxAccumulatorSignal <*> newMaxSignal)
  scaleNewSignal = exp <$> ((-) <$> scoreThisTimeSignal <*> newMaxSignal)
  newSumSignal   = (+) <$> ((* ) <$> sumAccumulatorSignal <*> scaleOldSignal) <*> scaleNewSignal

  nextMaxAccumulatorSignal =
    mux startPulseSignal (pure (-(1/0)))
    $ mux dotBoundarySignal newMaxSignal maxAccumulatorSignal
  nextSumAccumulatorSignal =
    mux startPulseSignal (pure 0)
    $ mux dotBoundarySignal newSumSignal sumAccumulatorSignal

  scaleOldRegisterSignal = regEn 1 enableSignal (mux dotBoundarySignal scaleOldSignal scaleOldRegisterSignal)
  scaleNewRegisterSignal = regEn 0 enableSignal (mux dotBoundarySignal scaleNewSignal scaleNewRegisterSignal)

  numeratorVectorSignal = regEn (repeat 0 :: Vec HeadDimension Float) enableSignal nextNumeratorVectorSignal
  numeratorElementPrev  = (!!) <$> numeratorVectorSignal <*> previousDimensionCounterSignal
  numeratorElementUpd   =
    (\o v a b -> o * b + v * a) <$> numeratorElementPrev <*> valueElementSignal
                                <*> scaleNewRegisterSignal <*> scaleOldRegisterSignal

  canAccumulateAcc = warmupSignal .&&. isAccPhase
  nextNumeratorVectorSignal =
    mux startPulseSignal (pure (repeat 0))
    $ mux isAccPhase
        (mux canAccumulateAcc (replace <$> previousDimensionCounterSignal
                                       <*> numeratorElementUpd
                                       <*> numeratorVectorSignal)
                              numeratorVectorSignal)
        numeratorVectorSignal

  outputVectorSignal = regEn (repeat 0 :: Vec HeadDimension Float) enableSignal nextOutputVectorSignal
  outputElementDiv   = (/ ) <$> numeratorElementPrev <*> sumAccumulatorSignal
  canFinalize        = warmupSignal .&&. isFinalizePhase
  nextOutputVectorSignal =
    mux startPulseSignal (pure (repeat 0))
    $ mux isFinalizePhase
        (mux canFinalize (replace <$> previousDimensionCounterSignal
                                  <*> outputElementDiv
                                  <*> outputVectorSignal)
                         outputVectorSignal)
        outputVectorSignal

  nextPhaseSignal =
    liftA5
      (\ph warm dLast tLast dotB ->
        case ph of
          PhaseDot      -> if dotB then PhaseAcc else PhaseDot
          PhaseAcc      -> if warm && dLast then (if tLast then PhaseFinalize else PhaseDot) else PhaseAcc
          PhaseFinalize -> if warm && dLast then PhaseDot else PhaseFinalize)
      phaseSignal warmupSignal isLastDimensionSignal isLastTimeSignal dotBoundarySignal

  nextDimensionCounterSignal =
    liftA4
      (\ph warm dLast cur ->
        case ph of
          PhaseDot      -> if warm && dLast then 0 else if warm then succ cur else cur
          PhaseAcc      -> if warm && dLast then 0 else if warm then succ cur else cur
          PhaseFinalize -> if warm && dLast then 0 else if warm then succ cur else cur)
      phaseSignal warmupSignal isLastDimensionSignal dimensionCounterSignal

  nextTimeCounterSignal =
    liftA5
      (\ph warm dLast tLast cur ->
        case ph of
          PhaseDot      -> cur
          PhaseAcc      -> if warm && dLast then (if tLast then 0 else succ cur) else cur
          PhaseFinalize -> cur)
      phaseSignal warmupSignal isLastDimensionSignal isLastTimeSignal timeCounterSignal

  donePulseSignal =
    register False $ (&&) <$> (isFinalizePhase .&&. warmupSignal) <*> isLastDimensionSignal
