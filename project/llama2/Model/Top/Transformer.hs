module Model.Top.Transformer (
    multiCycleTransformer,
    multiCycleTransformerWithTap,
    argMax
) where

import Clash.Prelude
import Helpers (NumLayers, Temperature, Seed, EmbeddingComponent (..), liftA4, CArray2D (..), VocabSize, Token, ModelDim, SeqLen)

import Model.Core.Types (IntermediateData(..), nextProcessingState, initialProcessingState, initialIntermediateData, ProcessingState (..), CycleStage (..))

import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.TransformerLayer as TransformerLayer (TransformerLayerComponent(..), TransformerDecoderComponent(..), multiCycleTransformerLayer, multiCycleTransformerLayerTap)
import Model.Layers.TransformerLayer (TransformerDecoderComponent(..), transformerLogits)
import qualified Clash.Sized.Vector as CV
import Data.Maybe (fromMaybe)
import qualified Model.Embedding.PRNG as PRNG

-- One fold step over layers:
--   - Run a single transformer layer (multi-cycle)
--   - Collect its write-done and attention-done pulses into vectors
--   - Merge the new IntermediateData with the pipeline’s current data depending on stage
transformerLayerFold :: HiddenClockResetEnable dom
  => Signal dom ProcessingState
  -> ( Signal dom IntermediateData
      , Vec NumLayers (Signal dom Bool)  -- writeDone by layer
      , Vec NumLayers (Signal dom Bool)) -- attnDone  by layer (rising edge)
  -> (TransformerLayer.TransformerLayerComponent, Cache.KVRamOwner dom, Index NumLayers)
  -> ( Signal dom IntermediateData
      , Vec NumLayers (Signal dom Bool)
      , Vec NumLayers (Signal dom Bool))
transformerLayerFold processingStateSignal (currentDataSignal, writeDoneVector, attnDoneVector)
              (transformerLayerComponent, cacheOwner, layerIndex) =
  let (newIntermediateDataSignal, writeDoneSignal, attnDoneSignal, commitCycle3Signal) =
        TransformerLayer.multiCycleTransformerLayer transformerLayerComponent cacheOwner layerIndex processingStateSignal currentDataSignal

      selectedIntermediateDataSignal = liftA4
        (\processingState oldData newData commitCycle3Data ->
            if processingLayer processingState == layerIndex
              then if processingStage processingState == Cycle3_ComputeAttention
                    then commitCycle3Data
                    else newData
              else oldData)
        processingStateSignal currentDataSignal newIntermediateDataSignal commitCycle3Signal
  in ( selectedIntermediateDataSignal
      , replace layerIndex writeDoneSignal writeDoneVector
      , replace layerIndex attnDoneSignal  attnDoneVector)

-- Full multi-cycle transformer decoder with a 1-cycle pause after sampling
-- so L0 Cycle1 consumes the new token embedding (fixes boundary race).
multiCycleTransformer :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Vec NumLayers (Cache.KVRamOwner dom)
  -> Signal dom (Unsigned 32)  -- Input token signal
  -> Signal dom Temperature    -- Temperature signal
  -> Signal dom Seed           -- PRNG seed signal
  -> (Signal dom (Unsigned 32), Signal dom Bool)
multiCycleTransformer decoder cacheOwners inputTokenSignal temperatureSignal seedSignal =
  (outputTokenSignal readyPulseRaw temperatureSignal seedSignal decoder nextIntermediateDataSignal, readyPulseOut)
 where
  embeddingComponent      = modelEmbedding decoder
  transformerLayers       = modelLayers decoder
  tokenEmbeddingSignal    = embed (vocabulary embeddingComponent) <$> inputTokenSignal

  advanceProcessingState currentState stageFinished =
    if stageFinished then nextProcessingState currentState else currentState

  processingStateSignal =
    register initialProcessingState (advanceProcessingState <$> processingStateSignal <*> stageFinishedSignal)

  intermediateDataSignal = register initialIntermediateData nextIntermediateDataSignal

  -- Load layer input at Cycle1
  inputLoadedSignal =
    liftA3
      (\processingState currentIntermediateData tokenEmbedding ->
         if processingStage processingState == Cycle1_ReadCache
           then if processingLayer processingState == 0
                  then currentIntermediateData { inputVector = tokenEmbedding }                
                  else currentIntermediateData { inputVector = feedForwardOutput currentIntermediateData }
           else currentIntermediateData)
      processingStateSignal intermediateDataSignal tokenEmbeddingSignal

  ( nextIntermediateDataSignal
    , writeDoneVector
    , attnDoneVector) =
    foldl (transformerLayerFold processingStateSignal) (inputLoadedSignal, repeat (pure False), repeat (pure False))
                        (zip3 transformerLayers cacheOwners indicesI)

  layerIndexSignal   = processingLayer <$> processingStateSignal
  writeDoneThisLayer = (!!) <$> sequenceA writeDoneVector <*> layerIndexSignal
  attnDoneThisLayer  = (!!) <$> sequenceA attnDoneVector  <*> layerIndexSignal

  -- Stage selector
  stageSignal = processingStage <$> processingStateSignal
  isStage cycleStage = (== cycleStage) <$> stageSignal

  -- Ready pulse at last layer FFN completion (unchanged)
  isLastLayerFFN  = liftA2 (\processingState _ ->
                             processingStage processingState == Cycle5_ComputeFeedForward &&
                             processingLayer processingState == maxBound)
                           processingStateSignal (pure ())
  readyPulseRaw = liftA2 (\now prev -> now && not prev)
                         isLastLayerFFN (register False isLastLayerFFN)

  -- One-cycle pause after sampling so the new token appears on input
  pauseAfterReady :: Signal dom Bool
  pauseAfterReady = register False readyPulseRaw

  -- Stage done selection with pause on Cycle5
  stageFinishedSignal =
    mux (isStage Cycle1_ReadCache)              (pure True)   $
    mux (isStage Cycle2_ComputeQKV)             (pure True)   $
    mux (isStage Cycle3_ComputeAttention)       attnDoneThisLayer  $
    mux (isStage Cycle4_WriteCache)             writeDoneThisLayer $
    mux (isStage Cycle5_ComputeFeedForward)     (not <$> pauseAfterReady) $
    pure False

  -- Export the ready pulse (unchanged externally)
  readyPulseOut = readyPulseRaw

-- logits from the current data
logitsSignal :: TransformerLayer.TransformerDecoderComponent -> Signal dom IntermediateData -> Signal dom (Vec VocabSize Float)
logitsSignal decoder nextIntermediateDataSignal = transformerLogits decoder . feedForwardOutput <$> nextIntermediateDataSignal

-- Sampling on pulse
sampledTokenSignal :: forall dom
   . HiddenClockResetEnable dom
  =>  Signal dom Bool -> Signal dom Temperature -> Signal dom (Unsigned 32) -> TransformerLayer.TransformerDecoderComponent -> Signal dom IntermediateData -> Signal dom (Unsigned 32)
sampledTokenSignal readyPulseSignal temperatureSignal seedSignal decoder nextIntermediateDataSignal =
  liftA3
    (\temperature logits randomVal ->
        if temperature <= 0.0 then argMax logits
        else let probabilities = softmax temperature logits
            in sampleFromProbs randomVal probabilities)
    temperatureSignal (logitsSignal decoder nextIntermediateDataSignal) (uniformRandom01Signal readyPulseSignal seedSignal)

outputTokenSignal :: forall dom
   . HiddenClockResetEnable dom
  =>  Signal dom Bool -> Signal dom Temperature -> Signal dom (Unsigned 32) -> TransformerLayer.TransformerDecoderComponent -> Signal dom IntermediateData -> Signal dom (Unsigned 32)
outputTokenSignal readyPulseSignal temperatureSignal seedSignal decoder nextIntermediateDataSignal = regEn 0 readyPulseSignal (sampledTokenSignal readyPulseSignal temperatureSignal seedSignal decoder nextIntermediateDataSignal)

-- PRNG state
firstPulseSignal :: forall dom
   . HiddenClockResetEnable dom
  =>  Signal dom Bool -> Signal dom Bool
firstPulseSignal readyPulseSignal = regEn True readyPulseSignal (pure False)

mixedSeedSignal :: Signal dom (Unsigned 32) -> Signal dom (Unsigned 32)
mixedSeedSignal seedSignal = (`xor` 0x9E3779B9) <$> seedSignal

prngStateSignal :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool -> Signal dom (Unsigned 32) ->Signal dom (Unsigned 32)
prngStateSignal readyPulseSignal seedSignal =
  let nextVal = mux (firstPulseSignal readyPulseSignal) (PRNG.xorshift32 <$> mixedSeedSignal seedSignal)
                                      (PRNG.xorshift32 <$> prngStateSignal readyPulseSignal seedSignal)
  in regEn 2463534242 readyPulseSignal nextVal

uniformRandom01Signal :: forall dom
   . HiddenClockResetEnable dom
  =>  Signal dom Bool -> Signal dom (Unsigned 32) -> Signal dom Float
uniformRandom01Signal readyPulseSignal seedSignal = (/ 16777216.0) . fromIntegral . (`shiftR` 8) <$> prngStateSignal readyPulseSignal seedSignal

-- Pure, synthesizable categorical sampling from probabilities summing to ~1.0
sampleFromProbs :: forall n. (KnownNat (n + 1), KnownNat n) => Float -> Vec (n + 1) Float -> Unsigned 32
sampleFromProbs u probs =
  let
    cdf = CV.scanl1 (+) probs
    idx = fromMaybe maxBound (findIndex (>= u) cdf)
  in fromIntegral (fromEnum idx)

softmax :: forall n. KnownNat (n + 1) => Float -> Vec (n + 1) Float -> Vec (n + 1) Float
softmax t xs =
  let
    m    = maximum xs
    exps = map (\x -> exp ((x - m) / t)) xs
    s    = sum exps
  in map (/ s) exps

-- | Find the index of the maximum element in a non-empty vector
argMax :: forall n. ( KnownNat (n + 1)) =>Vec (n+1) Float -> Unsigned 32
argMax vec = fst $ foldl compareMax (0, head vec) (imap (\i x -> (fromIntegral i, x)) vec)
  where
    compareMax :: (Unsigned 32, Float) -> (Unsigned 32, Float) -> (Unsigned 32, Float)
    compareMax (maxIdx, maxVal) (i, x)
      | x > maxVal = (i, x)
      | otherwise  = (maxIdx, maxVal)

-- Embed a token
embed :: CArray2D VocabSize ModelDim -> Token -> Vec ModelDim Float
embed (CArray2D vocab) tokenCode = vocab !! (fromIntegral tokenCode :: Int)

-- ====== NEW: top-level with per-layer attention tap ======
multiCycleTransformerWithTap :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Vec NumLayers (Cache.KVRamOwner dom)
  -> Signal dom (Unsigned 32)
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom (Unsigned 32)
     , Signal dom Bool
     , Signal dom Bool
     , Signal dom (Index NumLayers)
     , Signal dom (Index SeqLen)
     , Signal dom (Vec ModelDim Float)
     , Signal dom (Vec ModelDim Float)
     , Signal dom (Vec ModelDim Float)
     )
multiCycleTransformerWithTap decoder cacheOwners inputTokenSignal temperatureSignal seedSignal =
  ( outputTokenSignal readyPulseRaw temperatureSignal seedSignal decoder nextIntermediateDataSignal
  , readyPulseRaw
  , tapPulseAny
  , tapLayerIdxOut
  , tapSeqPosOut
  , dbgXHatAny
  , dbgWoAny
  , dbgXAfterAny
  )
 where
  embeddingComponent      = modelEmbedding decoder
  transformerLayers       = modelLayers decoder
  tokenEmbeddingSignal    = embed (vocabulary embeddingComponent) <$> inputTokenSignal

  advanceProcessingState currentState stageFinished =
    if stageFinished then nextProcessingState currentState else currentState
  processingStateSignal =
    register initialProcessingState (advanceProcessingState <$> processingStateSignal <*> stageFinishedSignal)
  intermediateDataSignal = register initialIntermediateData nextIntermediateDataSignal

  inputLoadedSignal =
    liftA3
      (\processingState currentIntermediateData tokenEmbedding ->
         if processingStage processingState == Cycle1_ReadCache
           then if processingLayer processingState == 0
                  then currentIntermediateData { inputVector = tokenEmbedding }
                  else currentIntermediateData { inputVector = feedForwardOutput currentIntermediateData }
           else currentIntermediateData)
      processingStateSignal intermediateDataSignal tokenEmbeddingSignal

  layerStep
    :: ( Signal dom IntermediateData
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom (Vec ModelDim Float))
       , Vec NumLayers (Signal dom (Vec ModelDim Float))
       , Vec NumLayers (Signal dom (Vec ModelDim Float)) )
    -> (TransformerLayer.TransformerLayerComponent, Cache.KVRamOwner dom, Index NumLayers)
    -> ( Signal dom IntermediateData
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom (Vec ModelDim Float))
       , Vec NumLayers (Signal dom (Vec ModelDim Float))
       , Vec NumLayers (Signal dom (Vec ModelDim Float)) )
  layerStep (currData, wDoneVec, attnDoneVec, tapVec, xHatVec, woVec, xaaVec)
            (layerComp, cacheOwner, lIx) =
    let (newData, wDone, attnDone, commitC3, tapPulse, dbgXHat, dbgWo, dbgXAfter) =
          TransformerLayer.multiCycleTransformerLayerTap layerComp cacheOwner lIx processingStateSignal currData
        selectedData =
          liftA4
            (\ps oldD newD c3D ->
               if processingLayer ps == lIx
                  then if processingStage ps == Cycle3_ComputeAttention
                         then c3D
                         else newD
                  else oldD)
            processingStateSignal currData newData commitC3
    in  ( selectedData
        , replace lIx wDone wDoneVec
        , replace lIx attnDone attnDoneVec
        , replace lIx tapPulse tapVec
        , replace lIx dbgXHat xHatVec
        , replace lIx dbgWo   woVec
        , replace lIx dbgXAfter xaaVec
        )

  ( nextIntermediateDataSignal
    , writeDoneVector
    , attnDoneVector
    , tapVec
    , xHatVec
    , woVec
    , xAfterVec ) =
      foldl layerStep (inputLoadedSignal, repeat (pure False), repeat (pure False),
                       repeat (pure False), repeat (pure (repeat 0)), repeat (pure (repeat 0)), repeat (pure (repeat 0)))
            (zip3 transformerLayers cacheOwners indicesI)

  layerIndexSignal   = processingLayer <$> processingStateSignal
  seqPosSignal       = sequencePosition <$> processingStateSignal

  writeDoneThisLayer = (!!) <$> sequenceA writeDoneVector <*> layerIndexSignal
  attnDoneThisLayer  = (!!) <$> sequenceA attnDoneVector  <*> layerIndexSignal

  stageSignal = processingStage <$> processingStateSignal
  isStage cycleStage = (== cycleStage) <$> stageSignal

  -- Ready pulse at last layer FFN completion
  isLastLayerFFN  = liftA2 (\ps _ ->
                             processingStage ps == Cycle5_ComputeFeedForward
                          && processingLayer ps == maxBound)
                           processingStateSignal (pure ())
  readyPulseRaw = liftA2 (\now prev -> now && not prev)
                         isLastLayerFFN (register False isLastLayerFFN)

  -- Pause one cycle on the ready pulse
  stageFinishedSignal =
    mux (isStage Cycle1_ReadCache)           (pure True)   $
    mux (isStage Cycle2_ComputeQKV)          (pure True)   $
    mux (isStage Cycle3_ComputeAttention)    attnDoneThisLayer  $
    mux (isStage Cycle4_WriteCache)          writeDoneThisLayer $
    mux (isStage Cycle5_ComputeFeedForward)  (not <$> readyPulseRaw) $
    pure False

  -- Fan-in taps
  tapPulseAny  = or <$> sequenceA tapVec
  dbgXHatAny   = mux tapPulseAny ( (!!) <$> sequenceA xHatVec    <*> layerIndexSignal ) (pure (repeat 0))
  dbgWoAny     = mux tapPulseAny ( (!!) <$> sequenceA woVec      <*> layerIndexSignal ) (pure (repeat 0))
  dbgXAfterAny = mux tapPulseAny ( (!!) <$> sequenceA xAfterVec  <*> layerIndexSignal ) (pure (repeat 0))

  tapLayerIdxOut = regEn 0 tapPulseAny layerIndexSignal
  tapSeqPosOut   = regEn 0 tapPulseAny seqPosSignal