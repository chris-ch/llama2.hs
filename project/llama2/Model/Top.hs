module Model.Top
  ( multiCycleTransformer
  , topEntity
  ) where

import Clash.Prelude

import Helpers
  ( TransformerDecoderComponent(..), EmbeddingComponent(..)
  , embed, transformerLogits, argMax, softmax, sampleFromProbs
  , Temperature, Seed, NumLayers, TransformerLayerComponent, liftA4, xorshift32
  )

import Model.Types
  ( ProcessingState(..), initialProcessingState, nextProcessingState
  , IntermediateData(..), initialIntermediateData, CycleStage (..)
  )

import Model.Cache
  ( KVRamOwner(..), makeRamOwnerKV)

import Model.Layer (multiCycleTransformerLayer)

-- Run layers; gather per-stage done pulses
foldLayerStep
  :: HiddenClockResetEnable dom
  => Signal dom ProcessingState
  -> ( Signal dom IntermediateData
      , Vec NumLayers (Signal dom Bool)  -- writeDone by layer
      , Vec NumLayers (Signal dom Bool)) -- attnDone  by layer (rising edge)
  -> (TransformerLayerComponent, KVRamOwner dom, Index NumLayers)
  -> ( Signal dom IntermediateData
      , Vec NumLayers (Signal dom Bool)
      , Vec NumLayers (Signal dom Bool))
foldLayerStep processingStateSignal (currentDataSignal, writeDoneVector, attnDoneVector)
              (transformerLayerComponent, cacheOwner, layerIndex) =
  let (newIntermediateDataSignal, writeDoneSignal, attnDoneSignal, commitCycle3Signal) =
        multiCycleTransformerLayer transformerLayerComponent cacheOwner layerIndex processingStateSignal currentDataSignal

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

-- Full transformer (multi-cycle, advance on stageDone)
multiCycleTransformer
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> Vec NumLayers (KVRamOwner dom)
  -> Signal dom (Unsigned 32)  -- Input token signal
  -> Signal dom Temperature    -- Temperature signal
  -> Signal dom Seed           -- PRNG seed signal
  -> (Signal dom (Unsigned 32), Signal dom Bool)
multiCycleTransformer decoder cacheOwners inputTokenSignal temperatureSignal seedSignal =
  (outputTokenSignal, readyPulseSignal)
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
    foldl (foldLayerStep processingStateSignal) (inputLoadedSignal, repeat (pure False), repeat (pure False))
                        (zip3 transformerLayers cacheOwners indicesI)

  layerIndexSignal   = processingLayer <$> processingStateSignal
  writeDoneThisLayer = (!!) <$> sequenceA writeDoneVector <*> layerIndexSignal
  attnDoneThisLayer  = (!!) <$> sequenceA attnDoneVector  <*> layerIndexSignal

  -- Stage done selection
  stageSignal = processingStage <$> processingStateSignal
  isStage cycleStage = (== cycleStage) <$> stageSignal

  stageFinishedSignal =
    mux (isStage Cycle1_ReadCache)   (pure True)   $
    mux (isStage Cycle2_ComputeQKV)  (pure True)   $
    mux (isStage Cycle3_ComputeAttention) attnDoneThisLayer  $
    mux (isStage Cycle4_WriteCache)  writeDoneThisLayer $
    mux (isStage Cycle5_ComputeFeedForward)  (pure True)   $
    pure False

  -- Ready pulse at last layer FFN completion
  isLastLayerFFN  = liftA2 (\processingState _ ->
                             processingStage processingState == Cycle5_ComputeFeedForward &&
                             processingLayer processingState == maxBound)
                           processingStateSignal (pure ())
  readyPulseSignal = liftA2 (\now prev -> now && not prev)
                            isLastLayerFFN (register False isLastLayerFFN)

  -- logits from the current data
  logitsSignal = transformerLogits decoder . feedForwardOutput <$> nextIntermediateDataSignal

  -- PRNG state
  firstPulseSignal :: Signal dom Bool
  firstPulseSignal = regEn True readyPulseSignal (pure False)

  mixedSeedSignal :: Signal dom (Unsigned 32)
  mixedSeedSignal = (`xor` 0x9E3779B9) <$> seedSignal

  prngStateSignal :: Signal dom (Unsigned 32)
  prngStateSignal =
    let nextVal = mux firstPulseSignal (xorshift32 <$> mixedSeedSignal)
                                        (xorshift32 <$> prngStateSignal)
    in regEn 2463534242 readyPulseSignal nextVal

  uniformRandom01Signal :: Signal dom Float
  uniformRandom01Signal = (/ 16777216.0) . fromIntegral . (`shiftR` 8) <$> prngStateSignal

  -- Sampling on pulse
  sampledTokenSignal :: Signal dom (Unsigned 32)
  sampledTokenSignal =
    liftA3
      (\temperature logits randomVal ->
         if temperature <= 0.0 then argMax logits
         else let probabilities = softmax temperature logits
              in sampleFromProbs randomVal probabilities)
      temperatureSignal logitsSignal uniformRandom01Signal

  outputTokenSignal = regEn 0 readyPulseSignal sampledTokenSignal

-- ============================================================================
-- Top Entity
-- ============================================================================
topEntity
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> Signal dom (Unsigned 32)  -- Input token
  -> Signal dom Temperature
  -> Signal dom Seed
  -> (Signal dom (Unsigned 32), Signal dom Bool)
topEntity decoder = multiCycleTransformer decoder (repeat makeRamOwnerKV)
