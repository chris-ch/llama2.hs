module Model.Top.Transformer (
    multiCycleTransformer
    , argMax
) where

import Clash.Prelude
import Helpers (NumLayers, Temperature, Seed, EmbeddingComponent (..), liftA4, CArray2D (..), VocabSize, Token, ModelDim)

import Model.Core.Types (IntermediateData(..), nextProcessingState, initialProcessingState, initialIntermediateData, ProcessingState (..), CycleStage (..))

import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.TransformerLayer as TransformerLayer (TransformerLayerComponent(..), TransformerDecoderComponent(..), multiCycleTransformerLayer)
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

-- Full multi-cycle transformer decoder:
--   - Steps through Cycle1..Cycle5 across all layers
--   - Uses stage-done pulses to advance ProcessingState
--   - Handles embedding, sampling, and PRNG state
--   - Emits (token, readyPulse) at the end of the last layer FFN
multiCycleTransformer :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Vec NumLayers (Cache.KVRamOwner dom)
  -> Signal dom (Unsigned 32)  -- Input token signal
  -> Signal dom Temperature    -- Temperature signal
  -> Signal dom Seed           -- PRNG seed signal
  -> (Signal dom (Unsigned 32), Signal dom Bool)
multiCycleTransformer decoder cacheOwners inputTokenSignal temperatureSignal seedSignal =
  (outputTokenSignal readyPulseSignal temperatureSignal seedSignal decoder nextIntermediateDataSignal, readyPulseSignal)
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

  -- One-cycle pulse emitted when the last layer finishes its feed-forward stage.
  -- Used to trigger sampling and PRNG reseeding.
  readyPulseSignal = liftA2 (\now prev -> now && not prev)
                            isLastLayerFFN (register False isLastLayerFFN)

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
