module Model.Top.Transformer (
    multiCycleTransformer,
    argMax
) where

import Clash.Prelude
import Helpers (NumLayers, Temperature, Seed, EmbeddingComponent (..), liftA4, CArray2D (..), VocabSize, Token, ModelDim, SeqLen)
import Model.Core.Types (IntermediateData(..), initialIntermediateData, ProcessingState (..), CycleStage (..))
import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.TransformerLayer as TransformerLayer (TransformerLayerComponent(..), TransformerDecoderComponent(..), multiCycleTransformerLayer)
import Model.Layers.TransformerLayer (TransformerDecoderComponent(..), transformerLogits)
import qualified Clash.Sized.Vector as CV
import Data.Maybe (fromMaybe, isJust)
import qualified Model.Embedding.PRNG as PRNG
import qualified Model.Core.PipelineController as PipelineController (runPipelineController, PipelineOutputs (..))

-- logits from the current data
logitsSignal :: TransformerLayer.TransformerDecoderComponent -> Signal dom IntermediateData -> Signal dom (Vec VocabSize Float)
logitsSignal decoder nextIntermediateDataSignal = transformerLogits decoder . feedForwardOutput <$> nextIntermediateDataSignal

-- Sampling on pulse
sampledTokenSignal :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool -> Signal dom Temperature -> Signal dom (Unsigned 32) -> TransformerLayer.TransformerDecoderComponent -> Signal dom IntermediateData -> Signal dom (Unsigned 32)
sampledTokenSignal readyPulseSignal temperatureSignal seedSignal decoder nextIntermediateDataSignal =
  liftA3
    (\temperature logits randomVal ->
        if temperature <= 0.0 then argMax logits
        else let probabilities = softmax temperature logits
             in sampleFromProbs randomVal probabilities)
    temperatureSignal (logitsSignal decoder nextIntermediateDataSignal) (uniformRandom01Signal readyPulseSignal seedSignal)

outputTokenSignal :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool -> Signal dom Temperature -> Signal dom (Unsigned 32) -> TransformerLayer.TransformerDecoderComponent -> Signal dom IntermediateData -> Signal dom (Unsigned 32)
outputTokenSignal readyPulseSignal temperatureSignal seedSignal decoder nextIntermediateDataSignal =
  regEn 0 readyPulseSignal (sampledTokenSignal readyPulseSignal temperatureSignal seedSignal decoder nextIntermediateDataSignal)

-- PRNG state
firstPulseSignal :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool -> Signal dom Bool
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
  => Signal dom Bool -> Signal dom (Unsigned 32) -> Signal dom Float
uniformRandom01Signal readyPulseSignal seedSignal = (/ 16777216.0) . fromIntegral . (`shiftR` 8) <$> prngStateSignal readyPulseSignal seedSignal

-- Pure, synthesizable categorical sampling from probabilities summing to ~1.0
sampleFromProbs :: forall n. (KnownNat (n + 1), KnownNat n) => Float -> Vec (n + 1) Float -> Unsigned 32
sampleFromProbs u probs =
  let cdf = CV.scanl1 (+) probs
      idx = fromMaybe maxBound (findIndex (>= u) cdf)
  in fromIntegral (fromEnum idx)

softmax :: forall n. KnownNat (n + 1) => Float -> Vec (n + 1) Float -> Vec (n + 1) Float
softmax t xs =
  let m    = maximum xs
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

-- Select first Just in a Vec
firstJustV :: Vec n (Maybe a) -> Maybe a
firstJustV = foldr (\m acc -> case m of { Just _ -> m; Nothing -> acc }) Nothing

-- Top transformer with centralized sequencing and cycle-aligned taps
multiCycleTransformer :: forall dom
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
multiCycleTransformer decoder cacheOwners inputTokenSignal temperatureSignal seedSignal =
  ( outputTokenSignal (PipelineController.readyPulse ctrl) temperatureSignal seedSignal decoder nextIntermediateDataSignal
  , PipelineController.readyPulse ctrl
  , tapValid
  , tapLayerIdxOut
  , tapSeqPosOut
  , dbgXHatOut
  , dbgWoOut
  , dbgXAfterOut
  )
 where
  embeddingComponent      = modelEmbedding decoder
  transformerLayers       = modelLayers decoder
  tokenEmbeddingSignal    = embed (vocabulary embeddingComponent) <$> inputTokenSignal

  -- Storage for IntermediateData
  intermediateDataSignal = register initialIntermediateData nextIntermediateDataSignal

  -- Load input at Stage1
  inputLoadedSignal =
    liftA3
      (\ps current tokenEmbedding ->
         if processingStage ps == Stage1_LoadKV
           then if processingLayer ps == 0
                  then current { inputVector = tokenEmbedding }
                  else current { inputVector = feedForwardOutput current }
           else current)
      (PipelineController.processingState ctrl) intermediateDataSignal tokenEmbeddingSignal

  -- Per-layer step: same function you had, unchanged
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
  layerStep (currData, wDoneVec, attnDoneVec, tapVecIn, xHatVecIn, woVecIn, xaaVecIn)
            (layerComp, cacheOwner, lIx) =
    let (newData, wDone, attnDone, commitC3, tapPulse, dbgXHat, dbgWo, dbgXAfter) =
          TransformerLayer.multiCycleTransformerLayer layerComp cacheOwner lIx (PipelineController.processingState ctrl) currData
        selectedData =
          liftA4
            (\ps oldD newD c3D ->
               if processingLayer ps == lIx
                  then if processingStage ps == Stage3_Attend
                         then c3D
                         else newD
                  else oldD)
            (PipelineController.processingState ctrl) currData newData commitC3
    in  ( selectedData
        , replace lIx wDone    wDoneVec
        , replace lIx attnDone attnDoneVec
        , replace lIx tapPulse tapVecIn
        , replace lIx dbgXHat  xHatVecIn
        , replace lIx dbgWo    woVecIn
        , replace lIx dbgXAfter xaaVecIn
        )

  ( nextIntermediateDataSignal
    , writeDoneVector
    , attnDoneVector
    , tapVec
    , xHatVec
    , woVec
    , xAfterVec ) =
      foldl layerStep (inputLoadedSignal
                      , repeat (pure False)
                      , repeat (pure False)
                      , repeat (pure False)
                      , repeat (pure (repeat 0))
                      , repeat (pure (repeat 0))
                      , repeat (pure (repeat 0)))
            (zip3 transformerLayers cacheOwners indicesI)

  -- Select “this layer” done signals using the current FSM layer index
  writeDoneThisLayer = (!!) <$> sequenceA writeDoneVector <*> PipelineController.layerIndex ctrl
  attnDoneThisLayer  = (!!) <$> sequenceA attnDoneVector  <*> PipelineController.layerIndex ctrl

  -- Centralized FSM + readyPulse + stageFinished
  ctrl = PipelineController.runPipelineController attnDoneThisLayer writeDoneThisLayer

  -- =================== Layer-accurate tap fan-in ====================
  tapPayloadPerLayer
    :: Vec NumLayers (Signal dom (Maybe ( Index NumLayers
                                        , Index SeqLen
                                        , Vec ModelDim Float
                                        , Vec ModelDim Float
                                        , Vec ModelDim Float )))
  tapPayloadPerLayer =
    map (\lIx ->
          let pulse = tapVec    !! lIx
              xh    = xHatVec   !! lIx
              wo    = woVec     !! lIx
              xa    = xAfterVec !! lIx
              tup   = bundle (pure lIx, PipelineController.seqPos ctrl, xh, wo, xa)
          in mux pulse (Just <$> tup) (pure Nothing)
        )
        indicesI

  tapSelected = firstJustV <$> sequenceA tapPayloadPerLayer
  tapValid = isJust <$> tapSelected

  tapLayerIdxOut = regEn 0           tapValid ((\(l,_,_,_,_) -> l) . fromJustX <$> tapSelected)
  tapSeqPosOut   = regEn 0           tapValid ((\(_,p,_,_,_) -> p) . fromJustX <$> tapSelected)
  dbgXHatOut     = regEn (repeat 0)  tapValid ((\(_,_,xh,_,_) -> xh) . fromJustX <$> tapSelected)
  dbgWoOut       = regEn (repeat 0)  tapValid ((\(_,_,_,wo,_) -> wo) . fromJustX <$> tapSelected)
  dbgXAfterOut   = regEn (repeat 0)  tapValid ((\(_,_,_,_,xa) -> xa) . fromJustX <$> tapSelected)
