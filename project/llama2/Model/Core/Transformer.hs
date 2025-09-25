module Model.Core.Transformer (
    multiCycleTransformer
) where

import Clash.Prelude
import Helpers (liftA4)
import Model.Core.Types
  ( IntermediateData(..)
  , ProcessingState (..)
  , CycleStage (..)
  , NumLayers, Temperature, Seed
  , EmbeddingComponent (..)
  , CArray2D (..)
  , VocabSize, Token, ModelDim, SeqLen
  )
import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.TransformerLayer as TransformerLayer
  ( TransformerLayerComponent(..)
  , TransformerDecoderComponent(..)
  , multiCycleTransformerLayer
  )
import Model.Layers.TransformerLayer (TransformerDecoderComponent(..))
import Data.Maybe (isJust)
import qualified Model.Embedding.PRNG as PRNG
import qualified Model.Core.PipelineController as PipelineController
  ( runPipelineController
  , PipelineOutputs (..)
  )

-- Initial contents for the per-token IntermediateData wave
initialIntermediateData :: IntermediateData
initialIntermediateData = IntermediateData
  { inputVector       = repeat 0
  , queryVectors      = repeat (repeat 0)
  , keyVectors        = repeat (repeat 0)
  , valueVectors      = repeat (repeat 0)
  , attentionOutput   = repeat 0
  , feedForwardOutput = repeat 0
  }

-- Feed logits into the sampler and register the chosen token on readyPulse
outputTokenSignal
  :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom (Unsigned 32)
  -> TransformerLayer.TransformerDecoderComponent
  -> Signal dom IntermediateData
  -> Signal dom (Unsigned 32)
outputTokenSignal readyPulseSignal temperatureSignal seedSignal decoder nextIntermediateDataSignal =
  regEn 0 readyPulseSignal
        (PRNG.sampledTokenSignal readyPulseSignal temperatureSignal seedSignal decoder nextIntermediateDataSignal)

-- Token embedding lookup (tied weights are handled in the classifier elsewhere)
embed :: CArray2D VocabSize ModelDim -> Token -> Vec ModelDim Float
embed (CArray2D vocab) tokenCode = vocab !! (fromIntegral tokenCode :: Int)

-- Select first Just in a Vec
firstJustV :: Vec n (Maybe a) -> Maybe a
firstJustV = foldr (\m acc -> case m of { Just _ -> m; Nothing -> acc }) Nothing

-- Top transformer with centralized sequencing and cycle-aligned taps
multiCycleTransformer
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Vec NumLayers (Cache.KVRamOwner dom)
  -> Signal dom Token
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token
     , Signal dom Bool
     , Signal dom Bool
     , Signal dom (Index NumLayers)
     , Signal dom (Index SeqLen)
     , Signal dom (Vec ModelDim Float)
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
  , dbgConcatHeadsOut
  , dbgWoOut
  , dbgXAfterOut
  )
 where
  embeddingComponent      = modelEmbedding decoder
  transformerLayers       = modelLayers decoder
  tokenEmbeddingSignal    = embed (vocabulary embeddingComponent) <$> inputTokenSignal

  -- Storage for IntermediateData (registered once per cycle)
  intermediateDataSignal = register initialIntermediateData nextIntermediateDataSignal

  -- Load input at new Stage1 (ProjectQKV in Option B)
  -- Layer 0 takes fresh token embedding; higher layers take previous layer's FFN output.
  inputLoadedSignal ::
       Signal dom IntermediateData
  inputLoadedSignal =
    liftA3
      (\ps current tokenEmbedding ->
         if processingStage ps == Stage1_ProjectQKV
           then if processingLayer ps == 0
                  then current { inputVector = tokenEmbedding }
                  else current { inputVector = feedForwardOutput current }
           else current)
      (PipelineController.processingState ctrl) intermediateDataSignal tokenEmbeddingSignal

  -- Per-layer step
  layerStep
    :: ( Signal dom IntermediateData
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom (Vec ModelDim Float))
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
       , Vec NumLayers (Signal dom (Vec ModelDim Float))
       , Vec NumLayers (Signal dom (Vec ModelDim Float)) )
  layerStep (currData, wDoneVec, attnDoneVec, tapVecIn, xHatVecIn, concatHeadsVecIn, woVecIn, xaaVecIn)
            (layerComp, cacheOwner, lIx) =
    let (newData, wDone, attnDone, commitC3, tapPulse, dbgXHat, dbgConcatHeads, dbgWo, dbgXAfter) =
          TransformerLayer.multiCycleTransformerLayer layerComp cacheOwner lIx (PipelineController.processingState ctrl) currData
        -- During Stage3_Attend we use the commitC3 view; otherwise, use newData.
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
        , replace lIx dbgConcatHeads concatHeadsVecIn
        , replace lIx dbgWo    woVecIn
        , replace lIx dbgXAfter xaaVecIn
        )

  ( nextIntermediateDataSignal
    , writeDoneVector
    , attnDoneVector
    , tapVec
    , xHatVec
    , chVec
    , woVec
    , xAfterVec ) =
      foldl layerStep ( inputLoadedSignal
                      , repeat (pure False)
                      , repeat (pure False)
                      , repeat (pure False)
                      , repeat (pure (repeat 0))
                      , repeat (pure (repeat 0))
                      , repeat (pure (repeat 0))
                      , repeat (pure (repeat 0)) )
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
                                        , Vec ModelDim Float
                                        , Vec ModelDim Float )))
  tapPayloadPerLayer =
    map (\lIx ->
          let pulse = tapVec    !! lIx
              xh    = xHatVec   !! lIx
              wo    = woVec     !! lIx
              ch    = chVec     !! lIx
              xa    = xAfterVec !! lIx
              tup   = bundle (pure lIx, PipelineController.seqPos ctrl, xh, ch, wo, xa)
          in mux pulse (Just <$> tup) (pure Nothing)
        )
        indicesI

  tapSelected = firstJustV <$> sequenceA tapPayloadPerLayer
  tapValid    = isJust <$> tapSelected

  tapLayerIdxOut    = regEn 0           tapValid ((\(l,_,_,_,_,_) -> l) . fromJustX <$> tapSelected)
  tapSeqPosOut      = regEn 0           tapValid ((\(_,p,_,_,_,_) -> p) . fromJustX <$> tapSelected)
  dbgXHatOut        = regEn (repeat 0)  tapValid ((\(_,_,xh,_,_,_) -> xh) . fromJustX <$> tapSelected)
  dbgConcatHeadsOut = regEn (repeat 0)  tapValid ((\(_,_,_,ch,_,_) -> ch) . fromJustX <$> tapSelected)
  dbgWoOut          = regEn (repeat 0)  tapValid ((\(_,_,_,_,wo,_) -> wo) . fromJustX <$> tapSelected)
  dbgXAfterOut      = regEn (repeat 0)  tapValid ((\(_,_,_,_,_,xa) -> xa) . fromJustX <$> tapSelected)
