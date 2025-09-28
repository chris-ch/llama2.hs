module Model.Core.Transformer (
    multiCycleTransformer
) where

import Clash.Prelude
import Model.Core.Transformer.Internal

import Helpers (liftA4)
import Model.Core.Types
  ( IntermediateData(..)
  , ProcessingState (..)
  , CycleStage (..)
  , NumLayers, Temperature, Seed
  , EmbeddingComponent (..)
  , CArray2D (..)
  , VocabSize, Token, ModelDim, SequenceLength
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

multiCycleTransformer :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Vec NumLayers (Cache.KVRamOwner dom)
  -> Signal dom Token
  -> Signal dom Bool            -- ^ inputTokenValid (True while external prompt is used)
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token
     , Signal dom Bool
     , Signal dom Bool
     , Signal dom (Index NumLayers)
     , Signal dom (Index SequenceLength)
     , Signal dom (Vec ModelDim Float)
     )
multiCycleTransformer decoder cacheOwners inputTokenSignal inputTokenValid temperatureSignal seedSignal =
  ( selectedTokenSignal
  , PipelineController.readyPulse ctrl
  , tapValid
  , tapLayerIdxOut
  , tapSeqPosOut
  , dbgXHatOut
  )
 where
  embeddingComponent = modelEmbedding decoder
  transformerLayers  = modelLayers decoder

  -- Done flags are selected after layer fan-in (mutual recursion via registers is fine)
  writeDoneThisLayer = (!!) <$> sequenceA writeDoneVector <*> PipelineController.layerIndex ctrl
  attnDoneThisLayer  = (!!) <$> sequenceA attnDoneVector  <*> PipelineController.layerIndex ctrl

  -- Controller now takes inputTokenValid to gate Stage1 at layer 0 every position
  ctrl = PipelineController.runPipelineController attnDoneThisLayer writeDoneThisLayer inputTokenValid

  -- Feedback (sampled) token: latched on readyPulse
  feedbackTokenSignal :: Signal dom Token
  feedbackTokenSignal =
    outputTokenSignal (PipelineController.readyPulse ctrl)
                      temperatureSignal
                      seedSignal
                      decoder
                      nextIntermediateDataSignal

  -- External prompt overrides feedback whenever inputTokenValid is True
  selectedTokenSignal :: Signal dom Token
  selectedTokenSignal =
    mux inputTokenValid inputTokenSignal feedbackTokenSignal

  tokenEmbeddingSignal = embed (vocabulary embeddingComponent) <$> selectedTokenSignal

  -- Per-position intermediate data register
  intermediateDataSignal = register initialIntermediateData nextIntermediateDataSignal

  -- Load input for Stage1: L0 takes token embedding; higher layers take previous FFN
  inputLoadedSignal :: Signal dom IntermediateData
  inputLoadedSignal =
    liftA3
      (\ps current tokenEmbedding ->
         if processingStage ps == Stage1_ProjectQKV
           then if processingLayer ps == 0
                  then current { inputVector = tokenEmbedding }
                  else current { inputVector = feedForwardOutput current }
           else current)
      (PipelineController.processingState ctrl) intermediateDataSignal tokenEmbeddingSignal

  -- One layer step
  layerStep :: ( Signal dom IntermediateData
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom (Vec ModelDim Float))
        )
    -> (TransformerLayer.TransformerLayerComponent, Cache.KVRamOwner dom, Index NumLayers)
    -> ( Signal dom IntermediateData
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom (Vec ModelDim Float))
       )
  layerStep (currData, wDoneVec, attnDoneVec, tapVecIn, xHatVecIn)
            (layerComp, cacheOwner, lIx) =
    let
      (newData, wDone, attnDone, commitC3, tapPulse, dbgXHat) =
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
        )

  ( nextIntermediateDataSignal
    , writeDoneVector
    , attnDoneVector
    , tapVec
    , xHatVec
    ) =
      foldl layerStep ( inputLoadedSignal
                      , repeat (pure False)
                      , repeat (pure False)
                      , repeat (pure False)
                      , repeat (pure (repeat 0))
                      )
            (zip3 transformerLayers cacheOwners indicesI)

  -- =================== Layer-accurate tap fan-in ====================
  tapPayloadPerLayer :: Vec NumLayers (Signal dom (Maybe ( Index NumLayers
                                        , Index SequenceLength
                                        , Vec ModelDim Float
                                        )))
  tapPayloadPerLayer =
    map (\lIx ->
          let pulse = tapVec    !! lIx
              xh    = xHatVec   !! lIx
              tup   = bundle (pure lIx, PipelineController.seqPos ctrl, xh)
          in mux pulse (Just <$> tup) (pure Nothing)
        )
        indicesI

  tapSelected = firstJustV <$> sequenceA tapPayloadPerLayer
  tapValid    = isJust <$> tapSelected

  tapLayerIdxOut    = regEn 0           tapValid ((\(l,_,_) -> l) . fromJustX <$> tapSelected)
  tapSeqPosOut      = regEn 0           tapValid ((\(_,p,_) -> p) . fromJustX <$> tapSelected)
  dbgXHatOut        = regEn (repeat 0)  tapValid ((\(_,_,xh) -> xh) . fromJustX <$> tapSelected)
