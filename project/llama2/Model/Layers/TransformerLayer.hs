module Model.Layers.TransformerLayer (
    transformerLogits
    , multiCycleTransformerLayer
    , TransformerDecoderComponent(..)
    , TransformerLayerComponent(..)
) where

import Clash.Prelude
import qualified Prelude as P
import Helpers (ModelDim, VocabSize, CArray2D (..), EmbeddingComponent (..), 
    NumLayers,
    rmsNorm, dotProduct, NumQueryHeads, NumKeyValueHeads, HeadDimension, matrixVectorMult, liftA4)
import qualified Model.Memory.KVCacheBank as Cache
import Model.Core.Types (ProcessingState (..), IntermediateData (..), CycleStage (..))
import qualified Model.Layer as Layer (processCycle, fillOneBank)
import qualified Model.Layers.FeedForward.FeedForwardNetwork as FeedForwardNetwork
import qualified Model.Layers.Attention.MultiHeadAttention as MultiHeadAttention

data TransformerLayerComponent = TransformerLayerComponent
  {
    multiHeadAttention :: MultiHeadAttention.MultiHeadAttentionComponent,
    feedforwardNetwork :: FeedForwardNetwork.FeedForwardNetworkComponent
  } deriving (Show)

data TransformerDecoderComponent = TransformerDecoderComponent
  { modelEmbedding :: EmbeddingComponent,
    modelLayers :: Vec NumLayers TransformerLayerComponent
  } deriving (Show)

-- classifier logits for a given token vector
transformerLogits :: TransformerDecoderComponent -> Vec ModelDim Float -> Vec VocabSize Float
transformerLogits decoder tokenVector = logits where
    vocab = vocabulary (modelEmbedding decoder)
    rmsWeight = rmsFinalWeight (modelEmbedding decoder)
    tokenWithRms = rmsNorm tokenVector rmsWeight
    CArray2D vocabRows = vocab
    logits = map (`dotProduct` tokenWithRms) vocabRows

-- One transformer layer running across multiple cycles:
--   Cycle2: compute Q/K/V
--   Cycle3: run attention heads with streaming RAM reads
--   Cycle4: write K/V to RAM
--   Cycle5: feed-forward network
-- Handles per-head attention and per-bank K/V cache writes.
multiCycleTransformerLayer
  :: HiddenClockResetEnable dom
  => TransformerLayerComponent
  -> Cache.KVRamOwner dom
  -> Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom IntermediateData
  -> ( Signal dom IntermediateData
     , Signal dom Bool     -- writeDone (Cycle4)
     , Signal dom Bool     -- attnDone  (Cycle3)
     , Signal dom IntermediateData) -- commitCycle3 (gated write-back)
multiCycleTransformerLayer transformerLayerComponent kvRamOwner layerIndex processingStateSignal intermediateDataSignal =
  (nextIntermediateDataSignal, writeDoneThisLayerSignal, attentionDoneThisLayerSignal, commitCycle3Signal)
 where
  multiHeadAttentionComponent  = multiHeadAttention transformerLayerComponent
  feedForwardNetworkComponent = feedforwardNetwork transformerLayerComponent

  -- For each KV bank: two attention runners share the bank; same bank’s writer runs in Cycle4.
  (perHeadOutputSignalsVec, perHeadDoneSignalsVec, perBankWriteDoneVec) =
    let
      initHeadOutputs  = repeat (pure (repeat 0)) :: Vec NumQueryHeads (Signal dom (Vec HeadDimension Float))
      initHeadDone     = repeat (pure False)      :: Vec NumQueryHeads (Signal dom Bool)
      initWriteDone    = repeat (pure False)      :: Vec NumKeyValueHeads (Signal dom Bool)
    in P.foldl
         (Layer.fillOneBank layerIndex processingStateSignal kvRamOwner intermediateDataSignal)
         (initHeadOutputs, initHeadDone, initWriteDone)
         indicesI

  -- Attention done for this layer = all heads done (rising edge)
  allHeadsDoneSignal          = fmap P.and (sequenceA perHeadDoneSignalsVec)
  allHeadsDonePreviousSignal  = register False allHeadsDoneSignal
  attentionDoneThisLayerSignal =
    liftA2 (\now prev -> now && not prev) allHeadsDoneSignal allHeadsDonePreviousSignal

  -- Project each head with its W_O slice, then sum across heads
  perHeadProjectedSignalsVec = zipWith
                                 (\wo hSig -> matrixVectorMult wo <$> hSig)
                                 (MultiHeadAttention.mWo multiHeadAttentionComponent)
                                 perHeadOutputSignalsVec
  perHeadProjectedSignal = sequenceA perHeadProjectedSignalsVec
  attentionSumSignal     = fmap (foldl1 (zipWith (+))) perHeadProjectedSignal

  -- Residual: x + sum_h(W_O @ h)
  attentionOutputSignal =
    (zipWith (+) P.. inputVector P.<$> intermediateDataSignal) <*> attentionSumSignal

  -- Commit attention output only on this layer’s attn done pulse in Cycle3
  nextIntermediateDataSignal =
    liftA4
      (\procState currentIntermediateData attentionOutput doneSignal ->
         if processingLayer procState == layerIndex && processingStage procState == Cycle3_ComputeAttention && doneSignal
           then currentIntermediateData { attentionOutput = attentionOutput }
           else currentIntermediateData)
      processingStateSignal baseNextIntermediateDataSignal attentionOutputSignal attentionDoneThisLayerSignal

  -- Layer write done = AND across banks, qualified to this layer/stage
  writeDoneThisLayerSignal =
    let allBanksDoneSignal = fmap P.and (sequenceA perBankWriteDoneVec)
    in  (\procState banksDone ->
           processingStage procState == Cycle4_WriteCache &&
           processingLayer procState == layerIndex &&
           banksDone)
        <$> processingStateSignal <*> allBanksDoneSignal

  -- Same gated “commit” view used by the outer pipeline
  commitCycle3Signal =
    liftA4
      (\procState currentIntermediateData attentionOutput doneSignal ->
        if processingLayer procState == layerIndex && processingStage procState == Cycle3_ComputeAttention && doneSignal
          then currentIntermediateData { attentionOutput = attentionOutput }
          else currentIntermediateData)
      processingStateSignal intermediateDataSignal attentionOutputSignal attentionDoneThisLayerSignal

  -- Default per-stage work within this layer
  baseNextIntermediateDataSignal =
    liftA2 (Layer.processCycle multiHeadAttentionComponent feedForwardNetworkComponent layerIndex)
           processingStateSignal
           intermediateDataSignal
