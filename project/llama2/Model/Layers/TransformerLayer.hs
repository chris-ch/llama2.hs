module Model.Layers.TransformerLayer (
    transformerLogits
    , multiCycleTransformerLayer
    , multiCycleTransformerLayerTap
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

-- ====== NEW: tapped layer variant ======
-- Emits a 1-cycle pulse at the end of Cycle3 for this layer with:
--   - dbgXHat:        x after rms_att (same shape as ModelDim)
--   - dbgWoHeads:     sum_h (W_O^h @ head_h) BEFORE residual
--   - dbgXAfterAttn:  x + sum_h(...)  (AFTER residual)
-- All three vectors are the exact quantities you see as
-- (a) xHat, (c) WO@head_concat, (d) x_after_attn in the C trace.
multiCycleTransformerLayerTap
  :: HiddenClockResetEnable dom
  => TransformerLayerComponent
  -> Cache.KVRamOwner dom
  -> Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom IntermediateData
  -> ( Signal dom IntermediateData
     , Signal dom Bool                         -- writeDone (Cycle4)
     , Signal dom Bool                         -- attnDone  (Cycle3, rising)
     , Signal dom IntermediateData             -- commitCycle3 (gated write-back)
     , Signal dom Bool                         -- tapPulse  (1-cycle on attnDone rising)
     , Signal dom (Vec ModelDim Float)         -- dbgXHat
     , Signal dom (Vec ModelDim Float)         -- dbgWoHeads
     , Signal dom (Vec ModelDim Float) )       -- dbgXAfterAttn
multiCycleTransformerLayerTap transformerLayerComponent kvRamOwner layerIndex processingStateSignal intermediateDataSignal =
  ( nextIntermediateDataSignal
  , writeDoneThisLayerSignal
  , attentionDoneThisLayerSignal
  , commitCycle3Signal
  , attentionDoneThisLayerSignal    -- tap pulse = attn done rising
  , xHatSignal
  , woHeadsSignal
  , xAfterAttnSignal
  )
 where
  multiHeadAttentionComponent  = multiHeadAttention transformerLayerComponent
  feedForwardNetworkComponent  = feedforwardNetwork transformerLayerComponent

  -- Per-bank attention and KV writes (unchanged from your original function)
  (perHeadOutputSignalsVec, perHeadDoneSignalsVec, perBankWriteDoneVec) =
    let
      initHeadOutputs  = repeat (pure (repeat 0))
      initHeadDone     = repeat (pure False)
      initWriteDone    = repeat (pure False)
    in foldl
         (Layer.fillOneBank layerIndex processingStateSignal kvRamOwner intermediateDataSignal)
         (initHeadOutputs, initHeadDone, initWriteDone)
         indicesI

  -- Rising-edge "all heads done" (Cycle3) (same as original)
  allHeadsDoneSignal         = fmap and (sequenceA perHeadDoneSignalsVec)
  allHeadsDonePrevSignal     = register False allHeadsDoneSignal
  attentionDoneThisLayerSignal =
    liftA2 (\now prev -> now && not prev) allHeadsDoneSignal allHeadsDonePrevSignal

  -- Re-compute xHat here (pure, cheap) so we can tap it:
  -- xHat = rmsNorm(x, rms_att)
  xHatSignal =
    (\idata -> rmsNorm (inputVector idata) (MultiHeadAttention.rmsAtt multiHeadAttentionComponent))
      <$> intermediateDataSignal

  -- Per-head W_O @ head and sum
  perHeadProjectedSignalsVec =
    zipWith (\wo hSig -> matrixVectorMult wo <$> hSig)
            (MultiHeadAttention.mWo multiHeadAttentionComponent)
            perHeadOutputSignalsVec
  perHeadProjectedSignal = sequenceA perHeadProjectedSignalsVec
  woHeadsSignal          = fmap (foldl1 (zipWith (+))) perHeadProjectedSignal

  -- x_after_attn = x + WO@heads
  xAfterAttnSignal =
    (zipWith (+) <$> (inputVector <$> intermediateDataSignal)) <*> woHeadsSignal

  -- Commit attention output only on this layer’s attn done pulse in Cycle3
  nextIntermediateDataSignal =
    liftA4
      (\procState currentIntermediateData attentionOutput doneSignal ->
         if processingLayer procState == layerIndex
            && processingStage procState == Cycle3_ComputeAttention
            && doneSignal
           then currentIntermediateData { attentionOutput = attentionOutput }
           else currentIntermediateData)
      processingStateSignal baseNextIntermediateDataSignal xAfterAttnSignal attentionDoneThisLayerSignal

  -- Layer write done = AND across banks (as in your original)
  writeDoneThisLayerSignal =
    let allBanksDoneSignal = fmap and (sequenceA perBankWriteDoneVec)
    in  (\procState banksDone ->
           processingStage procState == Cycle4_WriteCache &&
           processingLayer procState == layerIndex &&
           banksDone)
        <$> processingStateSignal <*> allBanksDoneSignal

  -- Same gated commit view for Cycle3
  commitCycle3Signal =
    liftA4
      (\procState currentIntermediateData attentionOutput doneSignal ->
        if processingLayer procState == layerIndex
           && processingStage procState == Cycle3_ComputeAttention
           && doneSignal
          then currentIntermediateData { attentionOutput = attentionOutput }
          else currentIntermediateData)
      processingStateSignal intermediateDataSignal xAfterAttnSignal attentionDoneThisLayerSignal

  -- Default per-stage work within this layer (unchanged)
  baseNextIntermediateDataSignal =
    liftA2 (Layer.processCycle multiHeadAttentionComponent feedForwardNetworkComponent layerIndex)
           processingStateSignal
           intermediateDataSignal

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
