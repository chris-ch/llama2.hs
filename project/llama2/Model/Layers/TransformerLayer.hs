module Model.Layers.TransformerLayer (
    multiCycleTransformerLayer
    , TransformerDecoderComponent(..)
    , TransformerLayerComponent(..)
) where

import Clash.Prelude
import Model.Core.Types (ModelDim, EmbeddingComponent (..),
    NumLayers, NumQueryHeads, NumKeyValueHeads,
    HeadDimension, ProcessingState (..), IntermediateData (..), CycleStage (..))
import Helpers (rmsNorm, matrixVectorMult, liftA4)

import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.FeedForward.FeedForwardNetwork as FeedForwardNetwork
import qualified Model.Layers.Attention.MultiHeadAttention as MultiHeadAttention
import qualified Model.Layers.Attention.AttentionHead as AttentionHead
import qualified Model.Layers.Attention.MultiHeadAttention as MultiAttentionHead

data TransformerLayerComponent = TransformerLayerComponent
  {
    multiHeadAttention :: MultiHeadAttention.MultiHeadAttentionComponent,
    feedforwardNetwork :: FeedForwardNetwork.FeedForwardNetworkComponent
  } deriving (Show)

data TransformerDecoderComponent = TransformerDecoderComponent
  { modelEmbedding :: EmbeddingComponent,
    modelLayers :: Vec NumLayers TransformerLayerComponent
  } deriving (Show)

multiCycleTransformerLayer
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
multiCycleTransformerLayer transformerLayerComponent kvRamOwner layerIndex processingStateSignal intermediateDataSignal =
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
         (fillOneBank layerIndex processingStateSignal kvRamOwner intermediateDataSignal)
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
    (zipWith (+) . inputVector <$> intermediateDataSignal) <*> woHeadsSignal

  -- Commit attention output only on this layer’s attn done pulse in Stage3
  nextIntermediateDataSignal =
    liftA4
      (\procState currentIntermediateData attentionOutput doneSignal ->
         if processingLayer procState == layerIndex
            && processingStage procState == Stage3_Attend
            && doneSignal
           then currentIntermediateData { attentionOutput = attentionOutput }
           else currentIntermediateData)
      processingStateSignal baseNextIntermediateDataSignal xAfterAttnSignal attentionDoneThisLayerSignal

  -- Layer write done = AND across banks (Stage4 writeback)
  writeDoneThisLayerSignal =
    let allBanksDoneSignal = fmap and (sequenceA perBankWriteDoneVec)
    in  (\procState banksDone ->
           processingStage procState == Stage4_WriteKV  &&
           processingLayer procState == layerIndex &&
           banksDone)
        <$> processingStateSignal <*> allBanksDoneSignal

  -- Same gated commit view for Cycle3
  commitCycle3Signal =
    liftA4
      (\procState currentIntermediateData attentionOutput doneSignal ->
        if processingLayer procState == layerIndex
           && processingStage procState == Stage3_Attend
           && doneSignal
          then currentIntermediateData { attentionOutput = attentionOutput }
          else currentIntermediateData)
      processingStateSignal intermediateDataSignal xAfterAttnSignal attentionDoneThisLayerSignal

  -- Default per-stage work within this layer
  baseNextIntermediateDataSignal =
    liftA2 (processStage multiHeadAttentionComponent feedForwardNetworkComponent layerIndex)
           processingStateSignal
           intermediateDataSignal

processStage
  :: MultiHeadAttention.MultiHeadAttentionComponent
  -> FeedForwardNetwork.FeedForwardNetworkComponent
  -> Index NumLayers
  -> ProcessingState
  -> IntermediateData
  -> IntermediateData
processStage multiHeadAttentionComponent feedForwardNetworkComponent layerIndex processingState intermediateData
  | processingLayer processingState /= layerIndex = intermediateData
  | otherwise =
      case processingStage processingState of
        Stage1_LoadKV -> intermediateData
        Stage2_ProjectQKV -> 
          let
            (queries, keys, values) = MultiAttentionHead.projectQKV multiHeadAttentionComponent processingState (inputVector intermediateData)
          in  intermediateData { queryVectors = queries, keyVectors = keys, valueVectors = values }
        Stage3_Attend -> intermediateData
        Stage4_WriteKV  -> intermediateData
        Stage5_FeedForward  ->
          let ffnOutput = FeedForwardNetwork.computeFeedForward feedForwardNetworkComponent (attentionOutput intermediateData)
          in intermediateData { feedForwardOutput = ffnOutput }

-- Query heads per KV head
queryHeadsPerKeyValueHead :: Int
queryHeadsPerKeyValueHead = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads

maxQueryHeadIndex :: Int
maxQueryHeadIndex = natToNum @NumQueryHeads - 1

baseQueryIndex :: Index NumKeyValueHeads -> Int
baseQueryIndex keyValueHeadIndex = fromEnum keyValueHeadIndex * queryHeadsPerKeyValueHead

queryHeadIndex0 :: Index NumKeyValueHeads -> Index NumQueryHeads
queryHeadIndex0 keyValueHeadIndex = toEnum (min maxQueryHeadIndex (baseQueryIndex keyValueHeadIndex))

hasSecondQueryHead :: Index NumKeyValueHeads -> Bool
hasSecondQueryHead keyValueHeadIndex = queryHeadsPerKeyValueHead >= 2 && (baseQueryIndex keyValueHeadIndex + 1 <= maxQueryHeadIndex)

queryHeadIndex1 :: Index NumKeyValueHeads -> Index NumQueryHeads
queryHeadIndex1 keyValueHeadIndex = if hasSecondQueryHead keyValueHeadIndex then toEnum (baseQueryIndex keyValueHeadIndex + 1) else queryHeadIndex0 keyValueHeadIndex

-- Access the per-head vectors from IntermediateData for use in attention.
getQueryVector :: Signal dom IntermediateData -> Index NumQueryHeads -> Signal dom (Vec HeadDimension Float)
getQueryVector intermediateDataSignal queryHeadIndex =
  fmap (\idata -> queryVectors idata !! queryHeadIndex) intermediateDataSignal

getKeyVector :: Signal dom IntermediateData -> Index NumKeyValueHeads -> Signal dom (Vec HeadDimension Float)
getKeyVector intermediateDataSignal keyValueHeadIndex =
  fmap (\idata -> keyVectors idata !! keyValueHeadIndex) intermediateDataSignal

getValueVector :: Signal dom IntermediateData -> Index NumKeyValueHeads -> Signal dom (Vec HeadDimension Float)
getValueVector intermediateDataSignal keyValueHeadIndex =
  fmap (\idata -> valueVectors idata !! keyValueHeadIndex) intermediateDataSignal

fillOneBank
  :: HiddenClockResetEnable dom
  => Index NumLayers
  -> Signal dom ProcessingState
  -> Cache.KVRamOwner dom
  -> Signal dom IntermediateData
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension Float))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool))
  -> Index NumKeyValueHeads
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension Float))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool))
fillOneBank layerIndex processingStateSignal kvRamOwner intermediateDataSignal (headOutputAcc, headDoneAcc, writeDoneAcc) keyValueHeadIndex =
  let
    stageEquals st =
      liftA2 (\ps _ -> processingStage ps == st && processingLayer ps == layerIndex)
             processingStateSignal (pure ())

    isStage3Attention = stageEquals Stage3_Attend
    isStage4Write     = stageEquals Stage4_WriteKV

    seqPosSignal = sequencePosition <$> processingStateSignal

    bank   = Cache.kvBanks kvRamOwner !! keyValueHeadIndex
    runKey = Cache.runKeyBank bank
    runVal = Cache.runValueBank bank

    -- Which query heads map to this KV head
    qIdx0 = queryHeadIndex0 keyValueHeadIndex
    hasQ1 = hasSecondQueryHead keyValueHeadIndex
    qIdx1 = queryHeadIndex1 keyValueHeadIndex

    query0 = getQueryVector intermediateDataSignal qIdx0
    query1 = if hasQ1 then getQueryVector intermediateDataSignal qIdx1 else pure (repeat 0)

    keyVec   = getKeyVector   intermediateDataSignal keyValueHeadIndex
    valueVec = getValueVector intermediateDataSignal keyValueHeadIndex

    -- Stage3: run two attention engines in parallel (one per RAM port)
    (addrA0, out0, _busy0, done0) =
      AttentionHead.streamHeadAttentionAddrIO
        isStage3Attention seqPosSignal query0 keyVec valueVec keyOutA valOutA

    (addrB1, out1raw, _busy1, done1raw) =
      AttentionHead.streamHeadAttentionAddrIO
        isStage3Attention seqPosSignal query1 keyVec valueVec keyOutB valOutB

    out1  = if hasQ1 then out1raw else pure (repeat 0)
    done1 = if hasQ1 then done1raw else done0
    doneBoth = done0 .&&. done1      -- align pulses for this bank

    -- Stage4: write K,V for current pos (one element per cycle) on Port B
    keyValuePairSignal = liftA2 (,) keyVec valueVec
    (writeAddrSig, keyWriteSig, valWriteSig, writeDoneThisBank) =
      Cache.writeSequencer isStage4Write seqPosSignal keyValuePairSignal

    -- Dual-port RAM wiring
    --   Stage3: Port A <- head0 addresses, Port B <- head1 addresses, no writes
    --   Stage4: Port B <- write sequencer; Port A parked (read-only, no write)
    addrA = mux isStage3Attention addrA0 (pure 0)          -- park A in Stage4
    addrB = mux isStage3Attention addrB1 writeAddrSig

    wrK_A = pure Nothing
    wrV_A = pure Nothing
    wrK_B = mux isStage4Write keyWriteSig (pure Nothing)
    wrV_B = mux isStage4Write valWriteSig (pure Nothing)

    (keyOutA, _keyOutB_unused) = runKey (addrA, wrK_A) (addrB, wrK_B)
    (valOutA, _valOutB_unused) = runVal (addrA, wrV_A) (addrB, wrV_B)
    -- IMPORTANT: head0 consumes Port A readouts; head1 consumes Port B readouts
    -- For head1, re-run the same RAMs’ Port B outputs as inputs to the streamers:
    keyOutB = snd (runKey (addrA, wrK_A) (addrB, wrK_B))
    valOutB = snd (runVal (addrA, wrV_A) (addrB, wrV_B))

    -- Accumulate outputs and aligned done pulses
    headOutputAcc0 = replace qIdx0 out0 headOutputAcc
    headDoneAcc0   = replace qIdx0 doneBoth headDoneAcc
    headOutputAcc1 = if hasQ1 then replace qIdx1 out1 headOutputAcc0 else headOutputAcc0
    headDoneAcc1   = if hasQ1 then replace qIdx1 doneBoth headDoneAcc0 else headDoneAcc0

    writeDoneAcc1  = replace keyValueHeadIndex writeDoneThisBank writeDoneAcc

  in
    (headOutputAcc1, headDoneAcc1, writeDoneAcc1)
