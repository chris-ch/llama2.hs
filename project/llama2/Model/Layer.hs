module Model.Layer
  ( getQueryVector, getKeyVector, getValueVector
  , multiCycleTransformerLayer
  ) where

import Clash.Prelude
import qualified Prelude as P

import Helpers
  ( NumQueryHeads, NumKeyValueHeads, NumLayers, HeadDimension
  , TransformerLayerComponent(..), MultiHeadAttentionComponent(..)
  , runSingleHeadQKV, applyRotaryToHead, StepCount (..)
  , computeFeedForward, rmsNorm, matrixVectorMult, liftA4, FeedForwardNetworkComponent
  )

import Model.Core.Types
  ( ProcessingState(..), IntermediateData(..), CycleStage (..)
  )

import qualified Model.Attention as Attention
import qualified Model.Cache as Cache

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
         (fillOneBank layerIndex processingStateSignal kvRamOwner intermediateDataSignal)
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
                                 (mWo multiHeadAttentionComponent)
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
    liftA2 (processCycle multiHeadAttentionComponent feedForwardNetworkComponent layerIndex)
           processingStateSignal
           intermediateDataSignal

-- For one K/V bank:
--   - Runs attention for one or two query heads mapped to this bank
--   - Reads K/V from RAM (or bypass at current t)
--   - Writes new K/V during Cycle4
-- Accumulates head outputs and per-bank writeDone pulses.
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
    stageEquals cycleStage =
      liftA2 (\procState _ -> processingStage procState == cycleStage && processingLayer procState == layerIndex)
             processingStateSignal (pure ())

    isCycle3Attention = stageEquals Cycle3_ComputeAttention
    isCycle4Write     = stageEquals Cycle4_WriteCache
    attentionStartSignal = isCycle3Attention
    sequencePositionSignal = sequencePosition <$> processingStateSignal

    bank   = Cache.kvBanks kvRamOwner !! keyValueHeadIndex
    runKey = Cache.runKeyBank bank
    runVal = Cache.runValueBank bank

    queryVectorSignal0 = getQueryVector intermediateDataSignal (queryHeadIndex0 keyValueHeadIndex)
    queryVectorSignal1 = if hasSecondQueryHead keyValueHeadIndex then getQueryVector intermediateDataSignal (queryHeadIndex1 keyValueHeadIndex)
                                               else pure (repeat 0)

    keyVectorSignal   = getKeyVector intermediateDataSignal keyValueHeadIndex
    valueVectorSignal = getValueVector intermediateDataSignal keyValueHeadIndex

    (address0, headOutput0, _busy0, done0) =
      Attention.streamHeadAttentionAddrIO
        attentionStartSignal sequencePositionSignal
        queryVectorSignal0 keyVectorSignal valueVectorSignal keyRamOutput0 valueRamOutput0
    (address1, headOutput1, _busy1, done1) =
      Attention.streamHeadAttentionAddrIO
        attentionStartSignal sequencePositionSignal
        queryVectorSignal1 keyVectorSignal valueVectorSignal keyRamOutput1 valueRamOutput1

    keyValuePairSignal                      = liftA2 (,) keyVectorSignal valueVectorSignal
    (writeAddressSignal, keyWriteSignal, valueWriteSignal, writeDoneBankSignal) =
      Cache.writeSequencer isCycle4Write sequencePositionSignal keyValuePairSignal

    addressA = address0
    writeA   = pure Nothing
    addressBForCycle3 = if hasSecondQueryHead keyValueHeadIndex then address1 else address0
    addressB = mux isCycle3Attention addressBForCycle3 writeAddressSignal
    keyWriteB  = mux isCycle3Attention (pure Nothing) keyWriteSignal
    valueWriteB= mux isCycle3Attention (pure Nothing) valueWriteSignal

    (keyRamOutput0, keyRamOutput1) = runKey (addressA, writeA) (addressB, keyWriteB)
    (valueRamOutput0, valueRamOutput1) = runVal (addressA, writeA) (addressB, valueWriteB)

    headOutputAcc0  = replace (queryHeadIndex0 keyValueHeadIndex) headOutput0 headOutputAcc
    headDoneAcc0    = replace (queryHeadIndex0 keyValueHeadIndex) done0 headDoneAcc

    headOutputAcc1  = if hasSecondQueryHead keyValueHeadIndex then replace (queryHeadIndex1 keyValueHeadIndex) headOutput1 headOutputAcc0 else headOutputAcc0
    headDoneAcc1    = if hasSecondQueryHead keyValueHeadIndex then replace (queryHeadIndex1 keyValueHeadIndex) done1 headDoneAcc0 else headDoneAcc0

    writeDoneAcc1   = replace keyValueHeadIndex writeDoneBankSignal writeDoneAcc
  in
    (headOutputAcc1, headDoneAcc1, writeDoneAcc1)

-- Stateless per-cycle computation for a single layer:
--   - In Cycle2 computes Q/K/V (with RoPE)
--   - In Cycle5 computes feed-forward network
--   - Otherwise passes IntermediateData through unchanged
processCycle
  :: MultiHeadAttentionComponent
  -> FeedForwardNetworkComponent
  -> Index NumLayers
  -> ProcessingState
  -> IntermediateData
  -> IntermediateData
processCycle multiHeadAttentionComponent feedForwardNetworkComponent layerIndex processingState intermediateData
  | processingLayer processingState /= layerIndex = intermediateData
  | otherwise =
      case processingStage processingState of
        Cycle1_ReadCache -> intermediateData
        Cycle2_ComputeQKV ->
          let
            normalizedInput = rmsNorm (inputVector intermediateData) (rmsAtt multiHeadAttentionComponent)
            -- Queries: one per Q head (with RoPE on Q)
            queries =
              imap (\queryHeadIdx _ ->
                let headComponent = heads multiHeadAttentionComponent !! queryHeadIdx
                    (query, _, _)   = runSingleHeadQKV headComponent normalizedInput
                    (queryRotated, _kU) =
                      applyRotaryToHead headComponent
                                        (StepCount $ fromIntegral $ sequencePosition processingState)
                                        (query, repeat 0)
                in queryRotated) indicesI

            -- Keys/Values: one per KV head (apply RoPE to K only)
            keysAndValues =
              imap (\keyValueHeadIdx _ ->
                let qIdx0 = fromEnum keyValueHeadIdx *
                              (natToNum @NumQueryHeads `P.div` natToNum @NumKeyValueHeads)
                    queryIndex = toEnum (min (natToNum @NumQueryHeads - 1) qIdx0) :: Index NumQueryHeads
                    headComponent     = heads multiHeadAttentionComponent !! queryIndex
                    (_q, key, value)   = runSingleHeadQKV headComponent normalizedInput
                    (_qU, keyRotated)  =
                      applyRotaryToHead headComponent
                                        (StepCount $ fromIntegral $ sequencePosition processingState)
                                        (repeat 0, key)
                in (keyRotated, value)) indicesI
            (keys, values) = unzip keysAndValues
          in intermediateData { queryVectors = queries, keyVectors = keys, valueVectors = values }
        Cycle3_ComputeAttention -> intermediateData
        Cycle4_WriteCache  -> intermediateData
        Cycle5_ComputeFeedForward  ->
          let ffnOutput = computeFeedForward feedForwardNetworkComponent (attentionOutput intermediateData)
          in intermediateData { feedForwardOutput = ffnOutput }
