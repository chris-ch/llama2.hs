module Model.Layer
  ( StepCount(..),
    getQueryVector, getKeyVector, getValueVector, processCycle, fillOneBank, applyRotaryToHead
  ) where

import Clash.Prelude
import qualified Prelude as P

import Helpers
  ( NumQueryHeads, NumKeyValueHeads, NumLayers, HeadDimension
  , rmsNorm, FreqDim, RotaryEncodingComponent (..), SingleHeadComponent (..), CArray2D (..), ModelDim, matrixVectorMult
  )

import Model.Core.Types
  ( ProcessingState(..), IntermediateData(..), CycleStage (..)
  )

import qualified Model.Attention as Attention
import qualified Model.Memory.KVCacheBank as Cache (KVRamOwner (..), writeSequencer, KvBank (..))
import qualified Model.Layers.FeedForward.FeedForwardNetwork as FeedForwardNetwork (FeedForwardNetworkComponent, computeFeedForward)
import qualified Model.Layers.Attention.MultiHeadAttention as MultiHeadAttention

newtype StepCount = StepCount (Unsigned 32) deriving (Show, Eq, Ord)

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
  :: MultiHeadAttention.MultiHeadAttentionComponent
  -> FeedForwardNetwork.FeedForwardNetworkComponent
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
            normalizedInput = rmsNorm (inputVector intermediateData) (MultiHeadAttention.rmsAtt multiHeadAttentionComponent)
            -- Queries: one per Q head (with RoPE on Q)
            queries =
              imap (\queryHeadIdx _ ->
                let headComponent = MultiHeadAttention.heads multiHeadAttentionComponent !! queryHeadIdx
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
                    headComponent     = MultiHeadAttention.heads multiHeadAttentionComponent !! queryIndex
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
          let ffnOutput = FeedForwardNetwork.computeFeedForward feedForwardNetworkComponent (attentionOutput intermediateData)
          in intermediateData { feedForwardOutput = ffnOutput }

-- Rotary application
applyRotaryToHead :: SingleHeadComponent -> StepCount -> (Vec HeadDimension Float , Vec HeadDimension Float ) -> (Vec HeadDimension Float , Vec HeadDimension Float )
applyRotaryToHead headComp step (q, k) = (q', k') where
    rot = rotary headComp
    q' = applyRotation rot step q
    k' = applyRotation rot step k

-- Apply rotation per head
applyRotation
  :: RotaryEncodingComponent
  -> StepCount
  -> Vec HeadDimension Float
  -> Vec HeadDimension Float
applyRotation rot step tokenVec =
  let cosFrequencies = getRow step (freqCos rot)
      sinFrequencies = getRow step (freqSin rot)
  in applyRotaryPositionEncoding tokenVec cosFrequencies sinFrequencies

applyRotaryPositionEncoding :: Vec HeadDimension Float    -- input vector
  -> Vec FreqDim Float  -- cosFrequencies
  -> Vec FreqDim Float  -- sinFrequencies
  -> Vec HeadDimension Float
applyRotaryPositionEncoding inputVec cosVec sinVec =
  imap rotatePair inputVec
  where
    rotatePair :: KnownNat headDim => Index headDim -> Float -> Float
    rotatePair i _
        | even idx = rotatedReal
        | otherwise = rotatedImag
        where
            idx :: Int
            idx = fromIntegral i
            pairIdx = idx `div` 2
            realComponent = inputVec !! (2 * pairIdx)
            imagComponent = inputVec !! (2 * pairIdx + 1)
            cosValue = cosVec !! pairIdx
            sinValue = sinVec !! pairIdx
            rotatedReal = realComponent * cosValue - imagComponent * sinValue
            rotatedImag = realComponent * sinValue + imagComponent * cosValue

-- Index into CArray2D to get a row
getRow :: forall n m. (KnownNat n) => StepCount -> CArray2D n m -> Vec m Float
getRow (StepCount i) (CArray2D arr) = arr !! (fromIntegral i :: Index n)

-- QKV per head - should return HeadDimension vectors, not ModelDim
runSingleHeadQKV :: SingleHeadComponent -> Vec ModelDim Float -> (Vec HeadDimension Float, Vec HeadDimension Float, Vec HeadDimension Float)
runSingleHeadQKV headComp normalizedInput = (q, k, v) where
    q = matrixVectorMult (wqHead headComp) normalizedInput  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    k = matrixVectorMult (wkHead headComp) normalizedInput  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    v = matrixVectorMult (wvHead headComp) normalizedInput  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
