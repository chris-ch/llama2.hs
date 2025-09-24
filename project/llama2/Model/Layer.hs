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

import qualified Model.Memory.KVCacheBank as Cache (KVRamOwner (..), writeSequencer, KvBank (..))
import qualified Model.Layers.FeedForward.FeedForwardNetwork as FeedForwardNetwork (FeedForwardNetworkComponent, computeFeedForward)
import qualified Model.Layers.Attention.MultiHeadAttention as MultiHeadAttention (MultiHeadAttentionComponent (..))
import qualified Model.Layers.Attention.AttentionHead as AttentionHead (streamHeadAttentionAddrIO)

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

    isCycle3Attention = stageEquals Cycle3_ComputeAttention
    isCycle4Write     = stageEquals Cycle4_WriteCache

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

    -- Cycle3: two attention engines, one per RAM port (read-only)
    (addrA0, out0, _busy0, done0) =
      AttentionHead.streamHeadAttentionAddrIO
        isCycle3Attention seqPosSignal query0 keyVec valueVec keyOutA valOutA

    (addrB1, out1raw, _busy1, done1raw) =
      AttentionHead.streamHeadAttentionAddrIO
        isCycle3Attention seqPosSignal query1 keyVec valueVec keyOutB valOutB

    -- If there is no second head on this bank, treat head1 as “instantly done” and zero output
    out1  = if hasQ1 then out1raw else pure (repeat 0)
    done1 = if hasQ1 then done1raw else done0

    -- Align both head-done pulses to the same cycle so the layer-wide AND reduces cleanly
    doneBoth = done0 .&&. done1

    -- Cycle4: write K,V for current pos (one element per cycle) on Port B
    keyValuePairSignal = liftA2 (,) keyVec valueVec
    (writeAddrSig, keyWriteSig, valWriteSig, writeDoneThisBank) =
      Cache.writeSequencer isCycle4Write seqPosSignal keyValuePairSignal

    -- Dual-port RAM wiring
    --   Cycle3: Port A -> head0 addresses, Port B -> head1 addresses, no writes
    --   Cycle4: Port B -> KV writes, Port A idle read
    addrA = mux isCycle3Attention addrA0 writeAddrSig
    addrB = mux isCycle3Attention addrB1 writeAddrSig

    wrK_A = pure Nothing
    wrV_A = pure Nothing
    wrK_B = mux isCycle4Write keyWriteSig (pure Nothing)
    wrV_B = mux isCycle4Write valWriteSig (pure Nothing)

    (keyOutA, keyOutB) = runKey (addrA, wrK_A) (addrB, wrK_B)
    (valOutA, valOutB) = runVal (addrA, wrV_A) (addrB, wrV_B)

    -- Accumulate outputs and aligned done pulses for just the heads this bank owns
    headOutputAcc0 = replace qIdx0 out0 headOutputAcc
    headDoneAcc0   = replace qIdx0 doneBoth headDoneAcc
    headOutputAcc1 = if hasQ1 then replace qIdx1 out1 headOutputAcc0 else headOutputAcc0
    headDoneAcc1   = if hasQ1 then replace qIdx1 doneBoth headDoneAcc0 else headDoneAcc0

    writeDoneAcc1  = replace keyValueHeadIndex writeDoneThisBank writeDoneAcc
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
