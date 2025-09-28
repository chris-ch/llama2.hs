module Model.Layers.TransformerLayer (
    multiCycleTransformerLayer
  , TransformerDecoderComponent(..)
  , TransformerLayerComponent(..)
) where

import Clash.Prelude
import Model.Core.Types
  ( ModelDim, EmbeddingComponent(..)
  , NumLayers, NumQueryHeads, NumKeyValueHeads
  , HeadDimension, ProcessingState(..), IntermediateData(..), CycleStage(..)
  )
import Helpers (rmsNorm, matrixVectorMult, liftA4)

import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.FeedForward.FeedForwardNetwork as FeedForwardNetwork
import qualified Model.Layers.Attention.MultiHeadAttention as MultiHeadAttention
import Model.Layers.Attention.MultiHeadAttention (StepCount(..))
import qualified Model.Layers.Attention.AttentionHead (attendHead)

data TransformerLayerComponent = TransformerLayerComponent
  { multiHeadAttention :: MultiHeadAttention.MultiHeadAttentionComponent
  , feedforwardNetwork :: FeedForwardNetwork.FeedForwardNetworkComponent
  } deriving (Show)

data TransformerDecoderComponent = TransformerDecoderComponent
  { modelEmbedding :: EmbeddingComponent
  , modelLayers    :: Vec NumLayers TransformerLayerComponent
  } deriving (Show)

multiCycleTransformerLayer
  :: HiddenClockResetEnable dom
  => TransformerLayerComponent
  -> Cache.KVRamOwner dom
  -> Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom IntermediateData
  -> ( Signal dom IntermediateData
     , Signal dom Bool                         -- writeDone (Stage2_WriteKV)
     , Signal dom Bool                         -- attnDone  (Stage3_Attend, rising)
     , Signal dom IntermediateData             -- commitCycle3 (gated write-back)
     , Signal dom Bool                         -- tapPulse  (1-cycle on attnDone rising)
     , Signal dom (Vec ModelDim Float)         -- dbgXHat
  )
multiCycleTransformerLayer layer kvRamOwner layerIndex processingStateSignal intermediateDataSignal =
  ( nextIntermediateDataSignal
  , writeDoneThisLayerSignal
  , attentionDoneThisLayerSignal
  , commitCycle3Signal
  , attentionDoneThisLayerSignal
  , xHatSignalDbg
  )
 where
  mha  = multiHeadAttention layer
  ffn  = feedforwardNetwork layer

  -- xHat = rmsnorm(x, rms_att) for debugging
  xHatSignalDbg = (\idata -> rmsNorm (inputVector idata) (MultiHeadAttention.rmsAtt mha)) <$> intermediateDataSignal

  -- Drive all KV banks; collect per-head outputs, head-done pulses, and per-bank write-done
  (perHeadOutputSignalsVec, perHeadDoneSignalsVec, perBankWriteDoneVec) =
    let initHeadOutputs = repeat (pure (repeat 0))
        initHeadDone    = repeat (pure False)
        initWriteDone   = repeat (pure False)
    in  foldl
          (fillOneBankComb layerIndex processingStateSignal kvRamOwner intermediateDataSignal)
          (initHeadOutputs, initHeadDone, initWriteDone)
          indicesI

  -- Attention done: rising edge once all heads finish for this layer
  allHeadsDoneSignal     = fmap and (sequenceA perHeadDoneSignalsVec)
  allHeadsDonePrevSignal = register False allHeadsDoneSignal
  attentionDoneThisLayerSignal =
    liftA2 (\now prev -> now && not prev) allHeadsDoneSignal allHeadsDonePrevSignal

  -- Default per-stage work within this layer
  baseNextIntermediateDataSignal =
    liftA2 (processStage mha ffn layerIndex) processingStateSignal intermediateDataSignal

  -- Layer write-done = AND across banks (Stage2_WriteKV)
  writeDoneThisLayerSignal =
    let allBanksDoneSignal = fmap and (sequenceA perBankWriteDoneVec)
    in  (\ps banksDone ->
           processingStage ps == Stage2_WriteKV
        && processingLayer ps == layerIndex
        && banksDone)
        <$> processingStateSignal <*> allBanksDoneSignal

  -- Per-head WO @ head, then sum across heads (equivalent to WO @ concatHeads)
  perHeadProjectedSignalsVec =
    zipWith (\wo hSig -> matrixVectorMult wo <$> hSig) (MultiHeadAttention.mWo mha) perHeadOutputSignalsVec

  perHeadProjectedSignal = sequenceA perHeadProjectedSignalsVec
  woHeadsSignal          = fmap (foldl1 (zipWith (+))) perHeadProjectedSignal

  -- x_after_attn = x + WO@heads
  xAfterAttnSignal =
      liftA2
        (\idata woHeads ->
          let xInput = inputVector idata
              summed = zipWith (+) xInput woHeads
          in summed)
        intermediateDataSignal
        woHeadsSignal


  -- Commit attention output on this layer’s attnDone pulse in Stage3_Attend.
  -- Print the exact vector being committed (first 8 elems) once per (L, P).
  nextIntermediateDataSignal =
    liftA4
      (\ps cur attOut done ->
         if processingLayer ps == layerIndex
            && processingStage ps == Stage3_Attend
            && done
           then cur { attentionOutput = attOut }
           else cur)
      processingStateSignal baseNextIntermediateDataSignal xAfterAttnSignal attentionDoneThisLayerSignal

  -- The same gated-commit view, exposed as a tap at Cycle3 (no trace here to avoid duplicate prints)
  commitCycle3Signal =
    liftA4
      (\ps cur attOut done ->
         if processingLayer ps == layerIndex
            && processingStage ps == Stage3_Attend
            && done
           then cur { attentionOutput = attOut }
           else cur)
      processingStateSignal intermediateDataSignal xAfterAttnSignal attentionDoneThisLayerSignal

processStage
  :: MultiHeadAttention.MultiHeadAttentionComponent
  -> FeedForwardNetwork.FeedForwardNetworkComponent
  -> Index NumLayers
  -> ProcessingState
  -> IntermediateData
  -> IntermediateData
processStage mha ffn layerIndex ps idata
  | processingLayer ps /= layerIndex = idata
  | otherwise = case processingStage ps of

      -- Stage1: compute Q,K,V for current layer/pos
      Stage1_ProjectQKV ->
        let
          stepCount = (StepCount $ fromIntegral $ sequencePosition ps)
          (qs, ks, vs) = MultiHeadAttention.projectQKV mha stepCount (inputVector idata)
        in idata { queryVectors = qs, keyVectors = ks, valueVectors = vs }

      -- Stage2: write K,V(pos) to cache
      Stage2_WriteKV -> idata

      -- Stage3: stream attention (sequenced outside).
      -- NOTE: This prints the previous attentionOutput. The new value is printed at commit.
      Stage3_Attend -> idata

      -- Stage4: FFN
      Stage4_FeedForward ->
        let
          ffnOut = FeedForwardNetwork.computeFeedForward ffn (attentionOutput idata)
        in idata { feedForwardOutput = ffnOut }

      -- Stage5: bookkeeping only
      Stage5_Bookkeeping -> idata


-- Query heads per KV head
queryHeadsPerKeyValueHead :: Int
queryHeadsPerKeyValueHead = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads

maxQueryHeadIndex :: Int
maxQueryHeadIndex = natToNum @NumQueryHeads - 1

baseQueryIndex :: Index NumKeyValueHeads -> Int
baseQueryIndex kvIx = fromEnum kvIx * queryHeadsPerKeyValueHead

queryHeadIndex0 :: Index NumKeyValueHeads -> Index NumQueryHeads
queryHeadIndex0 kvIx = toEnum (min maxQueryHeadIndex (baseQueryIndex kvIx))

hasSecondQueryHead :: Index NumKeyValueHeads -> Bool
hasSecondQueryHead kvIx = queryHeadsPerKeyValueHead >= 2 && (baseQueryIndex kvIx + 1 <= maxQueryHeadIndex)

queryHeadIndex1 :: Index NumKeyValueHeads -> Index NumQueryHeads
queryHeadIndex1 kvIx =
  if hasSecondQueryHead kvIx then toEnum (baseQueryIndex kvIx + 1) else queryHeadIndex0 kvIx

-- Access per-head vectors from IntermediateData
getQueryVector :: Signal dom IntermediateData -> Index NumQueryHeads -> Signal dom (Vec HeadDimension Float)
getQueryVector idSig qIx = (\i -> queryVectors i !! qIx) <$> idSig

getKeyVector :: Signal dom IntermediateData -> Index NumKeyValueHeads -> Signal dom (Vec HeadDimension Float)
getKeyVector idSig kvIx = (\i -> keyVectors i !! kvIx) <$> idSig

getValueVector :: Signal dom IntermediateData -> Index NumKeyValueHeads -> Signal dom (Vec HeadDimension Float)
getValueVector idSig kvIx = (\i -> valueVectors i !! kvIx) <$> idSig

fillOneBankComb
  :: HiddenClockResetEnable dom
  => Index NumLayers
  -> Signal dom ProcessingState
  -> Cache.KVRamOwner dom
  -> Signal dom IntermediateData
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension Float))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool) )
  -> Index NumKeyValueHeads
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension Float))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool) )
fillOneBankComb layerIx psSig kvOwner idSig (headOutAcc, headDoneAcc, writeDoneAcc) kvIx =
  let
    -- Stage predicates
    stageEquals st =
      liftA2 (\ps _ -> processingStage ps == st && processingLayer ps == layerIx)
             psSig (pure ())

    isStage3Attention = stageEquals Stage3_Attend
    isStage2Write     = stageEquals Stage2_WriteKV

    -- pos for this layer
    seqPosSignal = sequencePosition <$> psSig

    -- Bank handles (we still instantiate/wire BRAM for Stage2 writes)
    bank   = Cache.kvBanks kvOwner !! kvIx
    runKey = Cache.runKeyBank bank
    runVal = Cache.runValueBank bank

    -- Which query heads map to this KV head
    qIdx0 = queryHeadIndex0 kvIx
    hasQ1 = hasSecondQueryHead kvIx
    qIdx1 = queryHeadIndex1 kvIx

    query0 = getQueryVector idSig qIdx0
    query1 = if hasQ1 then getQueryVector idSig qIdx1 else pure (repeat 0)

    keyVec   = getKeyVector   idSig kvIx    -- K(pos) row for this KV head (HeadDimension)
    valueVec = getValueVector idSig kvIx    -- V(pos) row for this KV head

    -- Stage2: write K,V(pos) one element per cycle (unchanged wiring)
    keyValuePairSignal = liftA2 (,) keyVec valueVec
    (writeAddrSig, keyWriteSig, valWriteSig, writeDoneThisBank) =
      Cache.writeSequencer isStage2Write seqPosSignal keyValuePairSignal

    -- Dual-port BRAM wiring:
    --   Stage3: park both ports (no reads)
    --   Stage2: Port B <- write sequencer; Port A parked
    addrA = pure 0
    addrB = mux isStage2Write writeAddrSig (pure 0)

    wrK_A = pure Nothing
    wrV_A = pure Nothing
    wrK_B = mux isStage2Write keyWriteSig (pure Nothing)
    wrV_B = mux isStage2Write valWriteSig (pure Nothing)

    -- Instantiate bank BRAMs (we ignore their read data in Stage3)
    (_keyOutA, _keyOutB) = runKey (addrA, wrK_A) (addrB, wrK_B)
    (_valOutA, _valOutB) = runVal (addrA, wrV_A) (addrB, wrV_B)

    -- A tiny KV "mirror" in registers: for each pos, store the whole K(pos)/V(pos) row once.
    -- This lets us call the pure attendHeadComb with the whole window 0..pos.
    -- (Synthesis note: this costs SeqLen * HeadDimension registers per KV head.)
    kvKeysAll = mealy
      (\mem (we, p, rowK) ->
         let mem' = if we then replace p rowK mem else mem
         in  (mem', mem'))
      (repeat (repeat 0))
      (bundle (isStage2Write, seqPosSignal, keyVec))

    kvValsAll = mealy
      (\mem (we, p, rowV) ->
         let mem' = if we then replace p rowV mem else mem
         in  (mem', mem'))
      (repeat (repeat 0))
      (bundle (isStage2Write, seqPosSignal, valueVec))

    -- Combinational attention per query head using the KV mirror
    out0 = liftA4 Model.Layers.Attention.AttentionHead.attendHead query0 kvKeysAll kvValsAll seqPosSignal
    out1raw = liftA4 Model.Layers.Attention.AttentionHead.attendHead query1 kvKeysAll kvValsAll seqPosSignal
    out1 = if hasQ1 then out1raw else pure (repeat 0)

    -- A single 1-cycle done pulse on the rising edge of Stage3_Attend
    attnPrev = register False isStage3Attention
    donePulse = liftA2 (\now prev -> now && not prev) isStage3Attention attnPrev

    -- For “both” heads, the done pulse is shared (they’re computed in the same cycle)
    doneBoth = donePulse

    -- Accumulate outputs and done pulses into the outer vectors
    headOutAcc0  = replace qIdx0 out0 headOutAcc
    headDoneAcc0 = replace qIdx0 doneBoth headDoneAcc
    headOutAcc1  = if hasQ1 then replace qIdx1 out1 headOutAcc0 else headOutAcc0
    headDoneAcc1 = if hasQ1 then replace qIdx1 doneBoth headDoneAcc0 else headDoneAcc0

    writeDoneAcc1 = replace kvIx writeDoneThisBank writeDoneAcc

  in (headOutAcc1, headDoneAcc1, writeDoneAcc1)
