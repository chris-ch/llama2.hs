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
import qualified Model.Layers.Attention.AttentionHead as AttentionHead

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
  , xHatSignal
  )
 where
  mha  = multiHeadAttention layer
  ffn  = feedforwardNetwork layer

  -- Drive all KV banks; collect per-head outputs, head-done pulses, and per-bank write-done
  (perHeadOutputSignalsVec, perHeadDoneSignalsVec, perBankWriteDoneVec) =
    let initHeadOutputs = repeat (pure (repeat 0))
        initHeadDone    = repeat (pure False)
        initWriteDone   = repeat (pure False)
    in  foldl
          (fillOneBank layerIndex processingStateSignal kvRamOwner intermediateDataSignal)
          (initHeadOutputs, initHeadDone, initWriteDone)
          indicesI

  -- Attention done: rising edge once all heads finish for this layer
  allHeadsDoneSignal     = fmap and (sequenceA perHeadDoneSignalsVec)
  allHeadsDonePrevSignal = register False allHeadsDoneSignal
  attentionDoneThisLayerSignal =
    liftA2 (\now prev -> now && not prev) allHeadsDoneSignal allHeadsDonePrevSignal

  -- xHat = rmsnorm(x, rms_att) for debugging
  xHatSignal =
    (\idata -> rmsNorm (inputVector idata) (MultiHeadAttention.rmsAtt mha)) <$> intermediateDataSignal

  -- Per-head WO @ head, then sum across heads (equivalent to WO @ concatHeads)
  perHeadProjectedSignalsVec =
    zipWith (\wo hSig -> matrixVectorMult wo <$> hSig) (MultiHeadAttention.mWo mha) perHeadOutputSignalsVec
  perHeadProjectedSignal = sequenceA perHeadProjectedSignalsVec
  woHeadsSignal          = fmap (foldl1 (zipWith (+))) perHeadProjectedSignal

  -- x_after_attn = x + WO@heads
  xAfterAttnSignal = (zipWith (+) . inputVector <$> intermediateDataSignal) <*> woHeadsSignal

  -- Commit attention output on this layer’s attnDone pulse in Stage3_Attend
  nextIntermediateDataSignal =
    liftA4
      (\ps cur attOut done ->
         if processingLayer ps == layerIndex
            && processingStage ps == Stage3_Attend
            && done
           then cur { attentionOutput = attOut }
           else cur)
      processingStateSignal baseNextIntermediateDataSignal xAfterAttnSignal attentionDoneThisLayerSignal

  -- Layer write-done = AND across banks (Stage2_WriteKV)
  writeDoneThisLayerSignal =
    let allBanksDoneSignal = fmap and (sequenceA perBankWriteDoneVec)
    in  (\ps banksDone ->
           processingStage ps == Stage2_WriteKV
        && processingLayer ps == layerIndex
        && banksDone)
        <$> processingStateSignal <*> allBanksDoneSignal

  -- The same gated-commit view, exposed as a tap at Cycle3
  commitCycle3Signal =
    liftA4
      (\ps cur attOut done ->
         if processingLayer ps == layerIndex
            && processingStage ps == Stage3_Attend
            && done
           then cur { attentionOutput = attOut }
           else cur)
      processingStateSignal intermediateDataSignal xAfterAttnSignal attentionDoneThisLayerSignal

  -- Default per-stage work within this layer
  baseNextIntermediateDataSignal =
    liftA2 (processStage mha ffn layerIndex) processingStateSignal intermediateDataSignal

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
        let (qs, ks, vs) = MultiHeadAttention.projectQKV mha ps (inputVector idata) layerIndex
        in  idata { queryVectors = qs, keyVectors = ks, valueVectors = vs }
      -- Stage2: write K,V(pos) to cache (sequenced outside in fillOneBank)
      Stage2_WriteKV    -> idata
      -- Stage3: stream attention (sequenced outside in fillOneBank)
      Stage3_Attend     -> idata
      -- Stage4: FFN
      Stage4_FeedForward ->
        let ffnOut = FeedForwardNetwork.computeFeedForward ffn (attentionOutput idata)
        in  idata { feedForwardOutput = ffnOut }
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

fillOneBank
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
fillOneBank layerIx psSig kvOwner idSig (headOutAcc, headDoneAcc, writeDoneAcc) kvIx =
  let
    stageEquals st =
      liftA2 (\ps _ -> processingStage ps == st && processingLayer ps == layerIx)
             psSig (pure ())

    isStage3Attention = stageEquals Stage3_Attend
    isStage2Write     = stageEquals Stage2_WriteKV

    seqPosSignal = sequencePosition <$> psSig

    bank   = Cache.kvBanks kvOwner !! kvIx
    runKey = Cache.runKeyBank bank
    runVal = Cache.runValueBank bank

    -- Which query heads map to this KV head
    qIdx0 = queryHeadIndex0 kvIx
    hasQ1 = hasSecondQueryHead kvIx
    qIdx1 = queryHeadIndex1 kvIx

    query0 = getQueryVector idSig qIdx0
    query1 = if hasQ1 then getQueryVector idSig qIdx1 else pure (repeat 0)

    keyVec   = getKeyVector   idSig kvIx
    valueVec = getValueVector idSig kvIx

    -- Stage3: stream attention; two readers (one per RAM port)
    -- We pass both the RAM readouts and the local K/V (for optional t==pos bypass inside the head)
    (addrA0, out0, _busy0, done0) =
      AttentionHead.streamHeadAttentionAddrIO
        isStage3Attention seqPosSignal query0 keyVec valueVec keyOutA valOutA

    (addrB1, out1raw, _busy1, done1raw) =
      AttentionHead.streamHeadAttentionAddrIO
        isStage3Attention seqPosSignal query1 keyVec valueVec keyOutB valOutB

    out1  = if hasQ1 then out1raw else pure (repeat 0)
    done1 = if hasQ1 then done1raw else done0
    doneBoth = done0 .&&. done1

    -- Stage2: write K,V(pos) one element per cycle on Port B
    keyValuePairSignal = liftA2 (,) keyVec valueVec
    (writeAddrSig, keyWriteSig, valWriteSig, writeDoneThisBank) =
      Cache.writeSequencer isStage2Write seqPosSignal keyValuePairSignal

    -- Dual-port wiring:
    --   Stage3: Port A <- addrA0 (head0), Port B <- addrB1 (head1), no writes
    --   Stage2: Port B <- write sequencer; Port A parked (read-only, no write)
    addrA = mux isStage3Attention addrA0 (pure 0)          -- park A during write stage
    addrB = mux isStage3Attention addrB1 writeAddrSig

    wrK_A = pure Nothing
    wrV_A = pure Nothing
    wrK_B = mux isStage2Write keyWriteSig (pure Nothing)
    wrV_B = mux isStage2Write valWriteSig (pure Nothing)

    -- IMPORTANT: instantiate each KV bank exactly once; take both port outputs
    (keyOutA, keyOutB) = runKey (addrA, wrK_A) (addrB, wrK_B)
    (valOutA, valOutB) = runVal (addrA, wrV_A) (addrB, wrV_B)

    -- Accumulate outputs and aligned done pulses
    headOutAcc0 = replace qIdx0 out0 headOutAcc
    headDoneAcc0 = replace qIdx0 doneBoth headDoneAcc
    headOutAcc1  = if hasQ1 then replace qIdx1 out1 headOutAcc0 else headOutAcc0
    headDoneAcc1 = if hasQ1 then replace qIdx1 doneBoth headDoneAcc0 else headDoneAcc0

    writeDoneAcc1 = replace kvIx writeDoneThisBank writeDoneAcc

  in (headOutAcc1, headDoneAcc1, writeDoneAcc1)
