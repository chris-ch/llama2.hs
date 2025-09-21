module Model.Layer
  ( getQ, getK, getV
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

import Model.Types
  ( ProcessingState(..), IntermediateData(..), CycleStage (..)
  )

import qualified Model.Attention as Attention
import qualified Model.Cache as Cache

-- Accessors for IntermediateData signals
getQ :: Signal dom IntermediateData -> Index NumQueryHeads -> Signal dom (Vec HeadDimension Float)
getQ dataSig h = fmap (\idata -> idQueries idata !! h) dataSig

getK :: Signal dom IntermediateData -> Index NumKeyValueHeads -> Signal dom (Vec HeadDimension Float)
getK dataSig kv = fmap (\idata -> idKeys idata !! kv) dataSig

getV :: Signal dom IntermediateData -> Index NumKeyValueHeads -> Signal dom (Vec HeadDimension Float)
getV dataSig kv = fmap (\idata -> idValues idata !! kv) dataSig

headsPerGroupI :: Int
headsPerGroupI = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads

qHi :: Int
qHi = natToNum @NumQueryHeads - 1

-- One layer of the multi-cycle pipeline
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
multiCycleTransformerLayer layer owner layerIdx stateSig dataSig =
  (nextDataSig, writeDoneThisLayerSig, attnDoneThisLayerSig, commitCycle3)
 where
  mha = multiHeadAttention layer
  ffn = feedforwardNetwork layer

  -- For each KV bank: two attention runners share the bank; same bank’s writer runs in Cycle4.
  (headsOutSigsVec, headDoneSigsVec, wrBankDoneVec) =
    let
      initOut  = repeat (pure (repeat 0)) :: Vec NumQueryHeads (Signal dom (Vec HeadDimension Float))
      initDone = repeat (pure False)      :: Vec NumQueryHeads (Signal dom Bool)
      initWr   = repeat (pure False)      :: Vec NumKeyValueHeads (Signal dom Bool)

    in P.foldl (fillOneBank layerIdx stateSig owner dataSig) (initOut, initDone, initWr) indicesI

  -- Attention done for this layer = all heads done (rising edge)
  doneAllHeadsSig     = fmap P.and (sequenceA headDoneSigsVec)
  doneAllPrev         = register False doneAllHeadsSig
  attnDoneThisLayerSig = liftA2 (\n p -> n && not p) doneAllHeadsSig doneAllPrev

  -- Project each head with its W_O slice, then sum across heads
  perHeadProjectedVec = zipWith (\wo h -> matrixVectorMult wo <$> h) (mWo mha) headsOutSigsVec
  perHeadProjectedSig = sequenceA perHeadProjectedVec
  attnSumSig = fmap (foldl1 (zipWith (+))) perHeadProjectedSig

  -- Residual: x + sum_h(W_O @ h)
  attnOutSig = (zipWith (+) P.. idInputVec P.<$> dataSig) <*> attnSumSig

  -- Commit attention output only on this layer’s attn done pulse in Cycle3
  nextDataSig =
    liftA4
      (\st cur attOut done ->
         if psLayer st == layerIdx && psStage st == Cycle3_ComputeAttn && done
           then cur { idAttnOutput = attOut }
           else cur)
      stateSig baseNextDataSig attnOutSig attnDoneThisLayerSig

  -- Layer write done = AND across banks, qualified to this layer/stage
  writeDoneThisLayerSig =
    let allBanksDone = fmap P.and (sequenceA wrBankDoneVec)
    in  (\st d -> psStage st == Cycle4_WriteCache && psLayer st == layerIdx && d)
        <$> stateSig <*> allBanksDone

  -- Same gated “commit” view used by the outer pipeline
  commitCycle3 =
    liftA4
      (\st cur attOut done ->
        if psLayer st == layerIdx && psStage st == Cycle3_ComputeAttn && done
          then cur { idAttnOutput = attOut }
          else cur)
      stateSig dataSig attnOutSig attnDoneThisLayerSig

  -- Default per-stage work within this layer
  baseNextDataSig = liftA2 (processCycle mha ffn layerIdx) stateSig dataSig

fillOneBank :: HiddenClockResetEnable dom
  => Index NumLayers
  -> Signal dom ProcessingState
  -> Cache.KVRamOwner dom
  -> Signal dom IntermediateData
  -> (Vec NumQueryHeads (Signal dom (Vec HeadDimension Float)), Vec NumQueryHeads (Signal dom Bool), Vec NumKeyValueHeads (Signal dom Bool))
  -> Index NumKeyValueHeads
  -> (Vec NumQueryHeads (Signal dom (Vec HeadDimension Float)), Vec NumQueryHeads (Signal dom Bool), Vec NumKeyValueHeads (Signal dom Bool))
fillOneBank layerIdx stateSig owner dataSig (outAcc, doneAcc, wrAcc) kv =
  let
    -- Stage qualifiers
    stgEq s = liftA2 (\st _ -> psStage st == s && psLayer st == layerIdx) stateSig (pure ())
    isC3    = stgEq Cycle3_ComputeAttn
    isC4    = stgEq Cycle4_WriteCache
    attStart = isC3
    posSig   = psSeqPos <$> stateSig

    bank  = Cache.kvBanks owner !! kv
    kRun  = Cache.runK bank
    vRun  = Cache.runV bank

    base  = fromEnum kv * headsPerGroupI
    q0i   = toEnum (min qHi base)         :: Index NumQueryHeads
    hasQ1 = base + 1 <= qHi
    q1i   = if hasQ1 then toEnum (base + 1) else q0i

    q0S  = getQ dataSig q0i
    kCur = getK dataSig kv
    vCur = getV dataSig kv
    q1S  = if hasQ1 then getQ dataSig q1i else pure (repeat 0)

    (addr0, out0, _busy0, done0) = Attention.streamHeadAttentionAddrIO attStart posSig q0S kCur vCur kQ0 vQ0
    (addr1, out1, _busy1, done1) = Attention.streamHeadAttentionAddrIO attStart posSig q1S kCur vCur kQ1 vQ1

    -- Single writer instance for this bank
    kvPairSig                       = liftA2 (,) kCur vCur
    (wrAddr, kWr, vWr, wrDoneBank)  = Cache.writeSequencer isC4 posSig kvPairSig

    -- Port muxing: in C3 both ports read; in C4, port B writes
    addrA = addr0
    wrA   = pure Nothing
    addrB = mux isC3 addr1 wrAddr
    wrKB  = mux isC3 (pure Nothing) kWr
    wrVB  = mux isC3 (pure Nothing) vWr

    (kQA, kQB) = kRun (addrA, wrA) (addrB, wrKB)
    (vQA, vQB) = vRun (addrA, wrA) (addrB, wrVB)

    kQ0 = kQA; vQ0 = vQA
    kQ1 = kQB; vQ1 = vQB

    outAcc0  = replace q0i out0 outAcc
    doneAcc0 = replace q0i done0 doneAcc
    outAcc1  = if hasQ1 then replace q1i out1 outAcc0 else outAcc0
    doneAcc1 = if hasQ1 then replace q1i done1 doneAcc0 else doneAcc0
    wrAcc1   = replace kv wrDoneBank wrAcc
  in (outAcc1, doneAcc1, wrAcc1)

processCycle :: MultiHeadAttentionComponent -> FeedForwardNetworkComponent -> Index NumLayers -> ProcessingState -> IntermediateData ->  IntermediateData
processCycle mha ffn layerIdx st idata
  | psLayer st /= layerIdx = idata
  | otherwise =
      case psStage st of
        Cycle1_ReadCache -> idata
        Cycle2_ComputeQKV ->
          let
            xHat = rmsNorm (idInputVec idata) (rmsAtt mha)
            -- Queries: one per Q head (with RoPE on Q)
            queries =
              imap (\hIdx _ ->
                let headComp = heads mha !! hIdx
                    (q, _, _)   = runSingleHeadQKV headComp xHat
                    (qRot, _kU) = applyRotaryToHead headComp
                                    (StepCount $ fromIntegral $ psSeqPos st)
                                    (q, repeat 0)
                in qRot) indicesI

            -- Keys/Values: one per KV head (apply RoPE to K only)
            keysAndValues =
              imap (\kvIdx _ ->
                let qIdx0 = fromEnum kvIdx * (natToNum @NumQueryHeads `P.div` natToNum @NumKeyValueHeads)
                    qIdx  = toEnum (min (natToNum @NumQueryHeads - 1) qIdx0) :: Index NumQueryHeads
                    headComp     = heads mha !! qIdx
                    (_q, k, v)   = runSingleHeadQKV headComp xHat
                    (_qU, kRot)  = applyRotaryToHead headComp
                                      (StepCount $ fromIntegral $ psSeqPos st)
                                      (repeat 0, k)
                in (kRot, v)) indicesI
            (keys, values) = unzip keysAndValues
          in idata { idQueries = queries, idKeys = keys, idValues = values }
        Cycle3_ComputeAttn -> idata
        Cycle4_WriteCache  -> idata
        Cycle5_ComputeFFN  ->
          let ffnOut = computeFeedForward ffn (idAttnOutput idata)
          in idata { idFFNOutput = ffnOut }
