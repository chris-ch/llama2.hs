module Model.Top
  ( multiCycleTransformer
  , topEntity
  ) where

import Clash.Prelude

import Helpers
  ( TransformerDecoderComponent(..), EmbeddingComponent(..)
  , embed, transformerLogits, argMax, softmax, sampleFromProbs
  , Temperature, Seed, NumLayers, TransformerLayerComponent, liftA4, xorshift32
  )

import Model.Types
  ( ProcessingState(..), initialProcessingState, nextState
  , IntermediateData(..), initialIntermediateData, CycleStage (..)
  )

import Model.Cache
  ( KVRamOwner(..), mkKVRamOwner )

import Model.Layer (multiCycleTransformerLayer)

-- Full transformer (multi-cycle, advance on stageDone)
multiCycleTransformer
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> Vec NumLayers (KVRamOwner dom)
  -> Signal dom (Unsigned 32)  -- Token
  -> Signal dom Temperature
  -> Signal dom Seed
  -> (Signal dom (Unsigned 32), Signal dom Bool)
multiCycleTransformer decoder caches tokenSig temperatureSig seedSig =
  (outputTokenSig, readyPulseSig)
 where
  embedding    = modelEmbedding decoder
  layers       = modelLayers decoder
  inputVecSig  = embed (vocabulary embedding) <$> tokenSig

  -- Global controller: advance only when current stageDone is asserted
  advance st done = if done then nextState st else st
  procStateSig = register initialProcessingState (advance <$> procStateSig <*> stageDoneSig)

  -- Intermediate data register
  currentDataSig = register initialIntermediateData nextDataSig

  -- Load the layer input at Cycle1:
  -- - layer 0 gets the token embedding
  -- - layers >0 get the previous layer's FFN output
  inputLoadedSig =
    liftA3
      (\st cur inp ->
         if psStage st == Cycle1_ReadCache
           then if psLayer st == 0
                  then cur { idInputVec = inp }         -- layer 0: embedding
                  else cur { idInputVec = idFFNOutput cur } -- higher layers: previous FFN
           else cur)
      procStateSig currentDataSig inputVecSig

  -- Run layers; gather per-stage done pulses
  foldStep
    :: ( Signal dom IntermediateData
      , Vec NumLayers (Signal dom Bool)
      , Vec NumLayers (Signal dom Bool))
    -> (TransformerLayerComponent, KVRamOwner dom, Index NumLayers)
    -> ( Signal dom IntermediateData
      , Vec NumLayers (Signal dom Bool)
      , Vec NumLayers (Signal dom Bool))
  foldStep (dSig, wD, aD) (layer, cache, lidx) =
    let (newD, wDoneL, aDoneL, commitC3) =
          multiCycleTransformerLayer layer cache lidx procStateSig dSig

        dSel = liftA4
          (\st old new commit ->
             if psLayer st == lidx
               then if psStage st == Cycle3_ComputeAttn
                      then commit   -- gated write-back on attnDone
                      else new
               else old)
          procStateSig dSig newD commitC3
    in ( dSel
       , replace lidx wDoneL wD
       , replace lidx aDoneL aD)

  (nextDataSig, writeDoneVec, attnDoneVec) =
    foldl foldStep (inputLoadedSig, repeat (pure False), repeat (pure False))
                   (zip3 layers caches indicesI)

  writeDoneAny = fmap or (sequenceA writeDoneVec)
  attnDoneAny  = fmap or (sequenceA attnDoneVec)

  -- Stage done selection
  stgSig = psStage <$> procStateSig
  is c   = (== c) <$> stgSig

  stageDoneSig =
    mux (is Cycle1_ReadCache)   (pure True)  $
    mux (is Cycle2_ComputeQKV)  (pure True)  $
    mux (is Cycle3_ComputeAttn) attnDoneAny  $
    mux (is Cycle4_WriteCache)  writeDoneAny $
    mux (is Cycle5_ComputeFFN)  (pure True)  $
    pure False

  -- Ready pulse at last layer FFN completion (rising edge)
  isLastFFN  = liftA2 (\st _ -> psStage st == Cycle5_ComputeFFN && psLayer st == maxBound)
                      procStateSig (pure ())
  readyPulseSig = liftA2 (\now prev -> now && not prev) isLastFFN (register False isLastFFN)

  -- logits from the current (next) data
  logitsNow = transformerLogits decoder . idFFNOutput <$> nextDataSig

  -- PRNG state: seed is mixed in on the first pulse; otherwise advance each pulse
  firstPulseSig :: Signal dom Bool
  firstPulseSig = regEn True readyPulseSig (pure False)

  mixedSeedSig :: Signal dom (Unsigned 32)
  mixedSeedSig = (`xor` 0x9E3779B9) <$> seedSig

  prngStateSig :: Signal dom (Unsigned 32)
  prngStateSig =
    let nextVal = mux firstPulseSig (xorshift32 <$> mixedSeedSig)
                                    (xorshift32 <$> prngStateSig)
    in regEn 2463534242 readyPulseSig nextVal  -- non-zero default

  -- Convert to Float in [0,1). Use top 24 bits as mantissa fraction.
  uniform01Sig :: Signal dom Float
  uniform01Sig = (/ 16777216.0) . fromIntegral . (`shiftR` 8) <$> prngStateSig

  -- Sampling on pulse:
  sampledTokenOnPulse :: Signal dom (Unsigned 32)
  sampledTokenOnPulse =
    liftA3
      (\temperature logs u ->
         if temperature <= 0.0 then argMax logs
         else let probs = softmax temperature logs in sampleFromProbs u probs)
      temperatureSig logitsNow uniform01Sig

  outputTokenSig = regEn 0 readyPulseSig sampledTokenOnPulse

-- ============================================================================
-- Top Entity
-- ============================================================================

topEntity
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> Signal dom (Unsigned 32)  -- Token
  -> Signal dom Temperature
  -> Signal dom Seed
  -> (Signal dom (Unsigned 32), Signal dom Bool)
topEntity decoder = multiCycleTransformer decoder (repeat mkKVRamOwner)
