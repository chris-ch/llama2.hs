{-# LANGUAGE TypeApplications #-}
module Model.Layers.TransformerLayerSpec (spec) where

import Test.Hspec ( Spec, describe, it, shouldBe )
import Clash.Prelude
import Clash.Explicit.Testbench
import Model.Layers.TransformerLayer
    ( TransformerLayerComponent(TransformerLayerComponent,
                                feedforwardNetwork, multiHeadAttention),
      multiCycleTransformerLayer )
import Model.Core.Types
import qualified Model.Layers.FeedForward.FeedForwardNetwork as FF
import qualified Model.Layers.Attention.MultiHeadAttention.Internal as MHA
import Model.Memory.KVCacheBank ( makeRamOwnerKV )
import Model.Core.Transformer.Internal ( initialIntermediateData )

spec :: Spec
spec = do
    describe "TransformerLayer multiCycleTransformerLayer" $ do
        it "should produce valid intermediate data and stage pulses" $ do
            -- Create a trivial TransformerLayerComponent
            let mhaComp = MHA.MultiHeadAttentionComponent
                            { MHA.heads = repeat dummyHead  -- Add @h here if error persists, e.g., repeat @1 dummyHead
                            , MHA.mWo = repeat (CArray2D (repeat (repeat 0)))  -- Add @h @dHead @dModel, e.g., repeat @1 (CArray2D (repeat @2 (repeat @4 0)))
                            , MHA.rmsAtt = repeat 1.0  -- Add @l if per-layer, e.g., repeat @1 1.0
                            }
                ffnComp = FF.FeedForwardNetworkComponent
                            { FF.fW1 = CArray2D (repeat (repeat 0))  -- Add @dModel @dFF, e.g., CArray2D (repeat @4 (repeat @8 0))
                            , FF.fW2 = CArray2D (repeat (repeat 0))  -- Similarly
                            , FF.fW3 = CArray2D (repeat (repeat 0))  -- Similarly
                            , FF.fRMSFfn = repeat 1.0  -- Add @l if needed, e.g., repeat @1 1.0
                            }
                transformerLayer = TransformerLayerComponent
                                    { multiHeadAttention = mhaComp
                                    , feedforwardNetwork = ffnComp
                                    }

                clk = tbSystemClockGen (not <$> done)
                rst = systemResetGen
                enable = enableGen

                done = pure False

            -- Apply multiCycleTransformerLayer with required signals
            let (idataOut, writeDone, attnDone, _commitC3, tapPulse, _dbgXHat) =
                    withClockResetEnable @System clk rst enable $
                        multiCycleTransformerLayer
                            transformerLayer
                            makeRamOwnerKV
                            0
                            (pure $ ProcessingState { processingStage = Stage1_ProjectQKV, processingLayer = 0, sequencePosition = 0 })
                            (pure initialIntermediateData)

            -- Check basic properties
            sampleN 1 idataOut `shouldBe` [initialIntermediateData]
            sampleN 1 writeDone `shouldBe` [False]
            sampleN 1 attnDone `shouldBe` [False]
            sampleN 1 tapPulse `shouldBe` [False]

-- Dummy SingleHeadComponent
dummyHead :: SingleHeadComponent
dummyHead = SingleHeadComponent
  { wqHead = CArray2D (repeat (repeat 0))  -- Add @dModel @dHead, e.g., CArray2D (repeat @4 (repeat @2 0))
  , wkHead = CArray2D (repeat (repeat 0))  -- Similarly
  , wvHead = CArray2D (repeat (repeat 0))  -- Similarly
  , rotary = RotaryEncodingComponent
                  { freqCos = CArray2D (repeat (repeat 1))  -- Similarly, e.g., CArray2D (repeat @4 (repeat @2 1))
                  , freqSin = CArray2D (repeat (repeat 0))  -- Similarly
                  }
  }