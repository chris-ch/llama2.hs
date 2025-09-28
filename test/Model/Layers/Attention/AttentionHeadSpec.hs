module Model.Layers.Attention.AttentionHeadSpec (spec) where
    
import Clash.Prelude
import Model.Core.Types
import Test.Hspec (Spec, describe, it, shouldBe)
import Model.Layers.Attention.AttentionHead (attendHead)

spec :: Spec
spec = do
  describe "attendHead" $ do

    it "returns V at the only position when allowed" $ do
      let snH = SNat @HeadDimension

          q0 :: Vec HeadDimension Float
          q0 = replicate snH 0.0
          q  = replace (0 :: Float) 1.0 q0

          k0 :: Vec HeadDimension Float
          k0 = q
          v0 :: Vec HeadDimension Float
          v0 = replicate snH 2.0

          ks :: Vec SequenceLength (Vec HeadDimension Float)
          ks = repeat k0

          vs :: Vec SequenceLength (Vec HeadDimension Float)
          vs = repeat v0

          pos :: Index SequenceLength
          pos = 0

          out = attendHead q ks vs pos
      out `shouldBe` v0

    it "averages V when both positions have same score and are allowed" $ do
      let snH = SNat @HeadDimension

          q0 = replicate snH 0.0
          q  = replace (0 :: Float) 1.0 q0

          k0 = q
          k1 = q

          v0 = replicate snH 2.0
          v1 = replicate snH 4.0

          tailKs :: Vec (SequenceLength - 2) (Vec HeadDimension Float)
          tailKs = repeat k0

          tailVs :: Vec (SequenceLength - 2) (Vec HeadDimension Float)
          tailVs = repeat v0

          ks :: Vec SequenceLength (Vec HeadDimension Float)
          ks = k0 :> k1 :> tailKs

          vs :: Vec SequenceLength (Vec HeadDimension Float)
          vs = v0 :> v1 :> tailVs

          pos :: Index SequenceLength
          pos = 1

          expected :: Vec HeadDimension Float
          expected = map (/ 2.0) (zipWith (+) v0 v1)

          out = attendHead q ks vs pos
      out `shouldBe` expected
