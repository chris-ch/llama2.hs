module Model.Layers.Attention.MultiHeadAttentionSpec (spec) where

import Test.Hspec
import Clash.Prelude

import Model.Layers.Attention.MultiHeadAttention.Internal (applyRotaryPositionEncoding)
import Model.Core.Types ( HeadDimension, FreqDim )

spec :: Spec
spec = do
  describe "applyRotaryPositionEncoding" $
    it "is identity when cos=1 and sin=0" $ do
      -- Create a Vec HeadDimension Float as input
      let input :: Vec HeadDimension Float
          input = imap (\i _ -> fromIntegral (i + 1)) (repeat (0 :: Int)) 
          -- gives [1,2,3,...,HeadDimension] basically

          cosV :: Vec FreqDim Float
          cosV = repeat 1.0

          sinV :: Vec FreqDim Float
          sinV = repeat 0.0

      applyRotaryPositionEncoding input cosV sinV `shouldBe` input
