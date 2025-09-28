module Spec where

import Clash.Prelude

import Test.Hspec
import qualified Model.Layers.Attention.AttentionHeadSpec (spec)
import qualified Model.Layers.Attention.MultiHeadAttentionSpec (spec)
import qualified Model.Layers.FeedForward.FeedForwardNetworkSpec  (spec)
import qualified Model.Layers.TransformerLayerSpec (spec)

main :: IO ()
main = hspec $ do
  Model.Layers.Attention.AttentionHeadSpec.spec
  Model.Layers.Attention.MultiHeadAttentionSpec.spec
  Model.Layers.FeedForward.FeedForwardNetworkSpec.spec
  Model.Layers.TransformerLayerSpec.spec
