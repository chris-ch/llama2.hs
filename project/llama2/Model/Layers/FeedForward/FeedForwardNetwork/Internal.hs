module Model.Layers.FeedForward.FeedForwardNetwork.Internal  where

import Clash.Prelude

import Model.Core.Types (CArray2D, ModelDim, HiddenDim)
import Helpers (rmsNorm, matrixVectorMult)

data FeedForwardNetworkComponent = FeedForwardNetworkComponent
  { fW1 :: CArray2D HiddenDim ModelDim,
    fW2 :: CArray2D ModelDim HiddenDim,
    fW3 :: CArray2D HiddenDim ModelDim,
    fRMSFfn :: Vec ModelDim Float
  } deriving (Show)

runFeedForward :: FeedForwardNetworkComponent -> Vec ModelDim Float -> Vec ModelDim Float
runFeedForward ffn xHat =
  let gate = map sigmoidLinearUnit $ matrixVectorMult (fW1 ffn) xHat
      up   = matrixVectorMult (fW3 ffn) xHat
  in matrixVectorMult (fW2 ffn) (zipWith (*) gate up)

-- Activation
sigmoidLinearUnit :: Float -> Float
sigmoidLinearUnit x = x / (1.0 + exp (-x))
