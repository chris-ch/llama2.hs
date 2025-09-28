module Model.Layers.FeedForward.FeedForwardNetwork (
    FeedForwardNetworkComponent(..), computeFeedForward
) where

import Clash.Prelude

import Model.Core.Types (CArray2D, ModelDim, HiddenDim)
import Helpers (rmsNorm, matrixVectorMult)

data FeedForwardNetworkComponent = FeedForwardNetworkComponent
  { fW1 :: CArray2D HiddenDim ModelDim,
    fW2 :: CArray2D ModelDim HiddenDim,
    fW3 :: CArray2D HiddenDim ModelDim,
    fRMSFfn :: Vec ModelDim Float
  } deriving (Show)

-- Pure feed-forward computation
computeFeedForward :: FeedForwardNetworkComponent
  -> Vec ModelDim Float
  -> Vec ModelDim Float
computeFeedForward ffn inputVector =
  let
    xHat     = rmsNorm inputVector (fRMSFfn ffn)        -- single pre-norm here
    ffnCore  = runFeedForward ffn xHat                  -- no extra norm inside
    out = zipWith (+) inputVector ffnCore
  in out

runFeedForward :: FeedForwardNetworkComponent -> Vec ModelDim Float -> Vec ModelDim Float
runFeedForward ffn xHat =
  let gate = map sigmoidLinearUnit $ matrixVectorMult (fW1 ffn) xHat
      up   = matrixVectorMult (fW3 ffn) xHat
  in matrixVectorMult (fW2 ffn) (zipWith (*) gate up)

-- Activation
sigmoidLinearUnit :: Float -> Float
sigmoidLinearUnit x = x / (1.0 + exp (-x))
