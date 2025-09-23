module Model.Layers.FeedForward.FeedForwardNetwork (
    FeedForwardNetworkComponent(..), computeFeedForward
) where

import Clash.Prelude

import Helpers (CArray2D, ModelDim, HiddenDim, rmsNorm, matrixVectorMult)

data FeedForwardNetworkComponent = FeedForwardNetworkComponent
  { fW1 :: CArray2D HiddenDim ModelDim,
    fW2 :: CArray2D ModelDim HiddenDim,
    fW3 :: CArray2D HiddenDim ModelDim,
    fRMSFfn :: Vec ModelDim Float
  } deriving (Show)

-- Pure feed-forward computation
computeFeedForward
  :: FeedForwardNetworkComponent
  -> Vec ModelDim Float
  -> Vec ModelDim Float
computeFeedForward ffn x =
  let xHat     = rmsNorm x (fRMSFfn ffn)                  -- single pre-norm here
      ffnCore  = runFeedForward ffn xHat                  -- no extra norm inside
  in zipWith (+) x ffnCore

runFeedForward :: FeedForwardNetworkComponent -> Vec ModelDim Float -> Vec ModelDim Float
runFeedForward ffn xHat =
  let
    w1 = fW1 ffn
    w2 = fW2 ffn
    w3 = fW3 ffn
    gatePre   = matrixVectorMult w1 xHat
    upPre     = matrixVectorMult w3 xHat
    gate      = map sigmoidLinearUnit gatePre
  in
    matrixVectorMult w2 (zipWith (*) gate upPre)

-- Activation
sigmoidLinearUnit :: Float -> Float
sigmoidLinearUnit x = x / (1.0 + exp (-x))
