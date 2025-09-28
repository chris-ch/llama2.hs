module Model.Layers.FeedForward.FeedForwardNetwork (
    FeedForwardNetworkComponent(..), computeFeedForward
) where

import Clash.Prelude
import Model.Layers.FeedForward.FeedForwardNetwork.Internal

import Model.Core.Types (CArray2D, ModelDim, HiddenDim)
import Helpers (rmsNorm, matrixVectorMult)

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
