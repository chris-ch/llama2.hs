{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}

module DebugDump (dumpLayerSums) where

import Prelude

import qualified Clash.Prelude          as C
import           Numeric                (showFFloat)
import qualified Data.Foldable          as F

import Helpers
  ( TransformerDecoderComponent(..)
  , TransformerLayerComponent(..)
  , MultiHeadAttentionComponent(..)
  , FeedForwardNetworkComponent(..)
  , CArray2D(..), NumLayers
  )

-- Sum of absolute values, in Double for stability/printing
sumAbs :: [Float] -> Double
sumAbs = sum . fmap (realToFrac . abs)

-- Flatten a 2D Clash matrix into a Haskell list
toList2 :: CArray2D n m -> [Float]
toList2 (CArray2D mat) = concatMap C.toList (C.toList mat)

-- Dump per-layer absolute-sum checksums for all major tensors.
-- Output format mirrors the C helper so you can diff them directly.
dumpLayerSums :: TransformerDecoderComponent -> IO ()
dumpLayerSums TransformerDecoderComponent{..} = do
  putStrLn "=== Haskell per-layer checksum sums (abs) ==="
  -- Build one IO action per layer using C.imap, then run them with sequenceA_
  let actions :: C.Vec NumLayers (IO ())
      actions = C.imap dumpOne modelLayers
  F.sequenceA_ actions
  putStrLn "=== End checksums ==="
 where
  pr :: String -> Double -> IO ()
  pr tag v = putStrLn (tag ++ showFFloat (Just 6) v "")

  dumpOne :: C.Index n -> TransformerLayerComponent -> IO ()
  dumpOne lidx TransformerLayerComponent{..} = do
    let 
        MultiHeadAttentionComponent{..} = multiHeadAttention
        FeedForwardNetworkComponent{..} = feedforwardNetwork

        chk :: String -> [Float] -> IO ()
        chk name xs =
          pr ("[CHK] L=" ++ show lidx ++ " " ++ name ++ "  sum|x|=") (sumAbs xs)

    -- Attention-side
    chk "rms_att" (C.toList rmsAtt)
    -- Feed-forward-side
    chk "rms_ffn" (C.toList fRMSFfn)
    chk "W1"      (toList2 fW1)
    chk "W2"      (toList2 fW2)
    chk "W3"      (toList2 fW3)
    -- W_O is stored per head as (ModelDim x HeadDim); flatten across heads
    chk "WO"      (concatMap toList2 (C.toList mWo))
