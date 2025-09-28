module Model.Layers.Attention.AttentionHead
  ( attendHead ) where

import Clash.Prelude
import Model.Core.Types (HeadDimension, SequenceLength)

attendHead
  :: Vec HeadDimension Float                -- Q (this head)
  -> Vec SequenceLength (Vec HeadDimension Float)   -- all K rows for this head
  -> Vec SequenceLength (Vec HeadDimension Float)   -- all V rows for this head
  -> Index SequenceLength                           -- pos (inclusive)
  -> Vec HeadDimension Float                -- attended output
attendHead q ks vs pos =
  let
    scale :: Float
    scale = 1.0 / sqrt (natToNum @HeadDimension :: Float)

    negBig :: Float
    negBig = -1.0e30

    -- scores for all t, masked to -inf beyond pos
    scores :: Vec SequenceLength Float
    scores =
      imap (\t krow ->
              let s = sum (zipWith (*) q krow) * scale
              in if fromEnum t <= fromEnum pos then s else negBig)
           ks

    m   = maximum scores
    exps = map (\s -> if s == negBig then 0 else exp (s - m)) scores
    d   = fold (+) exps

    probs = if d == 0 then repeat 0 else map (/ d) exps

    -- weighted sum over V rows
    out  = foldl
             (\acc (p, vrow) -> zipWith (+) acc (map (* p) vrow))
             (repeat 0)
             (zip probs vs)
  in out
