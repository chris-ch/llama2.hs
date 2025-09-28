module Model.Layers.Attention.AttentionHeadComb
  ( attendHeadComb ) where

import Clash.Prelude
import Model.Core.Types (HeadDimension, SeqLen)

attendHeadComb
  :: Vec HeadDimension Float                -- Q (this head)
  -> Vec SeqLen (Vec HeadDimension Float)   -- all K rows for this head
  -> Vec SeqLen (Vec HeadDimension Float)   -- all V rows for this head
  -> Index SeqLen                           -- pos (inclusive)
  -> Vec HeadDimension Float                -- attended output
attendHeadComb q ks vs pos =
  let
    scale :: Float
    scale = 1.0 / sqrt (natToNum @HeadDimension :: Float)

    negBig :: Float
    negBig = -1.0e30

    -- scores for all t, masked to -inf beyond pos
    scores :: Vec SeqLen Float
    scores =
      imap (\t krow ->
              let s = sum (zipWith (*) q krow) * scale
              in if fromIntegral (fromEnum t) <= fromIntegral (fromEnum pos) then s else negBig)
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
