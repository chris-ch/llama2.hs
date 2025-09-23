module Model.Layers.Attention.MultiHeadAttention (
    MultiHeadAttentionComponent(..), computeMultiHeadAttention
) where

import Clash.Prelude

import Helpers (NumQueryHeads, ModelDim, NumKeyValueHeads, SeqLen, HeadDimension, CArray2D, SingleHeadComponent, matrixVectorMult, dotVec)

data MultiHeadAttentionComponent = MultiHeadAttentionComponent
  { heads  :: Vec NumQueryHeads SingleHeadComponent
  -- | Per-head output projection matrix W_O (shape HeadDim × ModelDim)
  , mWo :: Vec NumQueryHeads (CArray2D ModelDim HeadDimension)
  -- | RMSNorm before QKV projection (size ModelDim)
  , rmsAtt :: Vec ModelDim Float
  } deriving (Show)

-- Pure multi-head attention computation
computeMultiHeadAttention
  :: MultiHeadAttentionComponent
  -> Vec ModelDim Float
  -> Vec NumQueryHeads (Vec HeadDimension Float)
  -> Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float))
  -> Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float))
  -> Index SeqLen
  -> Vec ModelDim Float
computeMultiHeadAttention mha x qs kHeads vHeads pos =
  let
    headsPerGroup :: Int
    headsPerGroup = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads

    -- per-head attention outputs in HeadDimension
    headOutputs :: Vec NumQueryHeads (Vec HeadDimension Float)
    headOutputs = imap
      (\qIdx qVec ->
         let kvIdx :: Index NumKeyValueHeads
             kvIdx =
               let qi  = fromEnum qIdx
                   idx = qi `div` headsPerGroup
                   hi  = natToNum @NumKeyValueHeads - 1
               in toEnum (max 0 (min hi idx))

             ks = kHeads !! kvIdx
             vs = vHeads !! kvIdx

             scores  = computeAttentionScores qVec ks
             -- causal mask: t > pos => subtract large number
             weights = computeAttentionWeights
                         (imap (\t s -> if fromEnum t <= fromIntegral pos
                                          then s else s - 1.0e9) scores)
         in computeAttentionOutput weights vs)
      qs

    -- apply W_O per query head, then sum across heads to ModelDim
    perHeadProjected :: Vec NumQueryHeads (Vec ModelDim Float)
    perHeadProjected = zipWith matrixVectorMult (mWo mha) headOutputs

    woAttn :: Vec ModelDim Float
    woAttn = foldl1 (zipWith (+)) perHeadProjected
  in
    zipWith (+) x woAttn

-- Pure attention computation
computeAttentionScores
  :: Vec HeadDimension Float        -- query
  -> Vec SeqLen (Vec HeadDimension Float)  -- keys
  -> Vec SeqLen Float
computeAttentionScores query keys =
  let headDim = snatToNum (SNat @HeadDimension)
      scaling = sqrt (headDim :: Float)
  in map (\key -> dotVec query key / scaling) keys

computeAttentionWeights :: forall n. KnownNat (n+1) => Vec (n+1) Float -> Vec (n+1) Float
computeAttentionWeights xs =
  let m = maximum xs
      exps = map (\x -> exp (x - m)) xs
      s = sum exps
  in map (/ s) exps

-- Pure attention output computation
computeAttentionOutput
  :: Vec SeqLen Float                    -- attention weights
  -> Vec SeqLen (Vec HeadDimension Float)    -- values
  -> Vec HeadDimension Float
computeAttentionOutput weights values =
  let zeroVec = repeat 0 :: Vec HeadDimension Float
      weightedAdd acc (w, vrow) = zipWith (+) acc (map (* w) vrow)
      pairs = zip weights values
  in foldl weightedAdd zeroVec pairs
