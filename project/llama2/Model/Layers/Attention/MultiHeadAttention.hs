module Model.Layers.Attention.MultiHeadAttention (
    MultiHeadAttentionComponent(..), projectQKV, StepCount(..)
) where

import Clash.Prelude

import Model.Core.Types (NumQueryHeads, ModelDim, NumKeyValueHeads,
  HeadDimension, CArray2D (..), SingleHeadComponent (..),
  FreqDim, RotaryEncodingComponent (..), SequenceLength)
import Helpers (matrixVectorMult, rmsNorm)
import qualified Prelude as P

import Model.Layers.Attention.MultiHeadAttention.Internal
    ( MultiHeadAttentionComponent(..),
      StepCount(..),
      computeHeadKV,
      computeHeadQ )

projectQKV :: MultiHeadAttentionComponent
  -> Index SequenceLength
  -> Vec ModelDim Float
  -> (Vec NumQueryHeads (Vec HeadDimension Float), 
  Vec NumKeyValueHeads (Vec HeadDimension Float), 
  Vec NumKeyValueHeads (Vec HeadDimension Float))
projectQKV multiHeadAttentionComponent stepCount inputVector = 
  let
    normalizedInput = rmsNorm inputVector (rmsAtt multiHeadAttentionComponent)
    -- Queries: one per Q head (with RoPE on Q)
    queries =
      imap (\queryHeadIdx _ ->
        let
          headComponent = heads multiHeadAttentionComponent !! queryHeadIdx
          queryRotated = computeHeadQ headComponent stepCount normalizedInput
        in queryRotated) indicesI

    -- Keys/Values: one per KV head (apply RoPE to K only)
    keysAndValues =
      imap (\keyValueHeadIdx _ ->
        let
          qIdx0 = fromEnum keyValueHeadIdx * (natToNum @NumQueryHeads `P.div` natToNum @NumKeyValueHeads)
          queryIndex = toEnum (min (natToNum @NumQueryHeads - 1) qIdx0) :: Index NumQueryHeads
          headComponent = heads multiHeadAttentionComponent !! queryIndex
          (keyRotated, valueRotated) = computeHeadKV headComponent stepCount normalizedInput
        in (keyRotated, valueRotated)) indicesI
    (keys, values) = unzip keysAndValues
  in (queries, keys, values)
