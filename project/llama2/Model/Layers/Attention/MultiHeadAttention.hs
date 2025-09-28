module Model.Layers.Attention.MultiHeadAttention (
    MultiHeadAttentionComponent(..), projectQKV, StepCount(..)
) where

import Clash.Prelude

import Model.Core.Types (NumQueryHeads, ModelDim, NumKeyValueHeads,
  HeadDimension, CArray2D (..), SingleHeadComponent (..),
  FreqDim, RotaryEncodingComponent (..), SeqLen)
import Helpers (matrixVectorMult, rmsNorm)
import qualified Prelude as P

newtype StepCount = StepCount (Unsigned 32) deriving (Show, Eq, Ord)

data MultiHeadAttentionComponent = MultiHeadAttentionComponent
  { heads  :: Vec NumQueryHeads SingleHeadComponent
  -- | Per-head output projection matrix W_O (shape HeadDim × ModelDim)
  , mWo :: Vec NumQueryHeads (CArray2D ModelDim HeadDimension)
  -- | RMSNorm before QKV projection (size ModelDim)
  , rmsAtt :: Vec ModelDim Float
  } deriving (Show)

applyRotaryPositionEncoding :: Vec HeadDimension Float    -- input vector
  -> Vec FreqDim Float  -- cosFrequencies
  -> Vec FreqDim Float  -- sinFrequencies
  -> Vec HeadDimension Float
applyRotaryPositionEncoding inputVec cosVec sinVec =
  imap rotatePair inputVec
  where
    rotatePair :: KnownNat headDim => Index headDim -> Float -> Float
    rotatePair i _
        | even idx = rotatedReal
        | otherwise = rotatedImag
        where
            idx :: Int
            idx = fromIntegral i
            pairIdx = idx `div` 2
            realComponent = inputVec !! (2 * pairIdx)
            imagComponent = inputVec !! (2 * pairIdx + 1)
            cosValue = cosVec !! pairIdx
            sinValue = sinVec !! pairIdx
            rotatedReal = realComponent * cosValue - imagComponent * sinValue
            rotatedImag = realComponent * sinValue + imagComponent * cosValue

-- Index into CArray2D to get a row
getRow :: forall n m. (KnownNat n) => Index SeqLen -> CArray2D n m -> Vec m Float
getRow index (CArray2D arr) = arr !! (fromIntegral index :: Index n)

-- Apply rotation per head
applyRotation :: RotaryEncodingComponent
  -> Index SeqLen
  -> Vec HeadDimension Float
  -> Vec HeadDimension Float
applyRotation rot stepCount tokenVec =
  let cosFrequencies = getRow stepCount (freqCos rot)
      sinFrequencies = getRow stepCount (freqSin rot)
  in applyRotaryPositionEncoding tokenVec cosFrequencies sinFrequencies

-- Compute K/V for a head
computeHeadKV
  :: SingleHeadComponent
  -> Index SeqLen
  -> Vec ModelDim Float
  -> (Vec HeadDimension Float, Vec HeadDimension Float)
computeHeadKV headComp stepCount xHat =
  let
    k = matrixVectorMult (wkHead headComp) xHat  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    v = matrixVectorMult (wvHead headComp) xHat  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    kRot = applyRotation (rotary headComp) stepCount k
    CArray2D _wQ = wqHead headComp
    CArray2D _wK = wkHead headComp
    CArray2D _wV = wvHead headComp
  in (kRot, v)

-- Compute Q for a head
computeHeadQ
  :: SingleHeadComponent
  -> Index SeqLen
  -> Vec ModelDim Float
  -> Vec HeadDimension Float
computeHeadQ headComp stepCount xHat =
  let
    q = matrixVectorMult (wqHead headComp) xHat  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    qRot = applyRotation (rotary headComp) stepCount q

    CArray2D _wK = wkHead headComp
  in qRot

projectQKV :: MultiHeadAttentionComponent
  -> Index SeqLen
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
