module Model.Layers.Attention.MultiHeadAttention (
    MultiHeadAttentionComponent(..), projectQKV, StepCount(..)
) where

import Clash.Prelude

import Model.Core.Types (NumQueryHeads, ModelDim, NumKeyValueHeads,
  HeadDimension, CArray2D (..), SingleHeadComponent (..),
  FreqDim, RotaryEncodingComponent (..))
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
getRow :: forall n m. (KnownNat n) => StepCount -> CArray2D n m -> Vec m Float
getRow (StepCount i) (CArray2D arr) = arr !! (fromIntegral i :: Index n)

-- Apply rotation per head
applyRotation :: RotaryEncodingComponent
  -> StepCount
  -> Vec HeadDimension Float
  -> Vec HeadDimension Float
applyRotation rot step tokenVec =
  let cosFrequencies = getRow step (freqCos rot)
      sinFrequencies = getRow step (freqSin rot)
  in applyRotaryPositionEncoding tokenVec cosFrequencies sinFrequencies

-- Compute K/V for a head
computeHeadKV
  :: SingleHeadComponent
  -> StepCount
  -> Vec ModelDim Float
  -> (Vec HeadDimension Float, Vec HeadDimension Float)
computeHeadKV headComp step xHat =
  let
    k = matrixVectorMult (wkHead headComp) xHat  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    v = matrixVectorMult (wvHead headComp) xHat  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    kRot = applyRotation (rotary headComp) step k
    CArray2D _wQ = wqHead headComp
    CArray2D _wK = wkHead headComp
    CArray2D _wV = wvHead headComp
  in (kRot, v)

-- Compute Q for a head
computeHeadQ
  :: SingleHeadComponent
  -> StepCount
  -> Vec ModelDim Float
  -> Vec HeadDimension Float
computeHeadQ headComp step xHat =
  let
    q = matrixVectorMult (wqHead headComp) xHat  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    qRot = applyRotation (rotary headComp) step q

    CArray2D _wK = wkHead headComp
  in qRot

projectQKV :: MultiHeadAttentionComponent
  -> StepCount
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
