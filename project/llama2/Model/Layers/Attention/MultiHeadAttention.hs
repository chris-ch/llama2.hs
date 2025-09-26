module Model.Layers.Attention.MultiHeadAttention (
    MultiHeadAttentionComponent(..), projectQKV, StepCount(..)
) where

import Clash.Prelude

import Model.Core.Types (NumQueryHeads, ModelDim, NumKeyValueHeads, 
  HeadDimension, CArray2D (..), SingleHeadComponent (..),
  FreqDim, RotaryEncodingComponent (..), ProcessingState (..))
import Helpers (matrixVectorMult, rmsNorm)
import qualified Prelude as P
import Debug.Trace (trace)

newtype StepCount = StepCount (Unsigned 32) deriving (Show, Eq, Ord)

data MultiHeadAttentionComponent = MultiHeadAttentionComponent
  { heads  :: Vec NumQueryHeads SingleHeadComponent
  -- | Per-head output projection matrix W_O (shape HeadDim × ModelDim)
  , mWo :: Vec NumQueryHeads (CArray2D ModelDim HeadDimension)
  -- | RMSNorm before QKV projection (size ModelDim)
  , rmsAtt :: Vec ModelDim Float
  } deriving (Show)

-- QKV per head - should return HeadDimension vectors, not ModelDim
runSingleHeadQKV :: SingleHeadComponent -> Vec ModelDim Float -> (Vec HeadDimension Float, Vec HeadDimension Float, Vec HeadDimension Float)
runSingleHeadQKV headComp normalizedInput = (q, k, v) where
    q = matrixVectorMult (wqHead headComp) normalizedInput  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    k = matrixVectorMult (wkHead headComp) normalizedInput  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    v = matrixVectorMult (wvHead headComp) normalizedInput  -- HeadDimension x ModelDim * ModelDim -> HeadDimension

    -- Trace a preview of wkHead before using it
    CArray2D wK = wkHead headComp
    !_ = trace ("wkHead row0 first 8 elems = " P.++ show (P.take 8 $ toList $ wK !! 0)) ()


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

-- Rotary application
applyRotaryToHead :: SingleHeadComponent -> StepCount -> (Vec HeadDimension Float , Vec HeadDimension Float ) -> (Vec HeadDimension Float , Vec HeadDimension Float )
applyRotaryToHead headComp step (q, k) = (q', k') where
    rot = rotary headComp
    q' = applyRotation rot step q
    k' = applyRotation rot step k

-- Apply rotation per head
applyRotation
  :: RotaryEncodingComponent
  -> StepCount
  -> Vec HeadDimension Float
  -> Vec HeadDimension Float
applyRotation rot step tokenVec =
  let cosFrequencies = getRow step (freqCos rot)
      sinFrequencies = getRow step (freqSin rot)
  in applyRotaryPositionEncoding tokenVec cosFrequencies sinFrequencies

-- Compute Q/K/V for a head, with optional rotary on Q/K
computeHeadQKV
  :: SingleHeadComponent
  -> StepCount
  -> Vec ModelDim Float
  -> (Vec HeadDimension Float, Vec HeadDimension Float, Vec HeadDimension Float)
computeHeadQKV headComp step xHat =
  let (q, k, v) = runSingleHeadQKV headComp xHat
      (qRot, kRot) = applyRotaryToHead headComp step (q, k)
  in (qRot, kRot, v)

projectQKV :: MultiHeadAttentionComponent
  -> ProcessingState
  -> Vec ModelDim Float
  -> (Vec NumQueryHeads (Vec HeadDimension Float), 
  Vec NumKeyValueHeads (Vec HeadDimension Float), 
  Vec NumKeyValueHeads (Vec HeadDimension Float))
projectQKV multiHeadAttentionComponent processingState inputVector = 
  let
    normalizedInput = rmsNorm inputVector (rmsAtt multiHeadAttentionComponent)
    stepCount = (StepCount $ fromIntegral $ sequencePosition processingState)
    -- Queries: one per Q head (with RoPE on Q)
    queries =
      imap (\queryHeadIdx _ ->
        let
          headComponent = heads multiHeadAttentionComponent !! queryHeadIdx
          (queryRotated, _, _) = computeHeadQKV headComponent stepCount normalizedInput
        in queryRotated) indicesI

    -- Keys/Values: one per KV head (apply RoPE to K only)
    keysAndValues =
      imap (\keyValueHeadIdx _ ->
        let qIdx0 = fromEnum keyValueHeadIdx *
                      (natToNum @NumQueryHeads `P.div` natToNum @NumKeyValueHeads)
            queryIndex = toEnum (min (natToNum @NumQueryHeads - 1) qIdx0) :: Index NumQueryHeads
            headComponent = heads multiHeadAttentionComponent !! queryIndex
            (_q, keyRotated, valueRotated) = computeHeadQKV headComponent stepCount normalizedInput
        in (keyRotated, valueRotated)) indicesI
    (keys, values) = unzip keysAndValues
  in (queries, keys, values)
