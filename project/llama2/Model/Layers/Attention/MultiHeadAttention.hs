module Model.Layers.Attention.MultiHeadAttention (
    MultiHeadAttentionComponent(..), projectQKV, StepCount(..)
) where

import Clash.Prelude

import Model.Core.Types (NumQueryHeads, ModelDim, NumKeyValueHeads,
  HeadDimension, CArray2D (..), SingleHeadComponent (..),
  FreqDim, RotaryEncodingComponent (..), ProcessingState (..), NumLayers)
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
  -> Index NumKeyValueHeads
  -> Index NumLayers
  -> (Vec HeadDimension Float, Vec HeadDimension Float)
computeHeadKV headComp step xHat headIdx layerIdx =
  let
    k = matrixVectorMult (wkHead headComp) xHat  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    v = matrixVectorMult (wvHead headComp) xHat  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    kRot = applyRotation (rotary headComp) step k
    CArray2D _wK = wkHead headComp
    StepCount stp = step
    !_ = traceIf (headIdx == 0)
                ("[TRACE][L" P.++ P.show layerIdx P.++ " P" P.++ P.show stp P.++ "] k before rot first 8 elems = " P.++ P.show (P.take 8 $ toList k))
                ()
    !_ = traceIf (headIdx == 0)
                ("[TRACE][L" P.++ P.show layerIdx P.++ " P" P.++ P.show stp P.++ "] wkHead row0 first 8 elems = " P.++ P.show (P.take 8 $ toList (_wK !! 0)))
                ()
    !_ = traceIf (headIdx == 0)
                ("[TRACE][L" P.++ P.show layerIdx P.++ " P" P.++ P.show stp P.++ "] v first 8 elems = " P.++ P.show (P.take 8 $ toList v))
                ()
    !_ = traceIf (headIdx == 0)
                ("[TRACE][L" P.++ P.show layerIdx P.++ " P" P.++ P.show stp P.++ "] k after rot first 8 elems = " P.++ P.show (P.take 8 $ toList kRot))
                ()
  in (kRot, v)

-- Compute Q for a head
computeHeadQ
  :: SingleHeadComponent
  -> StepCount
  -> Vec ModelDim Float
  -> Index NumKeyValueHeads
  -> Index NumLayers
  -> Vec HeadDimension Float
computeHeadQ headComp step xHat headIdx layerIdx =
  let
    q = matrixVectorMult (wqHead headComp) xHat  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    qRot = applyRotation (rotary headComp) step q

    CArray2D _wK = wkHead headComp
    StepCount stp = step
    -- Trace only when we are processing the very first head (headIdx == 0)
    !_ = traceIf (headIdx == 0)
                ("[TRACE][L" P.++ P.show layerIdx P.++ " P" P.++ P.show stp P.++ "] q before rot first 8 elems = " P.++ P.show (P.take 8 $ toList q))
                ()
    !_ = traceIf (headIdx == 0)
                ("[TRACE][L" P.++ P.show layerIdx P.++ " P" P.++ P.show stp P.++ "] q after rot first 8 elems = " P.++ P.show (P.take 8 $ toList qRot))
                ()
  in qRot

traceIf :: Bool -> String -> a -> a
traceIf True  msg x = trace msg x
traceIf False _   x = x

projectQKV :: MultiHeadAttentionComponent
  -> ProcessingState
  -> Vec ModelDim Float
  -> Index NumLayers
  -> (Vec NumQueryHeads (Vec HeadDimension Float), 
  Vec NumKeyValueHeads (Vec HeadDimension Float), 
  Vec NumKeyValueHeads (Vec HeadDimension Float))
projectQKV multiHeadAttentionComponent processingState inputVector layerIdx = 
  let
    normalizedInput = rmsNorm inputVector (rmsAtt multiHeadAttentionComponent)
    stepCount = (StepCount $ fromIntegral $ sequencePosition processingState)
    -- Queries: one per Q head (with RoPE on Q)
    queries =
      imap (\queryHeadIdx _ ->
        let
          headComponent = heads multiHeadAttentionComponent !! queryHeadIdx
          queryRotated = computeHeadQ headComponent stepCount normalizedInput queryHeadIdx layerIdx
        in queryRotated) indicesI

    -- Keys/Values: one per KV head (apply RoPE to K only)
    keysAndValues =
      imap (\keyValueHeadIdx _ ->
        let
          qIdx0 = fromEnum keyValueHeadIdx * (natToNum @NumQueryHeads `P.div` natToNum @NumKeyValueHeads)
          queryIndex = toEnum (min (natToNum @NumQueryHeads - 1) qIdx0) :: Index NumQueryHeads
          headComponent = heads multiHeadAttentionComponent !! queryIndex
          (keyRotated, valueRotated) = computeHeadKV headComponent stepCount normalizedInput keyValueHeadIdx layerIdx
        in (keyRotated, valueRotated)) indicesI
    (keys, values) = unzip keysAndValues
  in (queries, keys, values)
