module Model.Layers.Attention.MultiHeadAttention (
    MultiHeadAttentionComponent(..), projectQKV, StepCount(..)
) where

import Clash.Prelude

import Model.Core.Types (NumQueryHeads, ModelDim, NumKeyValueHeads, SeqLen, HeadDimension, CArray2D (..), SingleHeadComponent (..), FreqDim, RotaryEncodingComponent (..), ProcessingState (..))
import Helpers (matrixVectorMult, dotVec, rmsNorm)
import qualified Prelude as P

newtype StepCount = StepCount (Unsigned 32) deriving (Show, Eq, Ord)

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
computeMultiHeadAttention mha input qs kHeads vHeads pos =
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
    zipWith (+) input woAttn

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

-- QKV per head - should return HeadDimension vectors, not ModelDim
runSingleHeadQKV :: SingleHeadComponent -> Vec ModelDim Float -> (Vec HeadDimension Float, Vec HeadDimension Float, Vec HeadDimension Float)
runSingleHeadQKV headComp normalizedInput = (q, k, v) where
    q = matrixVectorMult (wqHead headComp) normalizedInput  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    k = matrixVectorMult (wkHead headComp) normalizedInput  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    v = matrixVectorMult (wvHead headComp) normalizedInput  -- HeadDimension x ModelDim * ModelDim -> HeadDimension

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

projectQKV :: MultiHeadAttentionComponent
  -> ProcessingState
  -> Vec ModelDim Float
  -> (Vec NumQueryHeads (Vec HeadDimension Float), 
  Vec NumKeyValueHeads (Vec HeadDimension Float), 
  Vec NumKeyValueHeads (Vec HeadDimension Float))
projectQKV multiHeadAttentionComponent processingState inputVector = 
  let
    normalizedInput = rmsNorm inputVector (rmsAtt multiHeadAttentionComponent)
    -- Queries: one per Q head (with RoPE on Q)
    queries =
      imap (\queryHeadIdx _ ->
        let headComponent = heads multiHeadAttentionComponent !! queryHeadIdx
            (query, _, _)   = runSingleHeadQKV headComponent normalizedInput
            (queryRotated, _kU) =
              applyRotaryToHead headComponent
                                (StepCount $ fromIntegral $ sequencePosition processingState)
                                (query, repeat 0)
        in queryRotated) indicesI

    -- Keys/Values: one per KV head (apply RoPE to K only)
    keysAndValues =
      imap (\keyValueHeadIdx _ ->
        let qIdx0 = fromEnum keyValueHeadIdx *
                      (natToNum @NumQueryHeads `P.div` natToNum @NumKeyValueHeads)
            queryIndex = toEnum (min (natToNum @NumQueryHeads - 1) qIdx0) :: Index NumQueryHeads
            headComponent     = heads multiHeadAttentionComponent !! queryIndex
            (_q, key, value)   = runSingleHeadQKV headComponent normalizedInput
            (_qU, keyRotated)  =
              applyRotaryToHead headComponent
                                (StepCount $ fromIntegral $ sequencePosition processingState)
                                (repeat 0, key)
        in (keyRotated, value)) indicesI
    (keys, values) = unzip keysAndValues
  in (queries, keys, values)
