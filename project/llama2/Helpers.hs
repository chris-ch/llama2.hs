module Helpers (
  NumQueryHeads
  , NumLayers
  , NumKeyValueHeads
  , SeqLen
  , HeadDimension
  , ModelDim
  , FreqDim
  , HiddenDim
  , VocabSize
  , TransformerLayerComponent(..)
  , TransformerDecoderComponent(..)
  , SingleHeadComponent(..)
  , MultiHeadAttentionComponent(..)
  , RotaryEncodingComponent(..)
  , FeedForwardNetworkComponent(..)
  ,EmbeddingComponent(..)
  , StepCount(..)
  , CArray2D(..)
  , Token
  , runSingleHeadQKV
  ,applyRotaryToHead
  , rmsNorm
  , computeAttentionWeights
  , computeAttentionScores
  , computeAttentionOutput
  , matrixVectorMult
  , computeFeedForward
  , embed
  , transformerLogits
  , argMax
  , drawSample
  , computeMultiHeadAttention
  , vocabSize
  , seqLen
  , liftA4
) where

import Clash.Prelude
import qualified System.Random as R
import qualified Clash.Sized.Vector as CV
import Data.Maybe (fromMaybe)

-- model config 260K
type ModelDim = 64
type HiddenDim = 172
type NumLayers = 5
type NumQueryHeads = 8
type NumKeyValueHeads = 4
type HeadDimension  = 8
type FreqDim = 4
type VocabSize = 512 :: Nat
vocabSize :: Int
vocabSize = natToNum @VocabSize
type SeqLen         = 512
seqLen :: Int
seqLen = natToNum @SeqLen

{- 
-- model config 15M
type ModelDim = 288
type HiddenDim = 768
type NumLayers = 6
type NumQueryHeads = 6
type NumKeyValueHeads = 6
type HeadDimension  = 48
type FreqDim = 24
type VocabSize = 32000 :: Nat
vocabSize :: Int
vocabSize = natToNum @VocabSize
type SeqLen         = 256
seqLen :: Int
seqLen = natToNum @SeqLen
 -}
{-
-- model config 110M
type ModelDim = 768
type HiddenDim = 2048
type NumLayers = 12
type NumQueryHeads = 12
type NumKeyValueHeads = 12
type HeadDimension  = 64
type FreqDim = 32
type VocabSize = 32000 :: Nat
vocabSize :: Int
vocabSize = natToNum @VocabSize
type SeqLen         = 1024
seqLen :: Int
seqLen = natToNum @SeqLen
-}

{--------------------------------------------------- 

42M
modelDim 512
hiddenDim 1376
numLayers 8
NumQueryHeads 8
headDimension 64
FreqDim 32
numKeyValueHeads 8
vocabSize 32000
seqLen 1024

 ---------------------------------------------------}

newtype CArray2D (n :: Nat) (m :: Nat) = CArray2D (Vec n (Vec m Float)) deriving (Show)

-- Index into CArray2D to get a row
getRow :: forall n m. (KnownNat n) => StepCount -> CArray2D n m -> Vec m Float
getRow (StepCount i) (CArray2D arr) = arr !! (fromIntegral i :: Index n)

type Token = Unsigned 32

newtype StepCount = StepCount (Unsigned 32) deriving (Show, Eq, Ord)

-- Data definitions for LLM architecture

data EmbeddingComponent = EmbeddingComponent
  { vocabulary :: CArray2D VocabSize ModelDim,
    rmsFinalWeight :: Vec ModelDim Float
  } deriving (Show)

data RotaryEncodingComponent = RotaryEncodingComponent
  { freqCos :: CArray2D SeqLen FreqDim,
    freqSin :: CArray2D SeqLen FreqDim
  } deriving (Show)

data SingleHeadComponent = SingleHeadComponent
  { wqHead :: CArray2D HeadDimension ModelDim
  , wkHead :: CArray2D HeadDimension ModelDim
  , wvHead :: CArray2D HeadDimension ModelDim
  , rotary :: RotaryEncodingComponent
  } deriving (Show)

data MultiHeadAttentionComponent = MultiHeadAttentionComponent
  { heads  :: Vec NumQueryHeads SingleHeadComponent
  -- | Per-head output projection matrix W_O (shape HeadDim × ModelDim)
  , mWo :: Vec NumQueryHeads (CArray2D ModelDim HeadDimension)
  -- | RMSNorm before QKV projection (size ModelDim)
  , rmsAtt :: Vec ModelDim Float
  } deriving (Show)

data FeedForwardNetworkComponent = FeedForwardNetworkComponent
  { fW1 :: CArray2D HiddenDim ModelDim,
    fW2 :: CArray2D ModelDim HiddenDim,
    fW3 :: CArray2D HiddenDim ModelDim,
    fRMSFfn :: Vec ModelDim Float
  } deriving (Show)

data TransformerLayerComponent = TransformerLayerComponent
  {
    multiHeadAttention :: MultiHeadAttentionComponent,
    feedforwardNetwork :: FeedForwardNetworkComponent
  } deriving (Show)

data TransformerDecoderComponent = TransformerDecoderComponent
  { modelEmbedding :: EmbeddingComponent,
    modelLayers :: Vec NumLayers TransformerLayerComponent
  } deriving (Show)

liftA4 :: Applicative f => (a -> b -> c -> d -> e) -> f a -> f b -> f c -> f d -> f e
liftA4 f a b c d = liftA3 f a b c <*> d

-- Dot product of two Vecs
dotProduct :: KnownNat n => Vec n Float -> Vec n Float -> Float
dotProduct v1 v2 = sum $ zipWith (*) v1 v2

-- Vector multiplication by a Matrix
matrixVectorMult :: forall n m. KnownNat m => CArray2D n m -> Vec m Float -> Vec n Float
matrixVectorMult (CArray2D mat) vec = map (`dotProduct` vec) mat

-- RMS Norm
rmsNorm :: Vec ModelDim Float -> Vec ModelDim Float -> Vec ModelDim Float
rmsNorm vec weights =
  let squareNorm = sum (map (\v -> v * v) vec)
      ss = (squareNorm / fromIntegral (length vec)) + 1e-5
      normalized = map (* (1.0 / sqrt ss)) vec
  in zipWith (*) weights normalized

-- Activation
sigmoidLinearUnit :: Float -> Float
sigmoidLinearUnit x = x / (1.0 + exp (-x))

-- Embed a token
embed :: CArray2D VocabSize ModelDim -> Token -> Vec ModelDim Float
embed (CArray2D vocab) tokenCode = vocab !! (fromIntegral tokenCode :: Int)

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

runFeedForward :: FeedForwardNetworkComponent -> Vec ModelDim Float -> Vec ModelDim Float
runFeedForward feedForwardNetwork inputToken = feedforwardNetworkOutput' where
    rmsFfnWeights = fRMSFfn feedForwardNetwork
    w1 = fW1 feedForwardNetwork
    w2 = fW2 feedForwardNetwork
    w3 = fW3 feedForwardNetwork
    normalizedInput = rmsNorm inputToken rmsFfnWeights
    gateOutput' = matrixVectorMult w1 normalizedInput
    upProjectionOutput' = matrixVectorMult w3 normalizedInput
    gateOutput = map sigmoidLinearUnit gateOutput'

    feedforwardNetworkOutput' = matrixVectorMult w2 (zipWith (*) gateOutput upProjectionOutput')

-- QKV per head - should return HeadDimension vectors, not ModelDim
runSingleHeadQKV :: SingleHeadComponent -> Vec ModelDim Float -> (Vec HeadDimension Float, Vec HeadDimension Float, Vec HeadDimension Float)
runSingleHeadQKV headComp normalizedInput = (q, k, v) where
    q = matrixVectorMult (wqHead headComp) normalizedInput  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    k = matrixVectorMult (wkHead headComp) normalizedInput  -- HeadDimension x ModelDim * ModelDim -> HeadDimension
    v = matrixVectorMult (wvHead headComp) normalizedInput  -- HeadDimension x ModelDim * ModelDim -> HeadDimension

-- Rotary application
applyRotaryToHead :: SingleHeadComponent -> StepCount -> (Vec HeadDimension Float , Vec HeadDimension Float ) -> (Vec HeadDimension Float , Vec HeadDimension Float )
applyRotaryToHead headComp step (q, k) = (q', k') where
    rot = rotary headComp
    q' = applyRotation rot step q
    k' = applyRotation rot step k

-- classifier logits for a given token vector
transformerLogits :: TransformerDecoderComponent -> Vec ModelDim Float -> Vec VocabSize Float
transformerLogits decoder tokenVector = logits where
    vocab = vocabulary (modelEmbedding decoder)
    rmsWeight = rmsFinalWeight (modelEmbedding decoder)
    tokenWithRms = rmsNorm tokenVector rmsWeight
    CArray2D vocabRows = vocab
    logits = map (`dotProduct` tokenWithRms) vocabRows

-- | Find the index of the maximum element in a non-empty vector
argMax :: forall n. ( KnownNat (n + 1)) =>Vec (n+1) Float -> Unsigned 32
argMax vec = fst $ foldl compareMax (0, head vec) (imap (\i x -> (fromIntegral i, x)) vec)
  where
    compareMax :: (Unsigned 32, Float) -> (Unsigned 32, Float) -> (Unsigned 32, Float)
    compareMax (maxIdx, maxVal) (i, x)
      | x > maxVal = (i, x)
      | otherwise  = (maxIdx, maxVal)

dotVec :: forall n. KnownNat n => Vec n Float -> Vec n Float -> Float
dotVec xs ys = sum (zipWith (*) xs ys)

softmaxVec :: forall n. KnownNat (n+1) => Vec (n+1) Float -> Vec (n+1) Float
softmaxVec xs =
  let m = maximum xs
      exps = map (\x -> exp (x - m)) xs
      s = sum exps
  in map (/ s) exps

-- Pure deterministic sampling from probabilities
drawSample :: Int -> Vec VocabSize Float -> Unsigned 32
drawSample seed probabilities =
    let gen = R.mkStdGen seed
        (randomValue, _) = R.random gen :: (Float, R.StdGen)

        -- cumulative sum using scanl1'
        cumulativeDistribution :: Vec VocabSize Float
        cumulativeDistribution = CV.scanl1 (+) probabilities

        -- find first index where cumulative >= randomValue
        selectedIndex :: Index VocabSize
        selectedIndex = fromMaybe maxBound (findIndex (>= randomValue) cumulativeDistribution)

    in fromIntegral (fromEnum selectedIndex)

-- Pure attention computation
computeAttentionScores
  :: Vec HeadDimension Float        -- query
  -> Vec SeqLen (Vec HeadDimension Float)  -- keys
  -> Vec SeqLen Float
computeAttentionScores query keys =
  let headDim = snatToNum (SNat @HeadDimension)
      scaling = sqrt (headDim :: Float)
  in map (\key -> dotVec query key / scaling) keys

-- Pure attention weights computation
computeAttentionWeights
  :: Vec SeqLen Float
  -> Vec SeqLen Float
computeAttentionWeights = softmaxVec

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

-- Pure multi-head attention computation
computeMultiHeadAttention
  :: MultiHeadAttentionComponent
  -> Vec ModelDim Float
  -> Vec NumQueryHeads (Vec HeadDimension Float)
  -> Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float))
  -> Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float))
  -> Vec ModelDim Float
computeMultiHeadAttention mha input queries keysPerHead valuesPerHead =
  let
    -- Compute the number of query heads per key/value head
    headsPerGroup :: Int
    headsPerGroup = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads

    -- Map each query head to its corresponding key/value head
    getKVIndex :: Index NumQueryHeads -> Index NumKeyValueHeads
    getKVIndex qIdx = fromIntegral (fromIntegral qIdx `div` headsPerGroup :: Int)

    -- Compute per-head outputs (HeadDimension each)
    headOutputs :: Vec NumQueryHeads (Vec HeadDimension Float)
    headOutputs = imap
        (\qIdx q ->
           let kvIdx = getKVIndex qIdx
               ks = keysPerHead !! kvIdx
               vs = valuesPerHead !! kvIdx
               scores = computeAttentionScores q ks
               weights = computeAttentionWeights scores
           in computeAttentionOutput weights vs)
        queries

    -- Project each head’s output via its own W_O
    perHeadProjected :: Vec NumQueryHeads (Vec ModelDim Float)
    perHeadProjected = zipWith matrixVectorMult (mWo mha) headOutputs

    -- Sum across heads (elementwise sum across ModelDim)
    summedOutput :: Vec ModelDim Float
    summedOutput = foldl1 (zipWith (+)) perHeadProjected

    -- Residual connection
    normalizedInput = rmsNorm input (rmsAtt mha)
  in zipWith (+) normalizedInput summedOutput

-- Pure feed-forward computation
computeFeedForward
  :: FeedForwardNetworkComponent
  -> Vec ModelDim Float
  -> Vec ModelDim Float
computeFeedForward ffn input =
  let normalizedInput = rmsNorm input (fRMSFfn ffn)
      ffnOutput = runFeedForward ffn normalizedInput
  in zipWith (+) input ffnOutput
