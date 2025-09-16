module Helpers where

import Clash.Prelude
import qualified Prelude as P
import qualified Foreign as F

-- model config 110M
modelDim :: Int
modelDim = 768
type ModelDim = 768
hiddenDim :: Int
hiddenDim = 2048
type HiddenDim = 2048
numLayers :: Int
numLayers      = 12
type NumLayers = 12
numAttentionHeads :: Int
numAttentionHeads = 12
type NumAttentionHeads = 12
vocabSize :: Int
vocabSize = 32000
type VocabSize = 32000
seqLen :: Int
seqLen = 1024
type SeqLen         = 1024
headDimension :: Int
headDimension = 64
type HeadDimension  = 64
type FreqDim = 32

newtype CVector (n :: Nat) = CVector (Vec n Float) deriving (Show)
newtype CArray2D (n :: Nat) (m :: Nat) = CArray2D (Vec n (Vec m Float)) deriving (Show)

-- Index into CArray2D to get a row
getRow :: forall n m. (KnownNat n) => StepCount -> CArray2D n m -> CVector m
getRow (StepCount i) (CArray2D arr) = CVector $ arr !! (fromIntegral i :: Index n)

newtype Token = Token F.Int32 deriving (Show, Eq, Ord, Num)
newtype StepCount = StepCount Int deriving (Show, Eq, Ord, Num)

-- Data definitions for LLM architecture

data EmbeddingComponent = EmbeddingComponent
  { vocabulary :: CArray2D VocabSize ModelDim,
    rmsFinalWeight :: CVector ModelDim
  } deriving (Show)

data RotaryEncodingComponent = RotaryEncodingComponent
  { freqCos :: CArray2D SeqLen FreqDim,
    freqSin :: CArray2D SeqLen FreqDim
  } deriving (Show)

data SingleHeadComponent = SingleHeadComponent
  { wqHead :: CArray2D ModelDim ModelDim
  , wkHead :: CArray2D ModelDim ModelDim
  , wvHead :: CArray2D ModelDim ModelDim
  , rotary :: RotaryEncodingComponent
  } deriving (Show)

data MultiHeadAttentionComponent = MultiHeadAttentionComponent
  { heads    :: Vec NumAttentionHeads SingleHeadComponent
  , mWo      :: CArray2D ModelDim ModelDim         -- Output projection matrix
  , rmsAtt   :: CVector ModelDim  -- RMSNorm before QKV projection
  } deriving (Show)

data FeedForwardNetworkComponent = FeedForwardNetworkComponent
  { fW1 :: CArray2D HiddenDim ModelDim,
    fW2 :: CArray2D ModelDim HiddenDim,
    fW3 :: CArray2D HiddenDim ModelDim,
    fRMSFfn :: CVector ModelDim
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

-- Dot product of two Vecs
dotProduct :: KnownNat n => Vec n Float -> Vec n Float -> Float
dotProduct v1 v2 = sum $ zipWith (*) v1 v2

-- Vector multiplication by a Matrix
matrixVectorMult :: forall n m. ( KnownNat m) => CArray2D n m -> CVector m -> CVector n
matrixVectorMult (CArray2D mat) (CVector vec) =
  CVector $ map (`dotProduct` vec) mat

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

-- Softmax for non-empty vectors
softmax :: (KnownNat (n + 1)) => Vec (n + 1) Float -> Vec (n + 1) Float
softmax values = map (/ sumExpValues) expValues
  where
    maxVal = maximum values
    expValues = map (\x -> exp (x - maxVal)) values
    sumExpValues = sum expValues

-- Embed a token
embed :: CArray2D VocabSize ModelDim -> Token -> Vec ModelDim Float
embed (CArray2D vocab) (Token tokenCode) = vocab !! (fromIntegral tokenCode :: Int)

-- Apply rotation per head
applyRotation
  :: RotaryEncodingComponent
  -> StepCount
  -> Vec ModelDim Float
  -> Vec ModelDim Float
applyRotation rot step tokenVec =
  let CVector cosFrequencies = getRow step (freqCos rot)
      CVector sinFrequencies = getRow step (freqSin rot)
  in applyRotaryPositionEncoding tokenVec cosFrequencies sinFrequencies

applyRotaryPositionEncoding :: Vec ModelDim Float    -- input vector
  -> Vec FreqDim Float  -- cosFrequencies
  -> Vec FreqDim Float  -- sinFrequencies
  -> Vec ModelDim Float
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

-- Elementwise multiplication of two CVector
cVecZipWith :: (Float -> Float -> Float)
            -> CVector n
            -> CVector n
            -> CVector n
cVecZipWith f (CVector v1) (CVector v2) = CVector $ zipWith f v1 v2

runFeedForward :: FeedForwardNetworkComponent -> CVector ModelDim -> CVector ModelDim
runFeedForward feedForwardNetwork (CVector inputToken) = feedforwardNetworkOutput' where
    CVector rmsFfnWeights = fRMSFfn feedForwardNetwork
    w1 = fW1 feedForwardNetwork
    w2 = fW2 feedForwardNetwork
    w3 = fW3 feedForwardNetwork
    normalizedInput = CVector $ rmsNorm inputToken rmsFfnWeights
    CVector gateOutput' = matrixVectorMult w1 normalizedInput
    upProjectionOutput' = matrixVectorMult w3 normalizedInput
    gateOutput = CVector $ map sigmoidLinearUnit gateOutput'

    feedforwardNetworkOutput' = matrixVectorMult w2 (cVecZipWith (*) gateOutput upProjectionOutput')

-- QKV per head
runSingleHeadQKV :: SingleHeadComponent -> Vec ModelDim Float -> (CVector ModelDim, CVector ModelDim, CVector ModelDim)
runSingleHeadQKV headComp normalizedInput = (q, k, v) where
    q = matrixVectorMult (wqHead headComp) (CVector normalizedInput)
    k = matrixVectorMult (wkHead headComp) (CVector normalizedInput)
    v = matrixVectorMult (wvHead headComp) (CVector normalizedInput)

-- Rotary application
applyRotaryToHead :: SingleHeadComponent -> StepCount -> (Vec ModelDim Float , Vec ModelDim Float ) -> (Vec ModelDim Float , Vec ModelDim Float )
applyRotaryToHead headComp step (q, k) = (q', k') where
    rot = rotary headComp
    q' = applyRotation rot step q
    k' = applyRotation rot step k

-- classifier logits for a given token vector
transformerLogits :: TransformerDecoderComponent -> Vec ModelDim Float -> Vec VocabSize Float
transformerLogits decoder tokenVector = logits where
    vocab = vocabulary (modelEmbedding decoder)
    CVector rmsWeight = rmsFinalWeight (modelEmbedding decoder)
    tokenWithRms = rmsNorm tokenVector rmsWeight
    CArray2D vocabRows = vocab
    logits = map (`dotProduct` tokenWithRms) vocabRows
