module Helpers where

import Clash.Prelude
import qualified Prelude as P
import qualified Foreign as F

-- model config 110M
type ModelDim = 768
type HiddenDim = 2048
type NumLayers = 12
type NumAttentionHeads = 12
vocabSize :: Integer
vocabSize = 32000 
type VocabSize = 32000 :: Nat
type SeqLen         = 1024
type HeadDimension  = 64
type FreqDim = 32

newtype CArray2D (n :: Nat) (m :: Nat) = CArray2D (Vec n (Vec m Float)) deriving (Show)

-- Index into CArray2D to get a row
getRow :: forall n m. (KnownNat n) => StepCount -> CArray2D n m -> Vec m Float
getRow (StepCount i) (CArray2D arr) = arr !! (fromIntegral i :: Index n)

newtype Token = Token F.Int32 deriving (Show, Eq, Ord, Num)
newtype StepCount' = StepCount' Int deriving (Show, Eq, Ord, Num)

newtype StepCount = StepCount (Unsigned 32) deriving (Show, Eq, Ord, Num)

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
  { heads    :: Vec NumAttentionHeads SingleHeadComponent
  , mWo      :: CArray2D ModelDim ModelDim -- Output projection matrix
  , rmsAtt   :: Vec ModelDim Float -- RMSNorm before QKV projection
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

-- Dot product of two Vecs
dotProduct :: KnownNat n => Vec n Float -> Vec n Float -> Float
dotProduct v1 v2 = sum $ zipWith (*) v1 v2

-- Vector multiplication by a Matrix
matrixVectorMult :: forall n m. ( KnownNat m) => CArray2D n m -> Vec m Float -> Vec n Float
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
