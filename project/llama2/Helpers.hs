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
  , Temperature
  , Seed
  , dotProduct
  , liftA5
  , runSingleHeadQKV
  , applyRotaryToHead
  , rmsNorm
  , computeAttentionWeights
  , computeAttentionScores
  , computeAttentionOutput
  , matrixVectorMult
  , computeFeedForward
  , embed
  , transformerLogits
  , argMax
  , computeMultiHeadAttention
  , vocabSize
  , seqLen
  , liftA4
  , softmax
  , sampleFromProbs
  , xorshift32
) where

import Clash.Prelude

import qualified Clash.Sized.Vector as CV
import Data.Maybe (fromMaybe)
{- 
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
 -}

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

{- 
-- model config 42M
type ModelDim = 512
type HiddenDim = 1376
type NumLayers = 8
type NumQueryHeads = 8
type NumKeyValueHeads = 8
type HeadDimension  = 64
type FreqDim = 32
type VocabSize = 32000 :: Nat
vocabSize :: Int
vocabSize = natToNum @VocabSize
type SeqLen         = 1024
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

newtype CArray2D (n :: Nat) (m :: Nat) = CArray2D (Vec n (Vec m Float)) deriving (Show)

-- Index into CArray2D to get a row
getRow :: forall n m. (KnownNat n) => StepCount -> CArray2D n m -> Vec m Float
getRow (StepCount i) (CArray2D arr) = arr !! (fromIntegral i :: Index n)

type Token = Unsigned 32
type Temperature = Float
type Seed = Unsigned 32
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
runFeedForward ffn xHat =
  let
    w1 = fW1 ffn
    w2 = fW2 ffn
    w3 = fW3 ffn
    gatePre   = matrixVectorMult w1 xHat
    upPre     = matrixVectorMult w3 xHat
    gate      = map sigmoidLinearUnit gatePre
  in
    matrixVectorMult w2 (zipWith (*) gate upPre)

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

softmax :: forall n. KnownNat (n + 1) => Float -> Vec (n + 1) Float -> Vec (n + 1) Float
softmax t xs =
  let
    m    = maximum xs
    exps = map (\x -> exp ((x - m) / t)) xs
    s    = sum exps
  in map (/ s) exps

-- Pure, synthesizable categorical sampling from probabilities summing to ~1.0
sampleFromProbs :: forall n. (KnownNat (n + 1), KnownNat n) => Float -> Vec (n + 1) Float -> Unsigned 32
sampleFromProbs u probs =
  let
    cdf = CV.scanl1 (+) probs
    idx = fromMaybe maxBound (findIndex (>= u) cdf)
  in fromIntegral (fromEnum idx)

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

-- Pure feed-forward computation
computeFeedForward
  :: FeedForwardNetworkComponent
  -> Vec ModelDim Float
  -> Vec ModelDim Float
computeFeedForward ffn x =
  let xHat     = rmsNorm x (fRMSFfn ffn)                  -- single pre-norm here
      ffnCore  = runFeedForward ffn xHat                  -- no extra norm inside
  in zipWith (+) x ffnCore

liftA5 :: Applicative g => (a -> b -> c -> d -> e -> f) -> g a -> g b -> g c -> g d -> g e -> g f
liftA5 f fa fb fc fd fe = f <$> fa <*> fb <*> fc <*> fd <*> fe

-- xorshift32 core (synthesizable, 1-cycle combinational)
xorshift32 :: Unsigned 32 -> Unsigned 32
xorshift32 s0 =
  let s1 = s0 `xor` shiftL s0 13
      s2 = s1 `xor` shiftR s1 17
      s3 = s2 `xor` shiftL s2 5
  in s3
