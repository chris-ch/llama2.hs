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
  , SingleHeadComponent(..)
  , RotaryEncodingComponent(..)
  , EmbeddingComponent(..)
  , CArray2D(..)
  , Token
  , Temperature
  , Seed
  , dotVec
  , dotProduct
  , liftA5
  , rmsNorm
  , matrixVectorMult
  , liftA4
) where

import Clash.Prelude

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
type SeqLen         = 256

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

type Token = Unsigned 32
type Temperature = Float
type Seed = Unsigned 32

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

dotVec :: forall n. KnownNat n => Vec n Float -> Vec n Float -> Float
dotVec xs ys = sum (zipWith (*) xs ys)

liftA5 :: Applicative g => (a -> b -> c -> d -> e -> f) -> g a -> g b -> g c -> g d -> g e -> g f
liftA5 f fa fb fc fd fe = f <$> fa <*> fb <*> fc <*> fd <*> fe
