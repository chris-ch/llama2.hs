module Types
  ( NetworkConfig (..),
    TransformerWeighting (..),
    AttentionKV (..),
    StepCount (..),
    LayerIndex(..),
    HeadIndex(..),
    TokenVector(..),
    Vocabulary,
    VocabularyScores,
    PromptTokens,
    Token,
    MVectorFloat,
    Array2D(..),
    Array3D(..),
    readVector,
    readArray2D,
    readArray3D,
    getRow,
    getArray2D,
  )
where

import qualified Data.Binary.Get as BG
import qualified Data.ByteString.Lazy as BS
import Data.Int (Int32)
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV

--------------------------------------------------------------------------------
-- Types
--------------------------------------------------------------------------------

newtype HeadIndex = HeadIndex Int deriving (Show)

newtype LayerIndex = LayerIndex Int deriving (Show)

newtype StepCount = StepCount Int deriving (Show, Eq, Ord, Num)

newtype TokenVector = TokenVector (V.Vector Float) deriving (Show)

type MVectorFloat = MV.MVector (MV.PrimState IO) Float

type Vocabulary = [BS.ByteString]

type VocabularyScores = [Float]

type Token = Int32

type PromptTokens = [Token]

data AttentionKV = AttentionKV
  { keyCache :: MVectorFloat,
    valueCache :: MVectorFloat,
    gateOutput :: MVectorFloat,
    upProjectionOutput :: MVectorFloat,
    feedforwardNetworkOutput :: MVectorFloat,
    queryOutput :: MVectorFloat,
    keyOutput :: MVectorFloat,
    valueOutput :: MVectorFloat,
    projectedAttentionOutput :: MVectorFloat,
    multiHeadOutput :: MVectorFloat
  }

data TransformerWeighting = TransformerWeighting
  { tokenEmbeddingTable :: Array2D,
    rmsAttWeight :: Array2D,
    wq :: Array3D,
    wk :: Array3D,
    wv :: Array3D,
    wo :: Array3D,
    rmsFfnWeight :: Array2D,
    w1 :: Array3D,
    w2 :: Array3D,
    w3 :: Array3D,
    rmsFinalWeight :: V.Vector Float,
    freqCisReal :: Array2D,
    freqCisImag :: Array2D
  }
  deriving (Show)

data NetworkConfig = NetworkConfig
  { modelDim :: Int,
    hiddenDim :: Int,
    numLayers :: Int,
    numAttentionHeads :: Int,
    numKeyValueHeads :: Int,
    vocabSize :: Int,
    seqLen :: Int,
    headDimension :: Int,
    weighting :: TransformerWeighting
  }
  deriving (Show)

--------------------------------------------------------------------------------
-- Array2D Data Structure and Class
--------------------------------------------------------------------------------

data Array2D = Array2D
  { items2D :: V.Vector Float,
    nrows :: Int,
    ncols :: Int
  }
  deriving (Show)

data Array3D = Array3D
  { items3D :: V.Vector Float,
    sizeX :: Int,
    sizeY :: Int,
    sizeZ :: Int
  }
  deriving (Show)

class Array2DOps a where
  getRow :: Int -> a -> V.Vector Float
  readArray2D :: Int -> Int -> BG.Get a -- rows -> cols -> parser

instance Array2DOps Array2D where
  getRow :: Int -> Array2D -> V.Vector Float
  getRow i Array2D {items2D, ncols} = V.slice (i * ncols) ncols items2D

  readArray2D :: Int -> Int -> BG.Get Array2D
  readArray2D rows cols = do
    vec <- readVector (rows * cols)
    return Array2D {items2D = vec, nrows = rows, ncols = cols}

class Array3DOps a where
  getArray2D :: Int -> a -> Array2D
  readArray3D :: Int -> Int -> Int -> BG.Get a

instance Array3DOps Array3D where
  getArray2D :: Int -> Array3D -> Array2D
  getArray2D k Array3D {items3D, sizeX, sizeY, sizeZ}
    | k < 0 || k >= sizeX = error "getArray2D: index out of bounds"
    | otherwise =
        Array2D
          { items2D = V.slice (k * sizeY * sizeZ) (sizeY * sizeZ) items3D,
            nrows = sizeY,
            ncols = sizeZ
          }

  readArray3D :: Int -> Int -> Int -> BG.Get Array3D
  readArray3D sx sy sz = do
    let totalSize = sx * sy * sz
    vec <- readVector totalSize
    return Array3D {items3D = vec, sizeX = sx, sizeY = sy, sizeZ = sz}

readVector :: Int -> BG.Get (V.Vector Float)
readVector count = do
  byteData <- BG.getLazyByteString (fromIntegral count * 4) -- Read 4 bytes per Float
  return $ V.unfoldrN count parseFloatFromBytes byteData
  where
    parseFloatFromBytes :: BS.ByteString -> Maybe (Float, BS.ByteString)
    parseFloatFromBytes bs =
      if BS.null bs
        then Nothing
        else Just (BG.runGet BG.getFloatle (BS.take 4 bs), BS.drop 4 bs)
