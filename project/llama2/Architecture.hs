{-# LANGUAGE FlexibleContexts #-}
module Architecture
  ( Embedding (..),
    RotaryEncoding (..),
    MultiHeadAttention (..),
    FeedForwardNetwork (..),
    TransformerLayer (..),
    TransformerDecoder (..),
    TransformerParams(..),
    transformerLogits, computeQKV, TransformerResult
    , NetworkConfig (..)
    , embed, runFeedForward, runAttention, cacheIndex
  ) where

import Control.Monad.Reader (MonadIO (liftIO), MonadReader (ask), ReaderT)
import Control.Monad.State (StateT, gets)
import Control.Monad ( foldM)
import qualified Data.Vector.Unboxed as V


import Types
    ( getArray2D,
      Array2D(items2D, ncols, nrows),
      AttentionKV(..),
      Token,
      TokenVector(..),
      StepCount(..),
      LayerIndex(..), Array3D, HeadIndex (..), getRow )
import Primitives
    (
      softmax, rmsNorm, sigmoidLinearUnit, applyRotaryPositionEncoding, matrixVectorMult )
import Data.List (foldl')

-- Data definitions mirroring architecture boxes
newtype Embedding = Embedding
  { vocabulary :: Array2D
  } deriving (Show)

data RotaryEncoding = RotaryEncoding
  { freqCos :: Array2D,
    freqSin :: Array2D
  } deriving (Show)

data MultiHeadAttention = MultiHeadAttention
  { mWq     :: Array2D            -- Q projection
  , mWk     :: Array2D            -- K projection
  , mWv     :: Array2D            -- V projection
  , mWo     :: Array2D            -- Output projection
  , mRMSAtt :: V.Vector Float     -- RMS normalization for input
  , mRotary  :: RotaryEncoding     -- Rotary embeddings for attention
  } deriving (Show)

data FeedForwardNetwork = FeedForwardNetwork
  { fW1 :: Array2D,
    fW2 :: Array2D,
    fW3 :: Array2D,
    fRMSFfn :: V.Vector Float
  } deriving (Show)

data TransformerLayer = TransformerLayer
  {
    multiHeadAttention :: MultiHeadAttention,
    feedforwardNetwork :: FeedForwardNetwork
  } deriving (Show)

data TransformerDecoder = TransformerDecoder
  { modelEmbedding :: Embedding,
    modelRotary :: RotaryEncoding,
    modelLayers :: [TransformerLayer]
  } deriving (Show)

class Embeddable e where
  embed :: e -> Token -> TransformerResult TokenVector

class RotaryEncodable r where
  applyRotary :: r -> StepCount -> HeadIndex -> V.Vector Float -> TransformerResult (V.Vector Float)

class FeedForwardLayer f where
  runFeedForward :: f -> V.Vector Float -> TransformerResult (V.Vector Float)

class AttentionLayer a where
  runAttention :: a -> LayerIndex -> V.Vector Float -> StepCount -> TransformerResult (V.Vector Float)

class TransformerBlock l where
  runBlock :: l -> V.Vector Float -> StepCount -> TransformerResult (V.Vector Float)

class TransformerModel m where
  runModel :: m -> Token -> StepCount -> TransformerResult (V.Vector Float)

instance AttentionLayer MultiHeadAttention where
  runAttention :: MultiHeadAttention -> LayerIndex -> V.Vector Float -> StepCount -> TransformerResult (V.Vector Float)
  runAttention mha layerIndex inputToken currentStep = do
    network <- ask
    let numHeads = numAttentionHeads network
        headDim  = headDimension network

    let headsQ = [ V.slice (i * headDim) headDim inputToken | i <- [0 .. numHeads - 1] ]

    -- 1. Normalize input
    let normalizedInput = rmsNorm inputToken (mRMSAtt mha)

    -- 2. Compute Q/K/V for the whole layer
    let qAll = matrixVectorMult (mWq mha) normalizedInput
        kAll = matrixVectorMult (mWk mha) normalizedInput
        vAll = matrixVectorMult (mWv mha) normalizedInput
        
    -- 3. Process each head
{-     headOutputs <- mapM (\hIdx -> do
        let qHead = V.slice (hIdx * headDim) headDim qAll
            kHead = V.slice (hIdx * headDim) headDim kAll
            vHead = V.slice (hIdx * headDim) headDim vAll

        -- Apply rotary encoding per head
        rotatedQ <- applyRotary (mRotary mha) currentStep (HeadIndex hIdx) qHead
        rotatedK <- applyRotary (mRotary mha) currentStep (HeadIndex hIdx) kHead

        -- Compute attention using the cache
        headAttention layerIndex (HeadIndex hIdx) step rotatedQ rotatedK vHead
      ) [0 .. numHeads - 1] -}
    
    -- 4. Concatenate head outputs
    headOutputs <- mapM (\i -> headAttention layerIndex (HeadIndex i) currentStep (headsQ !! i)) [0 .. numHeads - 1]
    let concatenated = V.concat headOutputs

    -- 5. Apply output projection
    --return $ matrixVectorMult (mWo mha) concatenated

    return concatenated

instance Embeddable Embedding where
  embed :: Embedding -> Token -> TransformerResult TokenVector
  embed (Embedding vocabulary) tokenCode = do
    let
        rowStart = fromIntegral tokenCode * ncols vocabulary
        tokenVector = TokenVector $ V.slice rowStart (ncols vocabulary) (items2D vocabulary)
    return tokenVector

instance RotaryEncodable RotaryEncoding where
  applyRotary :: RotaryEncoding -> StepCount -> HeadIndex -> V.Vector Float -> TransformerResult (V.Vector Float)
  applyRotary (RotaryEncoding freqCos freqSin) (StepCount step) headIndex input = do
    network <- ask
    let
        cosFrequencies = getRow step freqCos
        sinFrequencies = getRow step freqSin
        headDim = headDimension network

    -- Apply rotation per head
    return $ applyRotaryPositionEncoding headDim headIndex cosFrequencies sinFrequencies input

instance FeedForwardLayer FeedForwardNetwork where
  runFeedForward :: FeedForwardNetwork -> V.Vector Float -> TransformerResult (V.Vector Float)
  runFeedForward ffn inputToken = do
    let
      rmsFfnWeights = fRMSFfn ffn
      w1 = fW1 ffn
      w2 = fW2 ffn
      w3 = fW3 ffn
      normalizedInput = rmsNorm inputToken rmsFfnWeights
      gateOutput' = matrixVectorMult w1 normalizedInput
      upProjectionOutput' = matrixVectorMult w3 normalizedInput
      gateOutput = V.map sigmoidLinearUnit gateOutput'
      feedforwardNetworkOutput' = matrixVectorMult w2 (V.zipWith (*) gateOutput upProjectionOutput')
    return feedforwardNetworkOutput' 

data TransformerParams = TransformerParams
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
    params :: TransformerParams,
    decoder :: TransformerDecoder
  }
  deriving (Show)

-- Runtime type alias
type TransformerResult a = ReaderT NetworkConfig (StateT AttentionKV IO) a

headAttention
  :: LayerIndex
  -> HeadIndex
  -> StepCount
  -> V.Vector Float
  -> TransformerResult (V.Vector Float)
headAttention layerIndex (HeadIndex hIdx) currentStep headQuery = do
  network <- ask
  kCache  <- gets keyCache
  vCache  <- gets valueCache

  -- freeze caches once
  kVec <- liftIO $ V.unsafeFreeze kCache
  vVec <- liftIO $ V.unsafeFreeze vCache

  let headDim = headDimension network

  -- compute scores (sequenceLength = step + 1)
  let rawScores = computeScores network layerIndex (HeadIndex hIdx) currentStep headDim kVec headQuery
      softValues = softmax rawScores (V.length rawScores)
      sequenceLength = V.length softValues   -- should be step + 1
      headScores = V.toList softValues

      zero = V.replicate headDim 0.0
      addScaled acc pos =
        let valueCacheOffset = cacheIndex network (StepCount pos) layerIndex (HeadIndex hIdx) 0
            vSlice = V.slice valueCacheOffset headDim vVec
            score  = headScores !! pos
            scaled = V.map (* score) vSlice
        in V.zipWith (+) acc scaled

  return $ foldl' addScaled zero [0 .. sequenceLength - 1]

-- computeScores' : uses cacheIndex, same semantics as the original computeScores
computeScores
  :: NetworkConfig                 -- need the network to compute cacheIndex
  -> LayerIndex
  -> HeadIndex
  -> StepCount
  -> Int                     -- headDim
  -> V.Vector Float          -- frozen key cache
  -> V.Vector Float          -- query vector for this head
  -> V.Vector Float          -- raw scores (length = step + 1)
computeScores network layerIndex headIndex (StepCount step) headDim kVec qHead =
  let scaling = sqrt (fromIntegral headDim)
      sequenceLength = step + 1
      scoreForPos pos =
        let keyCacheOffset = cacheIndex network (StepCount pos) layerIndex headIndex 0
            kSlice         = V.slice keyCacheOffset headDim kVec
            dotProd        = V.sum (V.zipWith (*) qHead kSlice)
        in dotProd / scaling
  in V.generate sequenceLength scoreForPos

-- QKV projection
computeQKV :: TransformerParams -> StepCount -> LayerIndex -> TokenVector -> TransformerResult (V.Vector Float, V.Vector Float, V.Vector Float)
computeQKV params currentStep layerIndex (TokenVector inputToken) = do
  network <- ask
  let
      numHeads = numAttentionHeads network
      LayerIndex layerIdx = layerIndex
      normalizedInput = rmsNorm inputToken (getRow layerIdx (rmsAttWeight params))
      dec = decoder network
      rotaryEncoding = modelRotary dec
      
  -- Compute initial query, key, and value vectors
  let queryVec = matrixVectorMult (getArray2D layerIdx (wq params)) normalizedInput
      keyVec = matrixVectorMult (getArray2D layerIdx (wk params)) normalizedInput
      valueVec = matrixVectorMult (getArray2D layerIdx (wv params)) normalizedInput

  -- Apply rotary encoding to query and key vectors for each head
  (rotatedQuery, rotatedKey) <- foldM
    (\(qAcc, kAcc) headIndex -> do
        let headIdx = HeadIndex headIndex
        rotatedQ <- applyRotary rotaryEncoding currentStep headIdx qAcc
        rotatedK <- applyRotary rotaryEncoding currentStep headIdx kAcc
        return (rotatedQ, rotatedK))
    (queryVec, keyVec)
    [0 .. numHeads - 1]

  return (rotatedQuery, rotatedKey, valueVec)

-- classifier logits for a given token vector
transformerLogits :: V.Vector Float -> TransformerResult (V.Vector Float)
transformerLogits tokenVec = do
  network <- ask
  let model = params network
      vocab = tokenEmbeddingTable model
      tokenWithRms = rmsNorm tokenVec (rmsFinalWeight model)
      logits = V.generate (nrows vocab) $ \row ->
        let start = row * ncols vocab
            end = start + ncols vocab
            dot = sum [(items2D vocab V.! i) * (tokenWithRms V.! (i - start)) | i <- [start .. end - 1]]
         in dot
  return logits

-- Cache indexing helper
cacheIndex :: NetworkConfig -> StepCount -> LayerIndex -> HeadIndex -> Int -> Int
cacheIndex NetworkConfig {numAttentionHeads, seqLen, headDimension} (StepCount stepIndex) (LayerIndex layerIndex) (HeadIndex headIndex) dimensionIndex =
  (((layerIndex * numAttentionHeads + headIndex) * seqLen) + stepIndex) * headDimension + dimensionIndex
