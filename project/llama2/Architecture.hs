module Architecture
  ( Embedding (..),
    RotaryEncoding (..),
    MultiHeadAttention (..),
    FeedForwardNetwork (..),
    TransformerLayer (..),
    TransformerDecoder (..),
    TransformerParams(..),
    transformerLogits, applyFeedForwardNetwork, computeQKV, computeMultiHeadAttention, TransformerResult
    , NetworkConfig (..)
    , embed
  ) where

import Control.Monad.Reader (MonadIO (liftIO), MonadReader (ask), ReaderT)
import Control.Monad.State (StateT, gets)
import Control.Monad ( forM_ )
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV


import Types
    ( getArray2D,
      Array2D(items2D, ncols, nrows),
      AttentionKV(..),
      Token,
      TokenVector(..),
      StepCount(..),
      LayerIndex(..), Array3D, MVectorFloat, HeadIndex (..), getRow )
import Primitives
    (
      applyMatrixVectorMult, dotProductMV, softmax, rmsNorm, silu, applyRotaryPositionEncoding )

-- Data definitions mirroring architecture boxes
newtype Embedding = Embedding
  { vocabulary :: Array2D
  } deriving (Show)

data RotaryEncoding = RotaryEncoding
  { freqCos :: Array2D,
    freqSin :: Array2D
  } deriving (Show)

data MultiHeadAttention = MultiHeadAttention
  { mWq :: Array2D,
    mWk :: Array2D,
    mWv :: Array2D,
    mWo :: Array2D,
    mRMSAtt :: V.Vector Float
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

class AttentionLayer a where
  runAttention :: a -> V.Vector Float -> StepCount -> TransformerResult (V.Vector Float)

class FeedForwardLayer f where
  runFeedForward :: f -> V.Vector Float -> TransformerResult (V.Vector Float)

class TransformerBlock l where
  runBlock :: l -> V.Vector Float -> StepCount -> TransformerResult (V.Vector Float)

class TransformerModel m where
  runModel :: m -> Token -> StepCount -> TransformerResult (V.Vector Float)

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

-- Attention accumulation
accumulateAttentionOutput :: LayerIndex -> HeadIndex -> [Float] -> MVectorFloat -> MVectorFloat -> TransformerResult ()
accumulateAttentionOutput layerIndex headIndex attentionScores valueCache outputBuffer = do
  network <- ask
  let attentionHeadDim = headDimension network
  MV.set outputBuffer 0.0
  let sequenceLength = length attentionScores
  forM_ [0 .. sequenceLength - 1] $ \positionIndex -> do
    let valueCacheOffset = cacheIndex network (StepCount positionIndex) layerIndex headIndex 0
        valueSlice = MV.slice valueCacheOffset attentionHeadDim valueCache
        attentionWeight = attentionScores !! positionIndex
    forM_ [0 .. attentionHeadDim - 1] $ \dimIndex -> do
      valueComponent <- MV.read valueSlice dimIndex
      MV.modify outputBuffer (+ (attentionWeight * valueComponent)) dimIndex

-- compute scores
computeScores :: LayerIndex -> HeadIndex -> MVectorFloat -> MVectorFloat -> StepCount -> TransformerResult (V.Vector Float)
computeScores layerIndex headIndex queryBuffer keyCache currentStep = do
  network <- ask
  let attentionHeadDim = headDimension network
      scalingFactor = sqrt (fromIntegral attentionHeadDim)
      StepCount step = currentStep
      sequenceLength = step + 1
  V.generateM sequenceLength $ \positionIndex -> do
    let keyCacheOffset = cacheIndex network (StepCount positionIndex) layerIndex headIndex 0
        keySlice = MV.slice keyCacheOffset attentionHeadDim keyCache
        HeadIndex hIndex = headIndex
        querySlice = MV.slice (hIndex * attentionHeadDim) attentionHeadDim queryBuffer
    dotProduct <- liftIO $ dotProductMV querySlice keySlice
    return $ dotProduct / scalingFactor

-- Multi-head
updateCacheWithHead :: LayerIndex -> HeadIndex -> StepCount -> MVectorFloat -> MVectorFloat -> TransformerResult ()
updateCacheWithHead layerIndex headIndex stepIndex headSlice cache = do
  network <- ask
  let headDim = headDimension network
      cacheOffset = cacheIndex network stepIndex layerIndex headIndex 0
  MV.copy (MV.slice cacheOffset headDim cache) headSlice

computeMultiHeadAttention :: StepCount -> LayerIndex -> MVectorFloat -> MVectorFloat -> MVectorFloat -> MVectorFloat -> TransformerResult ()
computeMultiHeadAttention currentStep layerIndex queryOutput keyCache valueCache multiHeadOutput = do
  network <- ask
  let attentionHeadDim = headDimension network
      numHeads = numAttentionHeads network
  forM_ [0 .. numHeads - 1] $ \headIndex -> do
    rawAttentionScores <- computeScores layerIndex (HeadIndex headIndex) queryOutput keyCache currentStep
    let normalizedScores = softmax rawAttentionScores (V.length rawAttentionScores)
        attentionWeights = V.toList normalizedScores
        outputOffset = headIndex * attentionHeadDim
        headOutputSlice = MV.slice outputOffset attentionHeadDim multiHeadOutput
    accumulateAttentionOutput layerIndex (HeadIndex headIndex) attentionWeights valueCache headOutputSlice

-- QKV projection
computeQKV :: TransformerParams -> StepCount -> LayerIndex -> TokenVector -> TransformerResult ()
computeQKV params currentStep layerIndex (TokenVector inputToken) = do
  network <- ask
  AttentionKV {queryOutput, keyOutput, valueOutput, keyCache, valueCache} <- gets id
  let headDim = headDimension network
      numHeads = numAttentionHeads network
      LayerIndex layerIdx = layerIndex
      normalizedInput = rmsNorm inputToken (getRow layerIdx (rmsAttWeight params))
      dec = decoder network
      rotary = modelRotary dec
      cosValues = freqCos rotary
      sinValues = freqSin rotary

  liftIO $ do
    normalizedInputMutable <- V.thaw normalizedInput
    applyMatrixVectorMult (getArray2D layerIdx (wq params)) normalizedInputMutable queryOutput
    applyMatrixVectorMult (getArray2D layerIdx (wk params)) normalizedInputMutable keyOutput
    applyMatrixVectorMult (getArray2D layerIdx (wv params)) normalizedInputMutable valueOutput
  
  let rotaryEncoding = RotaryEncoding { freqCos = cosValues, freqSin = sinValues }

  forM_ [0 .. numHeads - 1] $ \headIndex -> do
      -- Convert mutable queryOutput and keyOutput to immutable vectors
      queryInput <- liftIO $ V.freeze queryOutput
      keyInput <- liftIO $ V.freeze keyOutput

      -- Apply rotations to query and key
      rotatedQuery <- applyRotary rotaryEncoding currentStep (HeadIndex headIndex) queryInput
      rotatedKey   <- applyRotary rotaryEncoding currentStep (HeadIndex headIndex) keyInput


      -- Write the rotated results back to the mutable vectors
      liftIO $ do
        rotatedQueryMutable <- V.thaw rotatedQuery
        rotatedKeyMutable <- V.thaw rotatedKey
        MV.copy queryOutput rotatedQueryMutable
        MV.copy keyOutput rotatedKeyMutable

  -- We need all rotations applied before updating cache

  forM_ [0 .. numHeads - 1] $ \headIndex -> do
    -- Update cache with slices from keyOutput and valueOutput
    let keyHeadSlice = MV.slice (headIndex * headDim) headDim keyOutput
        valueHeadSlice = MV.slice (headIndex * headDim) headDim valueOutput
    updateCacheWithHead layerIndex (HeadIndex headIndex) currentStep keyHeadSlice keyCache
    updateCacheWithHead layerIndex (HeadIndex headIndex) currentStep valueHeadSlice valueCache

-- FFN
applyFeedForwardNetwork :: TransformerParams -> LayerIndex -> V.Vector Float -> TransformerResult ()
applyFeedForwardNetwork params (LayerIndex layerIndex) inputToken = do
  network <- ask
  AttentionKV {gateOutput, upProjectionOutput, feedforwardNetworkOutput} <- gets id
  let hiddenFFNDim = hiddenDim network
      rmsFFNWeights = getRow layerIndex $ rmsFfnWeight params
      normalizedInput = rmsNorm inputToken rmsFFNWeights

  liftIO $ do
    normalizedInputMutable <- V.thaw normalizedInput

    applyMatrixVectorMult (getArray2D layerIndex (w1 params)) normalizedInputMutable gateOutput

    forM_ [0 .. hiddenFFNDim - 1] $ \i -> MV.unsafeModify gateOutput silu i

    applyMatrixVectorMult (getArray2D layerIndex (w3 params)) normalizedInputMutable upProjectionOutput

    forM_ [0 .. hiddenFFNDim - 1] $ \i -> do
      upValue <- MV.unsafeRead upProjectionOutput i
      MV.unsafeModify gateOutput (* upValue) i

    applyMatrixVectorMult (getArray2D layerIndex (w2 params)) gateOutput feedforwardNetworkOutput

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
