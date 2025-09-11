module Architecture
  ( EmbeddingComponent (..),
    RotaryEncodingComponent (..),
    MultiHeadAttentionComponent (..),
    FeedForwardNetworkComponent (..),
    TransformerLayerComponent (..),
    TransformerDecoderComponent (..),
    transformerLogits, TransformerResult
    , NetworkConfig (..)
    , embed, runModel
  ) where

import Control.Monad.Reader (MonadIO (liftIO), MonadReader (ask), ReaderT)
import Control.Monad.State (StateT, gets)
import Control.Monad ( foldM, foldM, forM_)
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV

import Types
    (
      Array2D(items2D, ncols, nrows),
      AttentionKV(..),
      Token (..),
      TokenVector(..),
      StepCount(..),
      LayerIndex(..), HeadIndex (..), getRow, MVectorFloat )
import Primitives
    (
      softmax, rmsNorm, sigmoidLinearUnit, applyRotaryPositionEncoding, matrixVectorMult )
import Data.List (foldl')

-- Data definitions mirroring architectural components (static aspect)
data EmbeddingComponent = EmbeddingComponent
  { vocabulary :: Array2D,
    rmsFinalWeight :: V.Vector Float
  } deriving (Show)

data RotaryEncodingComponent = RotaryEncodingComponent
  { freqCos :: Array2D,
    freqSin :: Array2D
  } deriving (Show)

data MultiHeadAttentionComponent = MultiHeadAttentionComponent
  { mWq     :: Array2D            -- Q projection
  , mWk     :: Array2D            -- K projection
  , mWv     :: Array2D            -- V projection
  , mWo     :: Array2D            -- Output projection
  , mRMSAtt :: V.Vector Float     -- RMS normalization for input
  , mRotary  :: RotaryEncodingComponent     -- Rotary embeddings for attention
  } deriving (Show)

data FeedForwardNetworkComponent = FeedForwardNetworkComponent
  { fW1 :: Array2D,
    fW2 :: Array2D,
    fW3 :: Array2D,
    fRMSFfn :: V.Vector Float
  } deriving (Show)

data TransformerLayerComponent = TransformerLayerComponent
  {
    multiHeadAttention :: MultiHeadAttentionComponent,
    feedforwardNetwork :: FeedForwardNetworkComponent
  } deriving (Show)

data TransformerDecoderComponent = TransformerDecoderComponent
  { modelEmbedding :: EmbeddingComponent,
    modelRotary :: RotaryEncodingComponent,
    modelLayers :: [TransformerLayerComponent]
  } deriving (Show)

-- Type classes for components behaviour (dynamic aspect)
class Embedding e where
  embed :: e -> Token -> TransformerResult TokenVector

class RotaryEncoding r where
  applyRotary :: r -> StepCount -> HeadIndex -> V.Vector Float -> TransformerResult (V.Vector Float)

class FeedForwarding f where
  runFeedForward :: f -> V.Vector Float -> TransformerResult (V.Vector Float)

class Attending a where
  runAttention :: a -> LayerIndex -> TokenVector -> StepCount -> TransformerResult TokenVector
  computeQKV :: a -> StepCount -> TokenVector -> TransformerResult (TokenVector, TokenVector, TokenVector)

class TransformerProcessing m where
  runModel :: m -> Token -> StepCount -> TransformerResult TokenVector

-- Instancations for behaviour implementation
instance TransformerProcessing TransformerDecoderComponent where
  runModel :: TransformerDecoderComponent -> Token -> StepCount -> TransformerResult TokenVector
  runModel dec tokenCode stepCount = do
    let
      embeddingLayer = modelEmbedding dec
      layers = modelLayers dec
    tokenVector <- embed embeddingLayer tokenCode
    foldM
        (\accToken layerIndex -> do
          let layer = layers !! layerIndex
          let mha = multiHeadAttention layer
          runAttention mha (LayerIndex layerIndex) accToken stepCount
        )
        tokenVector
        [0 .. length layers - 1]

instance Attending MultiHeadAttentionComponent where
  
  runAttention :: MultiHeadAttentionComponent -> LayerIndex -> TokenVector -> StepCount -> TransformerResult TokenVector
  runAttention mha layerIndex inputToken currentStep = do
    network <- ask
    AttentionKV {keyCache, valueCache} <- gets id
    let dec = decoder network
        LayerIndex li = layerIndex
        layer = modelLayers dec !! li
        ffn = feedforwardNetwork layer
        numHeads = numAttentionHeads network
        headDim = headDimension network
        outputProjectionWeights = mWo mha

    (TokenVector queryOutput, TokenVector keyOutput, TokenVector valueOutput) <- computeQKV mha currentStep inputToken

    forM_ [0 .. numHeads - 1] $ \headIndex -> do
      -- Update cache with slices from keyOutput and valueOutput
      let keyHeadSlice = V.slice (headIndex * headDim) headDim keyOutput
          valueHeadSlice = V.slice (headIndex * headDim) headDim valueOutput
      updateCacheWithHead layerIndex (HeadIndex headIndex) currentStep keyHeadSlice keyCache
      updateCacheWithHead layerIndex (HeadIndex headIndex) currentStep valueHeadSlice valueCache

    -- Compute multi-head attention
    let headsQ = [ V.slice (i * headDim) headDim queryOutput | i <- [0 .. numHeads - 1] ]
    headOutputs <- mapM (\i -> headAttention layerIndex (HeadIndex i) currentStep (headsQ !! i)) [0 .. numHeads - 1]
    let multiHeadOut = V.concat headOutputs

    let attentionDelta = matrixVectorMult outputProjectionWeights multiHeadOut

    let TokenVector tokenVector = inputToken
        tokenAfterAttention = V.zipWith (+) tokenVector attentionDelta

    -- Apply FFN
    ffnOut <- runFeedForward ffn tokenAfterAttention
    return $ TokenVector $ V.zipWith (+) tokenAfterAttention ffnOut

  -- QKV projection
  computeQKV :: MultiHeadAttentionComponent -> StepCount -> TokenVector -> TransformerResult (TokenVector, TokenVector, TokenVector)
  computeQKV mha currentStep (TokenVector inputToken) = do
    network <- ask
    let
        numHeads = numAttentionHeads network
        normalizedInput = rmsNorm inputToken (mRMSAtt mha)
        rotaryEncoding = mRotary mha

    -- Compute initial query, key, and value vectors (unchanged from original)
    let queryVec = matrixVectorMult (mWq mha) normalizedInput
        keyVec = matrixVectorMult (mWk mha) normalizedInput
        valueVec = matrixVectorMult (mWv mha) normalizedInput

    -- Keep the original rotary encoding logic exactly as it was
    (rotatedQuery, rotatedKey) <- foldM
      (\(qAcc, kAcc) headIndex -> do
          let headIdx = HeadIndex headIndex
          rotatedQ <- applyRotary rotaryEncoding currentStep headIdx qAcc
          rotatedK <- applyRotary rotaryEncoding currentStep headIdx kAcc
          return (rotatedQ, rotatedK))
      (queryVec, keyVec)
      [0 .. numHeads - 1]

    return (TokenVector rotatedQuery, TokenVector rotatedKey, TokenVector valueVec)

instance Embedding EmbeddingComponent where
  embed :: EmbeddingComponent -> Token -> TransformerResult TokenVector
  embed embedding (Token tokenCode) = do
    let
      vocab = vocabulary embedding
      rowStart = fromIntegral tokenCode * ncols vocab
      tokenVector = TokenVector $ V.slice rowStart (ncols vocab) (items2D vocab)
    return tokenVector

instance RotaryEncoding RotaryEncodingComponent where
  applyRotary :: RotaryEncodingComponent -> StepCount -> HeadIndex -> V.Vector Float -> TransformerResult (V.Vector Float)
  applyRotary (RotaryEncodingComponent freqCos freqSin) (StepCount step) headIndex input = do
    network <- ask
    let
        cosFrequencies = getRow step freqCos
        sinFrequencies = getRow step freqSin
        headDim = headDimension network

    -- Apply rotation per head
    return $ applyRotaryPositionEncoding headDim headIndex cosFrequencies sinFrequencies input

instance FeedForwarding FeedForwardNetworkComponent where
  runFeedForward :: FeedForwardNetworkComponent -> V.Vector Float -> TransformerResult (V.Vector Float)
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

data NetworkConfig = NetworkConfig
  { modelDim :: Int,
    hiddenDim :: Int,
    numLayers :: Int,
    numAttentionHeads :: Int,
    numKeyValueHeads :: Int,
    vocabSize :: Int,
    seqLen :: Int,
    headDimension :: Int,
    decoder :: TransformerDecoderComponent
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

-- classifier logits for a given token vector
transformerLogits :: TransformerDecoderComponent -> V.Vector Float -> TransformerResult (V.Vector Float)
transformerLogits dec tokenVec = do
  let
      vocab = vocabulary (modelEmbedding dec)
      rmsWeight = rmsFinalWeight (modelEmbedding dec)
      tokenWithRms = rmsNorm tokenVec rmsWeight
      logits = V.generate (nrows vocab) $ \row ->
        let start = row * ncols vocab
            end = start + ncols vocab
            dot = sum [(items2D vocab V.! i) * (tokenWithRms V.! (i - start)) | i <- [start .. end - 1]]
         in dot
  return logits

-- Multi-head
updateCacheWithHead :: LayerIndex -> HeadIndex -> StepCount -> V.Vector Float -> MVectorFloat -> TransformerResult ()
updateCacheWithHead layerIndex headIndex stepIndex headSlice cache = do
  network <- ask
  let headDim = headDimension network
      cacheOffset = cacheIndex network stepIndex layerIndex headIndex 0
  hsm <- V.thaw headSlice
  MV.copy (MV.slice cacheOffset headDim cache) hsm

-- Cache indexing helper
cacheIndex :: NetworkConfig -> StepCount -> LayerIndex -> HeadIndex -> Int -> Int
cacheIndex NetworkConfig {numAttentionHeads, seqLen, headDimension} (StepCount stepIndex) (LayerIndex layerIndex) (HeadIndex headIndex) dimensionIndex =
  (((layerIndex * numAttentionHeads + headIndex) * seqLen) + stepIndex) * headDimension + dimensionIndex
