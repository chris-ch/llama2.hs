module Architecture
  ( EmbeddingComponent (..),
    RotaryEncodingComponent (..),
    MultiHeadAttentionComponent (..),
    FeedForwardNetworkComponent (..),
    TransformerLayerComponent (..),
    TransformerDecoderComponent (..),
    SingleHeadComponent (..),
    transformerLogits, TransformerResult
    , NetworkConfig (..)
    , embed, runModel
    , DecoderCache (..), LayerAttentionCache (..), HeadCache(..)
  ) where

import Control.Monad.Reader (MonadIO (liftIO), MonadReader (ask), ReaderT)
import Control.Monad.State (StateT, get, put)
import Control.Monad ( forM_, forM)
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV

import Types
    (
      Array2D(items2D, ncols, nrows),
      Token (..),
      TokenVector(..),
      StepCount(..), getRow, MVectorFloat )
import Primitives
    (
      softmax, rmsNorm, sigmoidLinearUnit, matrixVectorMult )
import Data.List (foldl')

-- Runtime type alias
type TransformerResult a = ReaderT NetworkConfig (StateT DecoderCache IO) a

-- Data definitions mirroring architectural components
data EmbeddingComponent = EmbeddingComponent
  { vocabulary :: Array2D,
    rmsFinalWeight :: V.Vector Float
  } deriving (Show)

data RotaryEncodingComponent = RotaryEncodingComponent
  { freqCos :: Array2D,
    freqSin :: Array2D
  } deriving (Show)

data SingleHeadComponent = SingleHeadComponent
  { wqHead :: Array2D    -- size headDim x modelDim
  , wkHead :: Array2D
  , wvHead :: Array2D
  , rotary :: RotaryEncodingComponent
  } deriving (Show)

data MultiHeadAttentionComponent = MultiHeadAttentionComponent
  { heads    :: [SingleHeadComponent]
  , mWo      :: Array2D         -- Output projection matrix
  , rmsAtt   :: V.Vector Float  -- RMSNorm before QKV projection
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
    modelLayers :: [TransformerLayerComponent]
  } deriving (Show)

data HeadCache = HeadCache
  { headKeyCache   :: MVectorFloat  -- Size: seqLen * headDim
  , headValueCache :: MVectorFloat  -- Size: seqLen * headDim
  }

newtype LayerAttentionCache = LayerAttentionCache
  { headCaches :: [HeadCache]
  }

newtype DecoderCache = DecoderCaches
  { layerCaches :: [LayerAttentionCache]
  }

runModel :: TransformerDecoderComponent -> Token -> StepCount -> TransformerResult TokenVector
runModel dec tokenCode stepCount = do
  let embeddingLayer = modelEmbedding dec
      layers = modelLayers dec
  tokenVector <- embed embeddingLayer tokenCode
  decoderCaches <- get
  (finalTokenVector, finalCaches) <- runLayers layers (layerCaches decoderCaches) tokenVector stepCount
  put $ DecoderCaches { layerCaches = finalCaches }
  return finalTokenVector

runLayers :: [TransformerLayerComponent] -> [LayerAttentionCache] -> TokenVector -> StepCount -> TransformerResult (TokenVector, [LayerAttentionCache])
runLayers [] [] tv _ = return (tv, [])
runLayers (layer : restLayers) (layerCaches : restCaches) tv step = do
  let mha = multiHeadAttention layer
      ffn = feedforwardNetwork layer
  (tv', updatedLayerCaches) <- runLayer mha ffn layerCaches tv step
  (tv'', restUpdated) <- runLayers restLayers restCaches tv' step
  return (tv'', updatedLayerCaches : restUpdated)
runLayers (_ : _) [] _ _ = fail "Mismatch: Non-empty layers but empty caches"
runLayers [] (_ : _) _ _ = fail "Mismatch: Empty layers but non-empty caches"

runLayer :: MultiHeadAttentionComponent -> FeedForwardNetworkComponent -> LayerAttentionCache -> TokenVector -> StepCount -> TransformerResult (TokenVector, LayerAttentionCache)
runLayer mha ffn layerCaches (TokenVector inputToken) step = do
  network <- ask
  let rmsWeights = rmsAtt mha
      normalizedInput = rmsNorm inputToken rmsWeights
      outputProjectionWeights = mWo mha
      headDim = headDimension network

  -- Compute per-head QKV (rotated Q/K)
  qkvList <- forM (heads mha) $ \headComp -> do
    (q, k, v) <- runSingleHeadQKV headComp normalizedInput
    (q', k') <- applyRotaryToHead headComp step (q, k)
    return (q', k', v)

  let (qList, _, _) = unzip3 qkvList

  -- Update per-head caches
  let headCachesList = headCaches layerCaches
  forM_ (zip headCachesList qkvList) $ \(hc, (_, k', v')) -> do
    let (HeadCache kc vc) = hc
    updateCache kc step headDim k'
    updateCache vc step headDim v'
    return ()

  -- Compute per-head attention outputs
  headOutputs <- forM (zip qList headCachesList) $ \(qHead, hc) ->
    headAttentionPerHead hc qHead step headDim

  let multiHeadOut = V.concat [vec | TokenVector vec <- headOutputs]
      attentionDelta = matrixVectorMult outputProjectionWeights multiHeadOut
      tokenAfterAttention = V.zipWith (+) inputToken attentionDelta

  -- Apply FFN (stateless)
  ffnOut <- runFeedForward ffn tokenAfterAttention
  let finalToken = V.zipWith (+) tokenAfterAttention ffnOut

  return (TokenVector finalToken, layerCaches)  -- Caches modified in-place

embed :: EmbeddingComponent -> Token -> TransformerResult TokenVector
embed embedding (Token tokenCode) = do
  let
    vocab = vocabulary embedding
    rowStart = fromIntegral tokenCode * ncols vocab
    tokenVector = TokenVector $ V.slice rowStart (ncols vocab) (items2D vocab)
  return tokenVector

applyRotary :: RotaryEncodingComponent -> StepCount -> V.Vector Float -> TransformerResult (V.Vector Float)
applyRotary (RotaryEncodingComponent freqCos freqSin) (StepCount step) input = do
  network <- ask
  let
      cosFrequencies = getRow step freqCos
      sinFrequencies = getRow step freqSin
      headDim = headDimension network

  -- Apply rotation per head
  return $ applyRotaryPositionEncoding headDim cosFrequencies sinFrequencies input

applyRotaryPositionEncoding :: Int -> V.Vector Float -> V.Vector Float -> V.Vector Float -> V.Vector Float
applyRotaryPositionEncoding headDim cosFrequencies sinFrequencies input = let

  processedPairs = map (\pairIndex ->
    let realComponent = input V.! pairIndex
        imagComponent = input V.! (pairIndex + 1)
        cosValue = cosFrequencies V.! (pairIndex `div` 2)
        sinValue = sinFrequencies V.! (pairIndex `div` 2)
        rotatedReal = realComponent * cosValue - imagComponent * sinValue
        rotatedImag = realComponent * sinValue + imagComponent * cosValue
    in [(pairIndex, rotatedReal), (pairIndex + 1, rotatedImag)]
    ) [0, 2 .. headDim - 2]

  updates = concat processedPairs
  rotated = input V.// updates
  result = input V.// zip [0..headDim-1] (V.toList rotated)
  in result

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

-- Per-head cache update
updateCache :: MVectorFloat -> StepCount -> Int -> V.Vector Float -> TransformerResult ()
updateCache cache (StepCount step) headDim slice = do
  let offset = step * headDim
  hsm <- V.thaw slice
  liftIO $ MV.copy (MV.slice offset headDim cache) hsm

-- Per-head attention computation
headAttentionPerHead :: HeadCache -> V.Vector Float -> StepCount -> Int -> TransformerResult TokenVector
headAttentionPerHead (HeadCache kc vc) qHead (StepCount stepInt) headDim = do
  kVec <- liftIO $ V.unsafeFreeze kc
  vVec <- liftIO $ V.unsafeFreeze vc
  let scaling = sqrt (fromIntegral headDim :: Float)
      sequenceLength = stepInt + 1
      rawScores = V.generate sequenceLength $ \pos ->
        let offset = pos * headDim
            kSlice = V.slice offset headDim kVec
            dotProd = V.sum (V.zipWith (*) qHead kSlice)
        in dotProd / scaling
      softValues = softmax rawScores sequenceLength
      headScores = V.toList softValues
      zeroVec = V.replicate headDim 0.0
      addScaled acc pos =
        let offset = pos * headDim
            vSlice = V.slice offset headDim vVec
            score = headScores !! pos
            scaled = V.map (* score) vSlice
        in V.zipWith (+) acc scaled
      result = foldl' addScaled zeroVec [0 .. sequenceLength - 1]
  return $ TokenVector result

-- QKV per head
runSingleHeadQKV :: SingleHeadComponent -> V.Vector Float -> TransformerResult (V.Vector Float, V.Vector Float, V.Vector Float)
runSingleHeadQKV headComp normalizedInput = do
  let q = matrixVectorMult (wqHead headComp) normalizedInput
      k = matrixVectorMult (wkHead headComp) normalizedInput
      v = matrixVectorMult (wvHead headComp) normalizedInput
  return (q, k, v)

-- Rotary application
applyRotaryToHead :: SingleHeadComponent -> StepCount -> (V.Vector Float, V.Vector Float) -> TransformerResult (V.Vector Float, V.Vector Float)
applyRotaryToHead headComp step (q, k) = do
  let rot = rotary headComp
  q' <- applyRotary rot step q
  k' <- applyRotary rot step k
  return (q', k')

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
