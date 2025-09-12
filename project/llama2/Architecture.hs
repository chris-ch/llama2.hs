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
  ) where

import Control.Monad.Reader (MonadIO (liftIO), MonadReader (ask), ReaderT)
import Control.Monad.State (StateT, gets)
import Control.Monad ( foldM, foldM, forM_, forM)
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
      softmax, rmsNorm, sigmoidLinearUnit, matrixVectorMult )
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

runAttention :: MultiHeadAttentionComponent -> LayerIndex -> TokenVector -> StepCount -> TransformerResult TokenVector
runAttention mha layerIndex inputToken currentStep = do
  network <- ask
  let dec = decoder network
      LayerIndex li = layerIndex
      layer = modelLayers dec !! li
      ffn = feedforwardNetwork layer
      outputProjectionWeights = mWo mha
      numHeads = numAttentionHeads network

  (TokenVector queryOutput, TokenVector keyOutput, TokenVector valueOutput) <- computeQKV mha currentStep inputToken

  forM_ [0 .. numHeads - 1] $ \headIndex -> do
    applyRotaryToQK layerIndex (HeadIndex headIndex) (TokenVector keyOutput) (TokenVector valueOutput) currentStep

  headOutputs <- mapM (\i -> attentionHeads layerIndex (HeadIndex i) (TokenVector queryOutput) currentStep) [0 .. numHeads - 1]

  multiHeadOut <- combineHeads headOutputs

  let attentionDelta = matrixVectorMult outputProjectionWeights multiHeadOut

  let TokenVector tokenVector = inputToken
      tokenAfterAttention = V.zipWith (+) tokenVector attentionDelta

  -- Apply FFN
  ffnOut <- runFeedForward ffn tokenAfterAttention
  return $ TokenVector $ V.zipWith (+) tokenAfterAttention ffnOut

-- QKV projection
computeQKV :: MultiHeadAttentionComponent
  -> StepCount
  -> TokenVector
  -> TransformerResult (TokenVector, TokenVector, TokenVector)
computeQKV mha step (TokenVector inputToken) = do
  let rmsWeights = rmsAtt mha
      normalizedInput = rmsNorm inputToken rmsWeights
  qHeads <- forM (heads mha) $ \headComp -> do
    (q,k,v) <- runSingleHeadQKV headComp normalizedInput
    (q',k') <- applyRotaryToHead headComp step (q,k)
    return (q',k',v)

  -- split out q, k, v from list of tuples
  let (qList, kList, vList) = unzip3 qHeads
  return (TokenVector (V.concat qList),
          TokenVector (V.concat kList),
          TokenVector (V.concat vList))

applyRotaryToQK :: LayerIndex -> HeadIndex -> TokenVector -> TokenVector -> StepCount -> TransformerResult ()
applyRotaryToQK layerIndex (HeadIndex headIndex) (TokenVector keyOutput) (TokenVector valueOutput) currentStep = do
  network <- ask
  AttentionKV {keyCache, valueCache} <- gets id
  let
      headDim = headDimension network
      
  -- Update cache with slices from keyOutput and valueOutput
  let keyHeadSlice = V.slice (headIndex * headDim) headDim keyOutput
      valueHeadSlice = V.slice (headIndex * headDim) headDim valueOutput
  updateCacheWithHead layerIndex (HeadIndex headIndex) currentStep keyHeadSlice keyCache
  updateCacheWithHead layerIndex (HeadIndex headIndex) currentStep valueHeadSlice valueCache

attentionHeads :: LayerIndex -> HeadIndex -> TokenVector -> StepCount -> TransformerResult TokenVector
attentionHeads layerIndex (HeadIndex headIndex) (TokenVector queryOutput) currentStep = do
  network <- ask
  let
      numHeads = numAttentionHeads network
      headDim = headDimension network

  -- Compute multi-head attention
  let headsQ = [ V.slice (i * headDim) headDim queryOutput | i <- [0 .. numHeads - 1] ]
  headAttention layerIndex (HeadIndex headIndex) currentStep (headsQ !! headIndex)

combineHeads :: [TokenVector] -> TransformerResult (V.Vector Float)
combineHeads headOutputs =  do
  return $ V.concat headOutputsVecs where
      headOutputsVecs = [v | TokenVector v <- headOutputs]

runSingleHeadQKV :: SingleHeadComponent
  -> V.Vector Float  -- normalized input (modelDim)
  -> TransformerResult (V.Vector Float, V.Vector Float, V.Vector Float)
runSingleHeadQKV headComp normalizedInput = do
  let q = matrixVectorMult (wqHead headComp) normalizedInput
      k = matrixVectorMult (wkHead headComp) normalizedInput
      v = matrixVectorMult (wvHead headComp) normalizedInput
  return (q, k, v)

applyRotaryToHead :: SingleHeadComponent
  -> StepCount
  -> (V.Vector Float, V.Vector Float)
  -> TransformerResult (V.Vector Float, V.Vector Float)
applyRotaryToHead headComp step (q, k) = do
  let rot = rotary headComp
  q' <- applyRotary rot step (HeadIndex 0) q
  k' <- applyRotary rot step (HeadIndex 0) k
  return (q',k')

embed :: EmbeddingComponent -> Token -> TransformerResult TokenVector
embed embedding (Token tokenCode) = do
  let
    vocab = vocabulary embedding
    rowStart = fromIntegral tokenCode * ncols vocab
    tokenVector = TokenVector $ V.slice rowStart (ncols vocab) (items2D vocab)
  return tokenVector

applyRotary :: RotaryEncodingComponent -> StepCount -> HeadIndex -> V.Vector Float -> TransformerResult (V.Vector Float)
applyRotary (RotaryEncodingComponent freqCos freqSin) (StepCount step) headIndex input = do
  network <- ask
  let
      cosFrequencies = getRow step freqCos
      sinFrequencies = getRow step freqSin
      headDim = headDimension network

  -- Apply rotation per head
  return $ applyRotaryPositionEncoding headDim headIndex cosFrequencies sinFrequencies input

applyRotaryPositionEncoding :: Int -> HeadIndex -> V.Vector Float -> V.Vector Float -> V.Vector Float -> V.Vector Float
applyRotaryPositionEncoding headDim (HeadIndex headIndex) cosFrequencies sinFrequencies input = let
  baseIndex = headIndex * headDim
  slice = V.slice baseIndex headDim input

  -- Process pairs just like the primed version
  processedPairs = map (\pairIndex ->
    let realComponent = slice V.! pairIndex
        imagComponent = slice V.! (pairIndex + 1)
        cosValue = cosFrequencies V.! (pairIndex `div` 2)
        sinValue = sinFrequencies V.! (pairIndex `div` 2)
        rotatedReal = realComponent * cosValue - imagComponent * sinValue
        rotatedImag = realComponent * sinValue + imagComponent * cosValue
    in [(pairIndex, rotatedReal), (pairIndex + 1, rotatedImag)]
    ) [0, 2 .. headDim - 2]

  updates = concat processedPairs
  rotated = slice V.// updates
  result = input V.// zip [baseIndex..baseIndex+headDim-1] (V.toList rotated)
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

-- Runtime type alias
type TransformerResult a = ReaderT NetworkConfig (StateT AttentionKV IO) a

headAttention
  :: LayerIndex
  -> HeadIndex
  -> StepCount
  -> V.Vector Float
  -> TransformerResult TokenVector
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

  return $ TokenVector $ foldl' addScaled zero [0 .. sequenceLength - 1]

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
