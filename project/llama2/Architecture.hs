module Architecture
  ( Embedding (..),
    RotaryEncoding (..),
    MultiHeadAttention (..),
    FeedForward (..),
    TransformerLayer (..),
    TransformerArchitecture (..),
    Embeddable (..),
    Positional (..),
    FeedForwardNet (..),
    LayerComp (..),
    Model (..),
    TransformerParams(..),
    runTransformerModel,
    transformerLogits, getTokenEmbedding, applyFeedForwardNetwork, computeQKV, computeMultiHeadAttention, TransformerResult
    , NetworkConfig (..)
  ) where

import Control.Monad.Reader (MonadIO (liftIO), MonadReader (ask), ReaderT)
import Control.Monad.State (StateT, gets)
import Control.Monad ( foldM, forM_ )
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV


import Types
    ( getArray2D,
      getRow,
      Array2D(items2D, ncols, nrows),
      AttentionKV(..),
      Token,
      TokenVector(..),
      StepCount(..),
      LayerIndex(..), Array3D, MVectorFloat, HeadIndex (..) )
import Primitives
    (
      applyMatrixVectorMult, dotProductMV, softmax, rmsNorm, silu )

-- Data definitions mirroring architecture boxes
newtype Embedding = Embedding
  { embMatrix :: Array2D
  }

data RotaryEncoding = RotaryEncoding
  { freqCos :: Array2D,
    freqSin :: Array2D
  }

data MultiHeadAttention = MultiHeadAttention
  { m_wq :: Array2D,
    m_wk :: Array2D,
    m_wv :: Array2D,
    m_wo :: Array2D,
    m_rmsAtt :: Array2D -- kept as placeholder; actual rms is per-row
  }

data FeedForward = FeedForward
  { f_w1 :: Array2D,
    f_w2 :: Array2D,
    f_w3 :: Array2D,
    f_rmsFfn :: Array2D
  }

data TransformerLayer = TransformerLayer
  { layerIndex :: LayerIndex,
    layerMha :: MultiHeadAttention,
    layerFfn :: FeedForward
  }

data TransformerArchitecture = TransformerArchitecture
  { modelEmbedding :: Embedding,
    modelRotary :: RotaryEncoding,
    modelLayers :: [TransformerLayer]
  }

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

-- Behavioural interfaces (typeclasses)
class Embeddable e where
  embedToken :: e -> Token -> TransformerResult TokenVector

class Positional p where
  applyPosition :: p -> StepCount -> TokenVector -> TransformerResult TokenVector

class FeedForwardNet f where
  runFFNComp :: f -> LayerIndex -> TokenVector -> TransformerResult TokenVector

class LayerComp l where
  runLayerComp :: l -> StepCount -> TokenVector -> TransformerResult TokenVector

class Model m where
  runTransformer :: m -> StepCount -> Token -> TransformerResult (V.Vector Float)

-- Instances wiring to primitives
instance Embeddable Embedding where
  embedToken :: Embedding -> Token -> TransformerResult TokenVector
  embedToken Embedding {embMatrix} token = do
    let vocab = embMatrix
        rowStart = fromIntegral token * ncols vocab
        tok = TokenVector $ V.slice rowStart (ncols vocab) (items2D vocab)
    return tok

instance Positional RotaryEncoding where
  applyPosition :: RotaryEncoding -> StepCount -> TokenVector -> TransformerResult TokenVector
  applyPosition _ (StepCount _) = return

instance FeedForwardNet FeedForward where
  runFFNComp :: FeedForward -> LayerIndex -> TokenVector -> TransformerResult TokenVector
  runFFNComp FeedForward {} layerIdx (TokenVector inputVec) = do
    -- wrap primitive applyFeedForwardNetwork
    network <- ask
    let model = params network
    applyFeedForwardNetwork model layerIdx inputVec
    st <- gets id
    outVec <- liftIO $ V.freeze (feedforwardNetworkOutput st)
    return $ TokenVector outVec

instance LayerComp TransformerLayer where
  runLayerComp :: TransformerLayer -> StepCount -> TokenVector -> TransformerResult TokenVector
  runLayerComp TransformerLayer {layerIndex} stepCount tokenVec = do
    network <- ask
    let model = params network
        li = layerIndex
        LayerIndex li' = li
        freqCosValues = getRow (let StepCount s = stepCount in s) (freqCisReal model)
        freqSinValues = getRow (let StepCount s = stepCount in s) (freqCisImag model)

    -- QKV + cache update
    computeQKV model stepCount li freqCosValues freqSinValues tokenVec

    -- Run multihead attention
    AttentionKV {multiHeadOutput, projectedAttentionOutput, keyCache = kc, valueCache = vc} <- gets id

    computeMultiHeadAttention stepCount li multiHeadOutput kc vc multiHeadOutput

    -- Output projection W_o
    liftIO $ applyMatrixVectorMult (getArray2D li' (wo model)) multiHeadOutput projectedAttentionOutput

    attentionDelta <- liftIO $ V.freeze projectedAttentionOutput
    let TokenVector token = tokenVec
        tokenAfterAttention = V.zipWith (+) token attentionDelta

    -- FFN
    applyFeedForwardNetwork model li tokenAfterAttention
    st <- gets id
    ffnOut <- liftIO $ V.freeze (feedforwardNetworkOutput st)
    return $ TokenVector $ V.zipWith (+) tokenAfterAttention ffnOut

-- Top-level model runner
instance Model TransformerArchitecture where
  runTransformer :: TransformerArchitecture -> StepCount -> Token -> TransformerResult (V.Vector Float)
  runTransformer TransformerArchitecture {modelEmbedding, modelLayers} stepCount tokenCode = do
    TokenVector vec <- embedToken modelEmbedding tokenCode
    finalToken <- foldM (\acc layer -> runLayerComp layer stepCount acc) (TokenVector vec) modelLayers
    let TokenVector finalVec = finalToken
    transformerLogits finalVec

runTransformerModel :: TransformerArchitecture -> StepCount -> Token -> TransformerResult (V.Vector Float)
runTransformerModel = runTransformer

data NetworkConfig = NetworkConfig
  { modelDim :: Int,
    hiddenDim :: Int,
    numLayers :: Int,
    numAttentionHeads :: Int,
    numKeyValueHeads :: Int,
    vocabSize :: Int,
    seqLen :: Int,
    headDimension :: Int,
    params :: TransformerParams
  }
  deriving (Show)

-- Runtime type alias
type TransformerResult a = ReaderT NetworkConfig (StateT AttentionKV IO) a

applyRotations :: V.Vector Float -> V.Vector Float -> MVectorFloat -> TransformerResult ()
applyRotations cosFrequencies sinFrequencies buffer = do
  network <- ask
  let headDim = headDimension network
      numHeads = numAttentionHeads network
  forM_ [0 .. numHeads - 1] $ \headIndex -> do
    applyRotaryPositionEncoding (headIndex * headDim) cosFrequencies sinFrequencies buffer

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
computeQKV :: TransformerParams -> StepCount -> LayerIndex -> V.Vector Float -> V.Vector Float -> TokenVector -> TransformerResult ()
computeQKV weights currentStep layerIndex freqCosValues freqSinValues (TokenVector inputToken) = do
  network <- ask
  AttentionKV {queryOutput, keyOutput, valueOutput, keyCache, valueCache} <- gets id
  let headDim = headDimension network
      numHeads = numAttentionHeads network
      LayerIndex layerIdx = layerIndex
      normalizedInput = rmsNorm inputToken (getRow layerIdx (rmsAttWeight weights))

  liftIO $ do
    normalizedInputMutable <- V.thaw normalizedInput
    applyMatrixVectorMult (getArray2D layerIdx (wq weights)) normalizedInputMutable queryOutput
    applyMatrixVectorMult (getArray2D layerIdx (wk weights)) normalizedInputMutable keyOutput
    applyMatrixVectorMult (getArray2D layerIdx (wv weights)) normalizedInputMutable valueOutput

  applyRotations freqCosValues freqSinValues queryOutput
  applyRotations freqCosValues freqSinValues keyOutput

  forM_ [0 .. numHeads - 1] $ \headIndex -> do
    let keyHeadSlice = MV.slice (headIndex * headDim) headDim keyOutput
        valueHeadSlice = MV.slice (headIndex * headDim) headDim valueOutput
    updateCacheWithHead layerIndex (HeadIndex headIndex) currentStep keyHeadSlice keyCache
    updateCacheWithHead layerIndex (HeadIndex headIndex) currentStep valueHeadSlice valueCache

-- FFN
applyFeedForwardNetwork :: TransformerParams -> LayerIndex -> V.Vector Float -> TransformerResult ()
applyFeedForwardNetwork weights (LayerIndex layerIndex) inputToken = do
  network <- ask
  AttentionKV {gateOutput, upProjectionOutput, feedforwardNetworkOutput} <- gets id
  let hiddenFFNDim = hiddenDim network
      rmsFFNWeights = getRow layerIndex $ rmsFfnWeight weights
      normalizedInput = rmsNorm inputToken rmsFFNWeights

  liftIO $ do
    normalizedInputMutable <- V.thaw normalizedInput

    applyMatrixVectorMult (getArray2D layerIndex (w1 weights)) normalizedInputMutable gateOutput

    forM_ [0 .. hiddenFFNDim - 1] $ \i -> MV.unsafeModify gateOutput silu i

    applyMatrixVectorMult (getArray2D layerIndex (w3 weights)) normalizedInputMutable upProjectionOutput

    forM_ [0 .. hiddenFFNDim - 1] $ \i -> do
      upValue <- MV.unsafeRead upProjectionOutput i
      MV.unsafeModify gateOutput (* upValue) i

    applyMatrixVectorMult (getArray2D layerIndex (w2 weights)) gateOutput feedforwardNetworkOutput

-- Embedding helper
getTokenEmbedding :: Token -> TransformerResult TokenVector
getTokenEmbedding tokenCode = do
  network <- ask
  let model = params network
      vocab = tokenEmbeddingTable model
      rowStart = fromIntegral tokenCode * ncols vocab
      tokenVector = TokenVector $ V.slice rowStart (ncols vocab) (items2D vocab)
  return tokenVector

-- classifier logits for a given token vector
transformerLogits :: V.Vector Float -> TransformerResult (V.Vector Float)
transformerLogits tokenVec = do
  network <- ask
  let weights = params network
      vocab = tokenEmbeddingTable weights
      tokenWithRms = rmsNorm tokenVec (rmsFinalWeight weights)
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

-- Rotary primitives
applyRotaryPositionEncoding :: Int -> V.Vector Float -> V.Vector Float -> MVectorFloat -> TransformerResult ()
applyRotaryPositionEncoding startOffset cosFrequencies sinFrequencies buffer = do
  let headDim = V.length cosFrequencies * 2
  forM_ [0, 2 .. headDim - 2] $ \pairIndex -> do
    let realIndex = startOffset + pairIndex
    realComponent <- MV.read buffer realIndex
    imagComponent <- MV.read buffer (realIndex + 1)
    let cosValue = cosFrequencies V.! (pairIndex `div` 2)
        sinValue = sinFrequencies V.! (pairIndex `div` 2)
        rotatedReal = realComponent * cosValue - imagComponent * sinValue
        rotatedImag = realComponent * sinValue + imagComponent * cosValue
    MV.write buffer realIndex rotatedReal
    MV.write buffer (realIndex + 1) rotatedImag
