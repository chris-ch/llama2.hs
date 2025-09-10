module Primitives
  ( TransformerResult,
    applyMatrixVectorMult,
    rmsNorm,
    silu,
    dotProductMV,
    softmax,
    drawSample,
    cacheIndex,
    applyRotaryPositionEncoding,
    applyRotations,
    accumulateAttentionOutput,
    computeScores,
    computeMultiHeadAttention,
    updateCacheWithHead,
    computeQKV,
    applyFeedForwardNetwork,
    getTokenEmbedding,
    transformerLogits
  ) where

import Control.Monad (foldM, forM_)
import Control.Monad.Reader (MonadIO (liftIO), MonadReader (ask), ReaderT)
import Control.Monad.State (StateT, gets)
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV
import qualified System.Random as R
import Types
  ( NetworkConfig (..),
    AttentionKV (..),
    StepCount (..),
    LayerIndex (..),
    HeadIndex (..),
    Token,
    MVectorFloat,
    Array2D (..),
    TransformerModel (..),
    TokenVector (..),
    getArray2D,
    getRow
  )

-- Runtime type alias
type TransformerResult a = ReaderT NetworkConfig (StateT AttentionKV IO) a

-- Cache indexing helper
cacheIndex :: NetworkConfig -> StepCount -> LayerIndex -> HeadIndex -> Int -> Int
cacheIndex NetworkConfig {numAttentionHeads, seqLen, headDimension} (StepCount stepIndex) (LayerIndex layerIndex) (HeadIndex headIndex) dimensionIndex =
  (((layerIndex * numAttentionHeads + headIndex) * seqLen) + stepIndex) * headDimension + dimensionIndex

-- Linear algebra helpers
applyMatrixVectorMult :: Array2D -> MVectorFloat -> MVectorFloat -> IO ()
applyMatrixVectorMult array2D vecM result = do
  let nrows' = nrows array2D
      ncols' = ncols array2D
      items = items2D array2D
  forM_ [0 .. nrows' - 1] $ \row -> do
    let rowStart = row * ncols'
        loop !i !acc
          | i >= ncols' = return acc
          | otherwise = do
            v <- MV.unsafeRead vecM i
            let m = V.unsafeIndex items (rowStart + i)
            loop (i + 1) (acc + v * m)
    s <- loop 0 0.0
    MV.unsafeWrite result row s

-- dot product on mutable vectors
dotProductMV :: MVectorFloat -> MVectorFloat -> IO Float
dotProductMV vec1 vec2 = do
  let len = min (MV.length vec1) (MV.length vec2)
  foldM
    ( \acc i -> do
        a <- MV.read vec1 i
        b <- MV.read vec2 i
        return $ acc + a * b
    )
    0.0
    [0 .. len - 1]

-- RMS Norm
rmsNorm :: V.Vector Float -> V.Vector Float -> V.Vector Float
rmsNorm vector weights =
  let squareNorm = V.foldl' (\acc v -> acc + v * v) 0.0 vector
      ss = (squareNorm / fromIntegral (V.length vector)) + 1e-5
      normalized = V.map (* (1.0 / sqrt ss)) vector
   in V.zipWith (*) weights normalized

-- Activation
silu :: Float -> Float
silu x = x / (1.0 + exp (-x))

-- Softmax (applies to prefix)
softmax :: V.Vector Float -> Int -> V.Vector Float
softmax values size = V.concat [softmaxValues, V.slice size (V.length values - size) values]
  where
    maxVal = V.maximum (V.take size values)
    expValues = V.map (\x -> exp (x - maxVal)) (V.take size values)
    sumExpValues = V.sum expValues
    softmaxValues = V.map (/ sumExpValues) expValues

-- Sampling
drawSample :: Int -> V.Vector Float -> IO Token
drawSample randomSeed probabilities = do
  let gen = R.mkStdGen randomSeed
      (randomValue, _) = R.random gen :: (Float, R.StdGen)
      cumulativeDistribution = V.scanl1 (+) probabilities
      selectedIndex = V.length (V.takeWhile (< randomValue) cumulativeDistribution)
  return $ fromIntegral (min selectedIndex (V.length probabilities - 1))

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
computeQKV :: TransformerModel -> StepCount -> LayerIndex -> V.Vector Float -> V.Vector Float -> TokenVector -> TransformerResult ()
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
applyFeedForwardNetwork :: TransformerModel -> LayerIndex -> V.Vector Float -> TransformerResult ()
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
  let weights = model network
      vocab = tokenEmbeddingTable weights
      rowStart = fromIntegral tokenCode * ncols vocab
      tokenVector = TokenVector $ V.slice rowStart (ncols vocab) (items2D vocab)
  return tokenVector

-- classifier logits for a given token vector
transformerLogits :: V.Vector Float -> TransformerResult (V.Vector Float)
transformerLogits tokenVec = do
  network <- ask
  let weights = model network
      vocab = tokenEmbeddingTable weights
      tokenWithRms = rmsNorm tokenVec (rmsFinalWeight weights)
      logits = V.generate (nrows vocab) $ \row ->
        let start = row * ncols vocab
            end = start + ncols vocab
            dot = sum [(items2D vocab V.! i) * (tokenWithRms V.! (i - start)) | i <- [start .. end - 1]]
         in dot
  return logits
