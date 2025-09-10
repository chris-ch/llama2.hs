module Transformer (generateTokens) where

import Control.Monad (foldM, forM_)
import Control.Monad.Reader (MonadIO (liftIO), MonadReader (ask), ReaderT)
import Control.Monad.State (StateT, gets)
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV
import GHC.Unicode (isSpace)
import System.IO (hFlush, stdout)
import qualified System.Random as R
import Types (NetworkConfig (..), AttentionKV (..), StepCount (..), LayerIndex(..), HeadIndex(..), Token, MVectorFloat, Array2D(..), TransformerWeighting (..), TokenVector(..), PromptTokens, Vocabulary, getArray2D, getRow)

--------------------------------------------------------------------------------
-- Transformer runtime
--------------------------------------------------------------------------------

type TransformerResult a = ReaderT NetworkConfig (StateT AttentionKV IO) a

-- Cache indexing helper
cacheIndex :: NetworkConfig -> StepCount -> LayerIndex -> HeadIndex -> Int -> Int
cacheIndex NetworkConfig {numAttentionHeads, seqLen, headDimension} (StepCount stepIndex) (LayerIndex layerIndex) (HeadIndex headIndex) dimensionIndex =
  (((layerIndex * numAttentionHeads + headIndex) * seqLen) + stepIndex) * headDimension + dimensionIndex

-- Softmax
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

-- Helper: write single head vector from mutable slice to the flat cache
updateCacheWithHead :: LayerIndex -> HeadIndex -> StepCount -> MVectorFloat -> MVectorFloat -> TransformerResult ()
updateCacheWithHead layerIndex headIndex stepIndex headSlice cache = do
  network <- ask
  let headDim = headDimension network
  let cacheOffset = cacheIndex network stepIndex layerIndex headIndex 0
  MV.copy (MV.slice cacheOffset headDim cache) headSlice

-- Rotary application to mutable buffer slice
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

-- Apply rotations to entire buffer (per head)
applyRotations :: V.Vector Float -> V.Vector Float -> MVectorFloat -> TransformerResult ()
applyRotations cosFrequencies sinFrequencies buffer = do
  network <- ask
  let headDim = headDimension network
      numHeads = numAttentionHeads network
  forM_ [0 .. numHeads - 1] $ \headIndex -> do
    applyRotaryPositionEncoding (headIndex * headDim) cosFrequencies sinFrequencies buffer

-- Build activation directly into a provided mutable vector
accumulateAttentionOutput ::
  LayerIndex ->
  HeadIndex ->
  -- | Attention scores
  [Float] ->
  -- | Value cache
  MVectorFloat ->
  -- | Output buffer (must be preallocated to headDim)
  MVectorFloat ->
  TransformerResult ()
accumulateAttentionOutput layerIndex headIndex attentionScores valueCache outputBuffer = do
  network <- ask
  let attentionHeadDim = headDimension network
  -- Zero the output buffer
  MV.set outputBuffer 0.0
  let sequenceLength = length attentionScores
  forM_ [0 .. sequenceLength - 1] $ \positionIndex -> do
    let valueCacheOffset = cacheIndex network (StepCount positionIndex) layerIndex headIndex 0
    let valueSlice = MV.slice valueCacheOffset attentionHeadDim valueCache
    let attentionWeight = attentionScores !! positionIndex
    forM_ [0 .. attentionHeadDim - 1] $ \dimIndex -> do
      valueComponent <- MV.read valueSlice dimIndex
      MV.modify outputBuffer (+ (attentionWeight * valueComponent)) dimIndex

-- Attention scores
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

-- Dot product on mutable vectors
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

-- Multihead
computeMultiHeadAttention ::
  StepCount ->
  LayerIndex ->
  -- | Query buffer
  MVectorFloat ->
  -- | Key cache
  MVectorFloat ->
  -- | Value cache
  MVectorFloat ->
  -- | Output buffer (concatenated heads)
  MVectorFloat ->
  TransformerResult ()
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

    -- Write directly to the appropriate slice of the output buffer
    accumulateAttentionOutput layerIndex (HeadIndex headIndex) attentionWeights valueCache headOutputSlice

-- Math utils

-- Multiply a matrix by a mutable input vector, writing into an output mutable vector
applyMatrixVectorMult :: Array2D -> MVectorFloat -> MVectorFloat -> IO ()
applyMatrixVectorMult array2D vecM result = do
  -- For each row
  let nrows' = nrows array2D
      ncols' = ncols array2D
      items = items2D array2D
  forM_ [0 .. nrows' - 1] $ \row -> do
    let rowStart = row * ncols'
    -- accumulate dot product in a strict loop
    let loop !i !acc
          | i >= ncols' = return acc
          | otherwise = do
              v <- MV.unsafeRead vecM i
              let m = V.unsafeIndex items (rowStart + i)
              loop (i + 1) (acc + v * m)
    s <- loop 0 0.0
    MV.unsafeWrite result row s

rmsNorm :: V.Vector Float -> V.Vector Float -> V.Vector Float
rmsNorm vector weights =
  let squareNorm = V.foldl' (\acc v -> acc + v * v) 0.0 vector
      ss = (squareNorm / fromIntegral (V.length vector)) + 1e-5
      normalized = V.map (* (1.0 / sqrt ss)) vector
   in V.zipWith (*) weights normalized

-- FFN
applyFeedForwardNetwork :: TransformerWeighting -> LayerIndex -> V.Vector Float -> TransformerResult ()
applyFeedForwardNetwork weights (LayerIndex layerIndex) inputToken = do
  network <- ask
  AttentionKV {gateOutput, upProjectionOutput, feedforwardNetworkOutput} <- gets id
  let hiddenFFNDim = hiddenDim network
      silu activation = activation / (1.0 + exp (-activation))
      rmsFFNWeights = getRow layerIndex $ rmsFfnWeight weights
      normalizedInput = rmsNorm inputToken rmsFFNWeights

  liftIO $ do
    -- thaw normalizedInput once (normalizedInput is a V.Vector), reuse it
    normalizedInputMutable <- liftIO $ V.thaw normalizedInput

    applyMatrixVectorMult (getArray2D layerIndex (w1 weights)) normalizedInputMutable gateOutput

    -- gateActivation = silu(W1 * normalizedInput)
    forM_ [0 .. hiddenFFNDim - 1] $ \i -> MV.unsafeModify gateOutput silu i

    -- upProjection = W3 * normalizedInput  -> upProjectionBuffer
    applyMatrixVectorMult (getArray2D layerIndex (w3 weights)) normalizedInputMutable upProjectionOutput

    -- gateActivation *= upProjection   (in-place on gateBuffer)
    forM_ [0 .. hiddenFFNDim - 1] $ \i -> do
      upValue <- MV.unsafeRead upProjectionOutput i
      MV.unsafeModify gateOutput (* upValue) i

    -- result = W2 * gateActivation
    applyMatrixVectorMult (getArray2D layerIndex (w2 weights)) gateOutput feedforwardNetworkOutput

-- QKV
computeQKV :: TransformerWeighting -> StepCount -> LayerIndex -> V.Vector Float -> V.Vector Float -> TokenVector -> TransformerResult ()
computeQKV weights currentStep layerIndex freqCosValues freqSinValues (TokenVector inputToken) = do
  network <- ask
  AttentionKV {queryOutput, keyOutput, valueOutput, keyCache, valueCache} <- gets id
  let headDim = headDimension network
      numHeads = numAttentionHeads network
      LayerIndex layerIdx = layerIndex
      -- The input token gets RMS normalized
      normalizedInput = rmsNorm inputToken (getRow layerIdx (rmsAttWeight weights))

  liftIO $ do
    normalizedInputMutable <- V.thaw normalizedInput -- mutable copy of normalizedInput (one allocation)

    -- This normalized input is projected into Q, K, V spaces
    applyMatrixVectorMult (getArray2D layerIdx (wq weights)) normalizedInputMutable queryOutput
    applyMatrixVectorMult (getArray2D layerIdx (wk weights)) normalizedInputMutable keyOutput
    applyMatrixVectorMult (getArray2D layerIdx (wv weights)) normalizedInputMutable valueOutput

  -- Rotary position encodings are applied to Q and K
  applyRotations freqCosValues freqSinValues queryOutput
  applyRotations freqCosValues freqSinValues keyOutput

  forM_ [0 .. numHeads - 1] $ \headIndex -> do
    let keyHeadSlice = MV.slice (headIndex * headDim) headDim keyOutput
    let valueHeadSlice = MV.slice (headIndex * headDim) headDim valueOutput
    updateCacheWithHead layerIndex (HeadIndex headIndex) currentStep keyHeadSlice keyCache
    updateCacheWithHead layerIndex (HeadIndex headIndex) currentStep valueHeadSlice valueCache

-- Layer Token
createLayerToken :: StepCount -> LayerIndex -> V.Vector Float -> V.Vector Float -> TokenVector -> TransformerResult TokenVector
createLayerToken currentStep layerIndex freqCosValues freqSinValues inputToken = do
  network <- ask
  AttentionKV {queryOutput, keyCache, valueCache, projectedAttentionOutput, feedforwardNetworkOutput, multiHeadOutput} <- gets id
  let weights = weighting network
      LayerIndex layerIdx = layerIndex
      outputProjectionWeights = getArray2D layerIdx (wo weights)

  computeQKV weights currentStep layerIndex freqCosValues freqSinValues inputToken

  -- Compute multi-head attention directly into concatenatedHeads buffer
  computeMultiHeadAttention currentStep layerIndex queryOutput keyCache valueCache multiHeadOutput

  -- Apply output projection in-place
  liftIO $ applyMatrixVectorMult outputProjectionWeights multiHeadOutput projectedAttentionOutput

  -- Convert mutable output vector back to immutable
  attentionDelta <- liftIO $ V.freeze projectedAttentionOutput
  let TokenVector tokenVector = inputToken
      tokenAfterAttention = V.zipWith (+) tokenVector attentionDelta

  -- Apply FFN
  applyFeedForwardNetwork weights layerIndex tokenAfterAttention
  ffnOut <- liftIO $ V.freeze feedforwardNetworkOutput
  return $ TokenVector $ V.zipWith (+) tokenAfterAttention ffnOut

-- Transformer step
transformer :: Token -> StepCount -> TransformerResult (V.Vector Float)
transformer tokenCode stepCount = do
  network <- ask

  -- Getting the token embedding
  let weights = weighting network
      vocab = tokenEmbeddingTable weights
      rowStart = fromIntegral tokenCode * ncols vocab
      tokenVector = TokenVector $ V.slice rowStart (ncols vocab) (items2D vocab)

  -- Plucking out the current row of freq_cis_real and freq_cis_imag
  let StepCount step = stepCount
      freqCosValues = getRow step (freqCisReal weights)
      freqSinValues = getRow step (freqCisImag weights)

  -- Forwarding all the layers
  TokenVector finalTokenVector <-
    foldM
      (\accToken layerIndex -> createLayerToken stepCount (LayerIndex layerIndex) freqCosValues freqSinValues accToken)
      tokenVector
      [0 .. numLayers network - 1]

  -- Final rmsnorm
  let tokenWithRms = rmsNorm finalTokenVector (rmsFinalWeight weights)

  -- Classifier into logits
  let logits = V.generate (nrows vocab) $ \row ->
        let start = row * ncols vocab
            end = start + ncols vocab
            dot = sum [(items2D vocab V.! i) * (tokenWithRms V.! (i - start)) | i <- [start .. end - 1]]
         in dot

  return logits

generateNextToken :: StepCount -> PromptTokens -> Float -> Vocabulary -> Token -> Int -> TransformerResult (BS.ByteString, Token)
generateNextToken timestep promptTokens temperature vocab tokenCode seedValue = do
  network <- ask
  logits <- transformer tokenCode timestep
  let StepCount step = timestep
  nextToken <-
    if step < length promptTokens
      then return (promptTokens !! step)
      else
        if temperature == 0.0
          then return $ fromIntegral (V.maxIndex logits)
          else do
            liftIO $ drawSample seedValue $ softmax (V.map (/ temperature) logits) (vocabSize network)
  let word = vocab !! fromIntegral nextToken :: BS.ByteString
      firstChar = BSC.head word :: Char
      tokenStr =
        if tokenCode == 1 && isSpace firstChar
          then BSC.tail (vocab !! fromIntegral nextToken)
          else vocab !! fromIntegral nextToken
  return (tokenStr, nextToken)

generateTokens :: StepCount -> PromptTokens -> Float -> Vocabulary -> Int -> TransformerResult ([BS.ByteString], StepCount)
generateTokens maxSteps promptTokens temperature vocab seedValue = do
  network <- ask
  go network (StepCount 0) [] 1
  where
    go network timestep result token
      | timestep >= maxSteps || (timestep /= StepCount 0 && token == 1) = return (result, timestep)
      | otherwise = do
          (tokenStr, nextToken) <- generateNextToken timestep promptTokens temperature vocab token seedValue
          liftIO $ putStr $ BSC.unpack tokenStr
          liftIO $ hFlush stdout
          go network (timestep + StepCount 1) (result ++ [tokenStr]) nextToken
