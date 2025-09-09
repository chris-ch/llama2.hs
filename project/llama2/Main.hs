{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main (main) where

import Control.DeepSeq (deepseq)
import Control.Monad (foldM, forM, forM_, replicateM)
import Control.Monad.Reader (MonadIO (liftIO), MonadReader (ask), ReaderT (runReaderT))
import Control.Monad.State (StateT, evalStateT, gets)
import Data.Binary.Get (getFloatle, getInt32le)
import qualified Data.Binary.Get as BG
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import Data.Int (Int32)
import qualified Data.List as DL
import Data.Maybe (fromMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV
import GHC.Unicode (isSpace)
import qualified Options.Applicative as OA
import System.IO (hFlush, stdout)
import qualified System.Random as R
import Text.Printf (printf)

--------------------------------------------------------------------------------
-- Array2D Data Structure and Class
--------------------------------------------------------------------------------

data Array2D = Array2D
  { struct2D :: V.Vector Float,
    nrows :: Int,
    ncols :: Int
  }
  deriving (Show)

data Array3D = Array3D
  { struct3D :: V.Vector Float,
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
  getRow i Array2D {struct2D, ncols} = V.slice (i * ncols) ncols struct2D

  readArray2D :: Int -> Int -> BG.Get Array2D
  readArray2D rows cols = do
    vec <- readVector (rows * cols) >>= \v -> v `deepseq` return v
    return $ Array2D {struct2D = vec, nrows = rows, ncols = cols}

class Array3DOps a where
  getArray2D :: Int -> a -> Array2D
  readArray3D :: Int -> Int -> Int -> BG.Get a

instance Array3DOps Array3D where
  getArray2D :: Int -> Array3D -> Array2D
  getArray2D k Array3D {struct3D, sizeX, sizeY, sizeZ}
    | k < 0 || k >= sizeX = error "getArray2D: index out of bounds"
    | otherwise =
        Array2D
          { struct2D = V.slice (k * sizeY * sizeZ) (sizeY * sizeZ) struct3D,
            nrows = sizeY,
            ncols = sizeZ
          }

  readArray3D :: Int -> Int -> Int -> BG.Get Array3D
  readArray3D sx sy sz = do
    let totalSize = sx * sy * sz
    vec <- readVector totalSize
    return $ Array3D {struct3D = vec, sizeX = sx, sizeY = sy, sizeZ = sz}

--------------------------------------------------------------------------------
-- Options
--------------------------------------------------------------------------------

data Options = Options
  { seed :: Maybe Int,
    tokenizerFile :: FilePath,
    modelFile :: FilePath,
    temperature :: Double,
    steps :: Int,
    prompt :: Maybe String
  }

-- Parser for command-line options
optionsParser :: OA.Parser Options
optionsParser =
  Options
    <$> OA.optional (OA.option OA.auto (OA.long "seed" <> OA.help "Seed for debugging"))
    <*> OA.strOption (OA.long "tokenizer-file" <> OA.value "./data/tokenizer.bin" <> OA.help "Tokenizer binary file")
    <*> OA.strOption (OA.long "model-file" <> OA.value "./data/stories15M.bin" <> OA.metavar "MODEL_FILE" <> OA.help "Model binary file")
    <*> OA.option OA.auto (OA.long "temperature" <> OA.value 0.0 <> OA.metavar "TEMPERATURE" <> OA.help "Temperature")
    <*> OA.option OA.auto (OA.long "steps" <> OA.value 256 <> OA.metavar "STEPS" <> OA.help "Number of steps")
    <*> OA.optional (OA.strArgument (OA.metavar "PROMPT" <> OA.help "Initial prompt"))

main :: IO ()
main = do
  Options {..} <- OA.execParser $ OA.info (optionsParser OA.<**> OA.helper) OA.fullDesc
  modelFileContent <- BS.readFile modelFile
  tokenizerFileContent <- BS.readFile tokenizerFile
  run modelFileContent tokenizerFileContent (realToFrac temperature) steps prompt seed

--------------------------------------------------------------------------------
-- Types
--------------------------------------------------------------------------------

type MVectorFloat = MV.MVector (MV.PrimState IO) Float

type Vocabulary = [BS.ByteString]

type VocabularyScores = [Float]

type Token = Int32

type PromptTokens = [Token]

data AttentionKV = AttentionKV
  { keyCache :: MVectorFloat,
    valueCache :: MVectorFloat,
    gateBuffer :: MVectorFloat,
    upProjectionBuffer :: MVectorFloat,
    feedforwardNetworkOutput :: MVectorFloat,
    queryBuffer :: MVectorFloat,
    keyBuffer :: MVectorFloat,
    valueBuffer :: MVectorFloat,
    attentionOutputBuffer :: MVectorFloat
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
  { dim :: Int,
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
-- Binary Parsing
--------------------------------------------------------------------------------

readVector :: Int -> BG.Get (V.Vector Float)
readVector count = V.replicateM count getFloatle >>= \v -> v `deepseq` return v

parseNetworkConfigFile :: BG.Get NetworkConfig
parseNetworkConfigFile = do
  dim' <- fromIntegral <$> getInt32le
  hiddenDim' <- fromIntegral <$> getInt32le
  nLayers' <- fromIntegral <$> getInt32le
  numAttentionHeads' <- fromIntegral <$> getInt32le
  numKeyValueHeads' <- fromIntegral <$> getInt32le
  vocabSize' <- fromIntegral <$> getInt32le
  seqLen' <- fromIntegral <$> getInt32le
  tokenEmbeddingTable' <- readArray2D vocabSize' dim'
  rmsAttWeight' <- readArray2D nLayers' dim'
  wq' <- readArray3D nLayers' dim' dim'
  wk' <- readArray3D nLayers' dim' dim'
  wv' <- readArray3D nLayers' dim' dim'
  wo' <- readArray3D nLayers' dim' dim'
  rmsFfnWeight' <- readArray2D nLayers' dim'
  w1' <- readArray3D nLayers' hiddenDim' dim'
  w2' <- readArray3D nLayers' dim' hiddenDim'
  w3' <- readArray3D nLayers' hiddenDim' dim'
  rmsFinalWeight' <- readVector dim'
  freqCisReal' <- readArray2D seqLen' ((dim' `div` numAttentionHeads') `div` 2)
  freqCisImag' <- readArray2D seqLen' ((dim' `div` numAttentionHeads') `div` 2)

  let headDim = dim' `div` numAttentionHeads'
      weights =
        TransformerWeighting
          { tokenEmbeddingTable = tokenEmbeddingTable',
            rmsAttWeight = rmsAttWeight',
            wq = wq',
            wk = wk',
            wv = wv',
            wo = wo',
            rmsFfnWeight = rmsFfnWeight',
            w1 = w1',
            w2 = w2',
            w3 = w3',
            rmsFinalWeight = rmsFinalWeight',
            freqCisReal = freqCisReal',
            freqCisImag = freqCisImag'
          }
  return $
    NetworkConfig
      { dim = dim',
        hiddenDim = hiddenDim',
        numLayers = nLayers',
        numAttentionHeads = numAttentionHeads',
        numKeyValueHeads = numKeyValueHeads',
        vocabSize = abs vocabSize',
        seqLen = seqLen',
        headDimension = headDim,
        weighting = weights
      }

initModel :: BS.ByteString -> NetworkConfig
initModel = BG.runGet parseNetworkConfigFile

--------------------------------------------------------------------------------
-- Tokenizer
--------------------------------------------------------------------------------

parseTokens :: BS.ByteString -> Int -> (Vocabulary, VocabularyScores)
parseTokens fileContent size = (vocab, vocabScores)
  where
    scoresTokens = BG.runGet scoresAndTokens fileContent
    vocabScores = fst <$> scoresTokens
    vocab = snd <$> scoresTokens

    scoresAndTokens :: BG.Get [(Float, BS.ByteString)]
    scoresAndTokens = replicateM size readToken

    readToken :: BG.Get (Float, BS.ByteString)
    readToken = do
      score <- BG.getFloatle
      tokenSize <- BG.getInt32le
      token <- BG.getLazyByteString (fromIntegral tokenSize)
      return (score, token)

tokenizerInit :: BS.ByteString -> Int -> BS.ByteString -> (PromptTokens, Vocabulary)
tokenizerInit file size prompt = (bpeEncode prompt vocab vocabScores, vocab)
  where
    (vocab, vocabScores) = parseTokens (BS.drop 4 file) size

strLookup :: BS.ByteString -> Vocabulary -> Int
strLookup occurrence = fromMaybe (-1) . DL.elemIndex occurrence

processTokens :: [Token] -> Vocabulary -> VocabularyScores -> PromptTokens
processTokens tokens vocab vocabScores = case findBestPair tokens of
  Just (bestIndex, bestToken) ->
    processTokens (mergePair bestIndex bestToken tokens) vocab vocabScores
  Nothing ->
    tokens
  where
    findBestPair :: [Token] -> Maybe (Int, Token)
    findBestPair tokens' = foldr checkPair Nothing (zip [0 ..] (zip tokens' (drop 1 tokens')))
      where
        checkPair :: (Int, (Token, Token)) -> Maybe (Int, Token) -> Maybe (Int, Token)
        checkPair (count, (tokenPrev, tokenNext)) acc =
          case strLookup ((vocab !! fromIntegral tokenPrev) `BS.append` (vocab !! fromIntegral tokenNext)) vocab of
            pos | pos /= -1 && vocabScores !! pos > bestScore -> Just (count, fromIntegral pos)
            _ -> acc

        bestScore :: Float
        bestScore = -1e10

    mergePair :: Int -> Token -> [Token] -> [Token]
    mergePair count token tokens' =
      take count tokens' ++ [token] ++ drop (count + 2) tokens'

bpeEncode :: BS.ByteString -> Vocabulary -> VocabularyScores -> PromptTokens
bpeEncode prompt vocab vocabScores =
  let tokens = map (\char -> fromMaybe (error "Character not found in vocabulary") (DL.elemIndex (BS.pack [char]) vocab)) (BS.unpack prompt)
   in processTokens (map fromIntegral tokens) vocab vocabScores

--------------------------------------------------------------------------------
-- Transformer runtime
--------------------------------------------------------------------------------

type TransformerResult a = ReaderT NetworkConfig (StateT AttentionKV IO) a

-- Cache indexing helper
cacheIndex :: NetworkConfig -> Int -> Int -> Int -> Int -> Int
cacheIndex NetworkConfig {numAttentionHeads, seqLen, headDimension} layerIndex headIndex stepIndex dimensionIndex =
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
updateCacheWithHead :: NetworkConfig -> Int -> Int -> Int -> MVectorFloat -> MVectorFloat -> IO ()
updateCacheWithHead network layerIndex headIndex stepIndex headSlice cache = do
  let headDim = headDimension network
  let cacheOffset = cacheIndex network layerIndex headIndex stepIndex 0
  MV.copy (MV.slice cacheOffset headDim cache) headSlice

-- Rotary application to mutable buffer slice
applyRotaryPositionEncoding :: MVectorFloat -> Int -> V.Vector Float -> V.Vector Float -> IO ()
applyRotaryPositionEncoding buffer startOffset cosFrequencies sinFrequencies = do
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
applyRotationsToBuf :: V.Vector Float -> V.Vector Float -> Int -> Int -> MVectorFloat -> IO ()
applyRotationsToBuf cosFrequencies sinFrequencies numHeads headDim buffer = do
  forM_ [0 .. numHeads - 1] $ \headIndex -> do
    applyRotaryPositionEncoding buffer (headIndex * headDim) cosFrequencies sinFrequencies

-- Build activation directly into a provided mutable vector
applyAttentionWeights ::
  NetworkConfig ->
  -- | Layer index
  Int ->
  -- | Head index
  Int ->
  -- | Head dimension
  Int ->
  -- | Attention scores
  [Float] ->
  -- | Value cache
  MVectorFloat ->
  -- | Output buffer (must be preallocated to headDim)
  MVectorFloat ->
  IO ()
applyAttentionWeights network layerIndex headIndex headDim attentionScores valueCache outputBuffer = do
  -- Zero the output buffer
  MV.set outputBuffer 0.0
  let sequenceLength = length attentionScores
  forM_ [0 .. sequenceLength - 1] $ \positionIndex -> do
    let valueCacheOffset = cacheIndex network layerIndex headIndex positionIndex 0
    let valueSlice = MV.slice valueCacheOffset headDim valueCache
    let attentionWeight = attentionScores !! positionIndex
    forM_ [0 .. headDim - 1] $ \dimIndex -> do
      valueComponent <- MV.read valueSlice dimIndex
      MV.modify outputBuffer (+ (attentionWeight * valueComponent)) dimIndex

-- Attention scores
computeScores :: NetworkConfig -> Int -> Int -> Int -> MVectorFloat -> MVectorFloat -> Int -> IO (V.Vector Float)
computeScores network headDim layerIndex headIndex queryBuffer keyCache currentStep = do
  let scalingFactor = sqrt (fromIntegral headDim)
  let sequenceLength = currentStep + 1
  V.generateM sequenceLength $ \positionIndex -> do
    let keyCacheOffset = cacheIndex network layerIndex headIndex positionIndex 0
    let keySlice = MV.slice keyCacheOffset headDim keyCache
    let querySlice = MV.slice (headIndex * headDim) headDim queryBuffer
    dotProduct <- dotProductMV querySlice keySlice
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
  NetworkConfig ->
  -- | Number of heads
  Int ->
  -- | Head dimension
  Int ->
  -- | Current step
  Int ->
  -- | Query buffer
  MVectorFloat ->
  -- | Key cache
  MVectorFloat ->
  -- | Value cache
  MVectorFloat ->
  -- | Layer index
  Int ->
  IO [V.Vector Float]
computeMultiHeadAttention network numHeads headDim currentStep queryBuffer keyCache valueCache layerIndex = do
  forM [0 .. numHeads - 1] $ \headIndex -> do
    rawAttentionScores <- computeScores network headDim layerIndex headIndex queryBuffer keyCache currentStep
    let normalizedScores = softmax rawAttentionScores (V.length rawAttentionScores)
        attentionWeights = V.toList normalizedScores

    -- Reuse a mutable vector for accumulation
    headOutputBuffer <- MV.new headDim
    applyAttentionWeights network layerIndex headIndex headDim attentionWeights valueCache headOutputBuffer
    V.freeze headOutputBuffer

-- Math utils
matrixVectorMult :: Array2D -> V.Vector Float -> V.Vector Float
matrixVectorMult array2D vec = V.generate (nrows array2D) $ \row ->
  let start = row * ncols array2D
      end = start + ncols array2D
      dot =
        sum
          [ (struct2D array2D V.! i) * (vec V.! (i - start))
            | i <- [start .. end - 1]
          ]
   in dot

-- Multiply a matrix by a mutable input vector, writing into an output mutable vector
applyMatrixVectorMult ::
  Array2D ->
  -- | Input vector (mutable)
  MVectorFloat ->
  -- | Output vector (mutable)
  MVectorFloat ->
  IO ()
applyMatrixVectorMult array2D vecM result = do
  -- For each row
  let nrows' = nrows array2D
      ncols' = ncols array2D
      flatMat = struct2D array2D
  forM_ [0 .. nrows' - 1] $ \row -> do
    let rowStart = row * ncols'
        -- local references to avoid repeated lookups
        flat = flatMat
        rs = rowStart
    -- accumulate dot product in a strict loop
    let loop !i !acc
          | i >= ncols' = return acc
          | otherwise = do
              v <- MV.unsafeRead vecM i
              let m = V.unsafeIndex flat (rs + i)
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
applyFeedForwardNetwork :: TransformerWeighting -> Int -> V.Vector Float -> TransformerResult ()
applyFeedForwardNetwork weights layerIndex inputToken = do
  network <- ask
  AttentionKV {gateBuffer, upProjectionBuffer, feedforwardNetworkOutput} <- gets id
  let hiddenFFNDim = hiddenDim network
      silu activation = activation / (1.0 + exp (-activation))
      rmsFFNWeights = getRow layerIndex $ rmsFfnWeight weights
      normalizedInput = rmsNorm inputToken rmsFFNWeights

  -- thaw normalizedInput once (normalizedInput is a V.Vector), reuse it
  normalizedInputMutable <- liftIO $ V.thaw normalizedInput
  liftIO $ applyMatrixVectorMult (getArray2D layerIndex (w1 weights)) normalizedInputMutable gateBuffer

  -- gateActivation = silu(W1 * normalizedInput)
  liftIO $ forM_ [0 .. hiddenFFNDim - 1] $ \i -> MV.unsafeModify gateBuffer silu i

  -- upProjection = W3 * normalizedInput  -> upProjectionBuffer
  liftIO $ applyMatrixVectorMult (getArray2D layerIndex (w3 weights)) normalizedInputMutable upProjectionBuffer

  -- gateActivation *= upProjection   (in-place on gateBuffer)
  liftIO $ forM_ [0 .. hiddenFFNDim - 1] $ \i -> do
    upValue <- MV.unsafeRead upProjectionBuffer i
    MV.unsafeModify gateBuffer (* upValue) i

  -- result = W2 * gateActivation
  liftIO $ applyMatrixVectorMult (getArray2D layerIndex (w2 weights)) gateBuffer feedforwardNetworkOutput

-- QKV
computeQKV :: TransformerWeighting -> Int -> V.Vector Float -> V.Vector Float -> V.Vector Float -> Int -> TransformerResult ()
computeQKV weights layerIndex freqCosValues freqSinValues inputToken currentStep = do
  network <- ask
  AttentionKV {queryBuffer, keyBuffer, valueBuffer, keyCache, valueCache} <- gets id
  let headDim = headDimension network
      numHeads = numAttentionHeads network
      -- The input token gets RMS normalized
      normalizedInput = rmsNorm inputToken (getRow layerIndex (rmsAttWeight weights))

  normalizedInputMutable <- liftIO $ V.thaw normalizedInput -- mutable copy of normalizedInput (one allocation)

  -- This normalized input is projected into Q, K, V spaces
  liftIO $ applyMatrixVectorMult (getArray2D layerIndex (wq weights)) normalizedInputMutable queryBuffer
  liftIO $ applyMatrixVectorMult (getArray2D layerIndex (wk weights)) normalizedInputMutable keyBuffer
  liftIO $ applyMatrixVectorMult (getArray2D layerIndex (wv weights)) normalizedInputMutable valueBuffer

  -- Rotary position encodings are applied to Q and K
  liftIO $ applyRotationsToBuf freqCosValues freqSinValues numHeads headDim queryBuffer
  liftIO $ applyRotationsToBuf freqCosValues freqSinValues numHeads headDim keyBuffer

  liftIO $ forM_ [0 .. numHeads - 1] $ \headIndex -> do
    let keyHeadSlice = MV.slice (headIndex * headDim) headDim keyBuffer
    let valueHeadSlice = MV.slice (headIndex * headDim) headDim valueBuffer
    updateCacheWithHead network layerIndex headIndex currentStep keyHeadSlice keyCache
    updateCacheWithHead network layerIndex headIndex currentStep valueHeadSlice valueCache

-- Layer
createLayerToken :: Int -> Int -> V.Vector Float -> V.Vector Float -> V.Vector Float -> TransformerResult (V.Vector Float)
createLayerToken currentStep layerIndex freqCosValues freqSinValues inputToken = do
  network <- ask
  AttentionKV {queryBuffer, keyCache, valueCache, attentionOutputBuffer, feedforwardNetworkOutput} <- gets id
  let weights = weighting network
      attentionHeadDim = headDimension network
      numHeads = numAttentionHeads network
  computeQKV weights layerIndex freqCosValues freqSinValues inputToken currentStep
  attentionOutputs <- liftIO $ computeMultiHeadAttention network numHeads attentionHeadDim currentStep queryBuffer keyCache valueCache layerIndex
  let outputProjectionWeights = getArray2D layerIndex (wo weights)
      concatenatedHeads = V.concat attentionOutputs
  -- Convert immutable input vector to mutable
  concatenatedHeadsMutable <- liftIO $ V.thaw concatenatedHeads
  -- Perform in-place matrix-vector multiplication
  liftIO $ applyMatrixVectorMult outputProjectionWeights concatenatedHeadsMutable attentionOutputBuffer
  -- Convert mutable output vector back to immutable
  attentionDelta <- liftIO $ V.freeze attentionOutputBuffer
  let tokenAfterAttention = V.zipWith (+) inputToken attentionDelta
  applyFeedForwardNetwork weights layerIndex tokenAfterAttention
  ffnOut <- V.freeze feedforwardNetworkOutput
  return $ V.zipWith (+) tokenAfterAttention ffnOut

-- Transformer step
transformer :: Token -> Int -> TransformerResult (V.Vector Float)
transformer tokenCode stepCount = do
  network <- ask

  -- Getting the token embedding
  let weights = weighting network
      vocab = tokenEmbeddingTable weights
      rowStart = fromIntegral tokenCode * ncols vocab
      token = V.slice rowStart (ncols vocab) (struct2D vocab)

  -- Plucking out the current row of freq_cis_real and freq_cis_imag
  let freqCosValues = getRow stepCount (freqCisReal weights)
  let freqSinValues = getRow stepCount (freqCisImag weights)

  -- Forwarding all the layers
  finalToken <-
    foldM
      (\accToken indexLayer -> createLayerToken stepCount indexLayer freqCosValues freqSinValues accToken)
      token
      [0 .. numLayers network - 1]

  -- Final rmsnorm
  let tokenWithRms = rmsNorm finalToken (rmsFinalWeight weights)

  -- Classifier into logits
  let logits = matrixVectorMult vocab tokenWithRms

  return logits

generateNextToken :: Int -> PromptTokens -> Float -> Vocabulary -> Token -> Int -> TransformerResult (BS.ByteString, Token)
generateNextToken timestep promptTokens temperature vocab tokenCode seedValue = do
  network <- ask
  logits <- transformer tokenCode timestep
  nextToken <-
    if timestep < length promptTokens
      then return (promptTokens !! timestep)
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

generateTokens :: Int -> PromptTokens -> Float -> Vocabulary -> Int -> TransformerResult ([BS.ByteString], Int)
generateTokens maxSteps promptTokens temperature vocab seedValue = do
  network <- ask
  go network 0 [] 1
  where
    go network timestep result token
      | timestep >= maxSteps || (timestep /= 0 && token == 1) = return (result, timestep)
      | otherwise = do
          (tokenStr, nextToken) <- generateNextToken timestep promptTokens temperature vocab token seedValue
          liftIO $ putStr $ BSC.unpack tokenStr
          liftIO $ hFlush stdout
          go network (timestep + 1) (result ++ [tokenStr]) nextToken

-- Initialize flat attention KV caches (flattened to one MVector each)
initAttentionKV :: NetworkConfig -> IO AttentionKV
initAttentionKV NetworkConfig {numLayers, numAttentionHeads, seqLen, headDimension, hiddenDim, dim} = do
  let size = numLayers * numAttentionHeads * seqLen * headDimension
  keyCache <- MV.new size
  valueCache <- MV.new size
  gateBuffer <- MV.new hiddenDim
  upProjectionBuffer <- MV.new hiddenDim
  feedforwardNetworkOutput <- MV.new dim
  queryBuffer <- MV.new dim
  keyBuffer <- MV.new dim
  valueBuffer <- MV.new dim
  attentionOutputBuffer <- MV.new dim
  return AttentionKV {keyCache, valueCache, gateBuffer, upProjectionBuffer, feedforwardNetworkOutput, queryBuffer, keyBuffer, valueBuffer, attentionOutputBuffer}

run :: BS.ByteString -> BS.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
run modelFileContent tokenizerFileContent temperature steps prompt seed = do
  currentTime <- getPOSIXTime
  let seedValue = fromMaybe (round currentTime) seed
      config = initModel modelFileContent
      prompt' = fromMaybe "" prompt
      (promptTokens, vocab) = tokenizerInit tokenizerFileContent (vocabSize config) (BSC.pack prompt')
  attentionKV <- initAttentionKV config
  putStrLn "<s>"
  startTime <- getPOSIXTime
  (_, countTokens) <- evalStateT (runReaderT (generateTokens steps promptTokens temperature vocab seedValue) config) attentionKV
  endTime <- getPOSIXTime
  let duration :: Integer
      duration = round (endTime - startTime)
      tokensPerSec :: Float
      tokensPerSec = fromIntegral countTokens / fromIntegral duration
  printf "\nduration: %ds - (%.02f tokens/s)\n" duration tokensPerSec
  return ()
