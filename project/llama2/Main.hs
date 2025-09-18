module Main (main) where

import qualified Data.Binary.Get as BG
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.List as DL
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV
import qualified Foreign as F
import qualified Options.Applicative as OA
import qualified System.Random as R
import Control.Monad.Reader (ReaderT(runReaderT), MonadIO(liftIO), MonadReader(ask))
import Control.Monad.State (evalStateT, StateT, get, put )
import Data.Maybe (fromMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)
import GHC.Unicode (isSpace)
import System.IO (hFlush, stdout)
import Control.Monad (forM_, forM, replicateM)
import Data.ByteString (useAsCString)
import GHC.IO (unsafePerformIO)
import Text.Printf (printf)

newtype StepCount = StepCount Int deriving (Show, Eq, Ord, Num)
newtype TokenVector = TokenVector (V.Vector Float) deriving (Show)
type MVectorFloat = MV.MVector (MV.PrimState IO) Float
type Vocabulary = [BS.ByteString]
type VocabularyScores = [Float]
newtype Token = Token F.Int32 deriving (Show, Eq, Ord, Num)
type PromptTokens = [Token]

-- Runtime type alias
type TransformerResult a = ReaderT ModelConfig (StateT DecoderCache IO) a

data ModelConfig = ModelConfig
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

--------------------------------------------------------------------------------
-- Data Structure and Class
--------------------------------------------------------------------------------

-- Generic data definitions

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
  byteData <- BG.getByteString (count * 4)
  return $! unsafePerformIO $ do
    useAsCString byteData $ \ptr -> do
      let floatPtr = F.castPtr ptr :: F.Ptr Float
      V.generateM count (F.peekElemOff floatPtr)

-- Data definitions for LLM architecture

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
  { multiHeadCache :: [HeadCache]
  }

newtype DecoderCache = DecoderCache
  { layerCache :: [LayerAttentionCache]
  }

--------------------------------------------------------------------------------
-- Main entry point
--------------------------------------------------------------------------------

main :: IO ()
main = do
  Options {seed, tokenizerFile, modelFile, temperature, steps,
         prompt} <- OA.execParser $ OA.info (optionsParser OA.<**> OA.helper) OA.fullDesc
  modelFileContent <- BS.readFile modelFile
  tokenizerFileContent <- BS.readFile tokenizerFile
  runModel modelFileContent tokenizerFileContent (realToFrac temperature) steps prompt seed

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

--------------------------------------------------------------------------------
-- Binary Parsing
--------------------------------------------------------------------------------
parseModelConfigFile :: BG.Get ModelConfig
parseModelConfigFile = do
  modelDim' <- fromIntegral <$> BG.getInt32le
  hiddenDim' <- fromIntegral <$> BG.getInt32le
  nLayers' <- fromIntegral <$> BG.getInt32le
  numAttentionHeads' <- fromIntegral <$> BG.getInt32le
  numKeyValueHeads' <- fromIntegral <$> BG.getInt32le
  vocabSize' <- fromIntegral <$> BG.getInt32le
  seqLen' <- fromIntegral <$> BG.getInt32le
  tokenEmbeddingTable' <- readArray2D vocabSize' modelDim'
  rmsAttWeight' <- readArray2D nLayers' modelDim' :: BG.Get Array2D
  wq' <- readArray3D nLayers' modelDim' modelDim' :: BG.Get Array3D
  wk' <- readArray3D nLayers' modelDim' (numKeyValueHeads' * (modelDim' `div` numAttentionHeads')) :: BG.Get Array3D
  wv' <- readArray3D nLayers' modelDim' (numKeyValueHeads' * (modelDim' `div` numAttentionHeads')) :: BG.Get Array3D
  wo' <- readArray3D nLayers' modelDim' modelDim' :: BG.Get Array3D
  rmsFfnWeight' <- readArray2D nLayers' modelDim' :: BG.Get Array2D
  w1' <- readArray3D nLayers' hiddenDim' modelDim' :: BG.Get Array3D
  w2' <- readArray3D nLayers' modelDim' hiddenDim' :: BG.Get Array3D
  w3' <- readArray3D nLayers' hiddenDim' modelDim' :: BG.Get Array3D
  rmsFinalWeight' <- readVector modelDim'
  freqCisReal' <- readArray2D seqLen' ((modelDim' `div` numAttentionHeads') `div` 2)
  freqCisImag' <- readArray2D seqLen' ((modelDim' `div` numAttentionHeads') `div` 2)

  let
      headDim = modelDim' `div` numAttentionHeads'
      -- Construct the Embedding
      embedding = EmbeddingComponent
        { vocabulary = tokenEmbeddingTable',
          rmsFinalWeight = rmsFinalWeight'
        }
      -- Construct the parameters for Transformer layers
      layers =
        [ TransformerLayerComponent {
          multiHeadAttention = MultiHeadAttentionComponent
                { heads = [ SingleHeadComponent
                              { wqHead = getHeadArray2D layerIdx headIdx headDim wq'
                              , wkHead = getHeadArray2D layerIdx (headIdx `div` (numAttentionHeads' `div` numKeyValueHeads')) headDim wk'
                              , wvHead = getHeadArray2D layerIdx headIdx headDim wv'
                              , rotary = RotaryEncodingComponent { freqCos = freqCisReal', freqSin = freqCisImag' }
                              }
                          | headIdx <- [0..numAttentionHeads' - 1]
                          ]
                , mWo     = getArray2D layerIdx wo'
                , rmsAtt = getRow layerIdx rmsAttWeight'
                },
            feedforwardNetwork = FeedForwardNetworkComponent
                     { fW1 = getArray2D layerIdx w1',
                       fW2 = getArray2D layerIdx w2',
                       fW3 = getArray2D layerIdx w3',
                       fRMSFfn = getRow layerIdx rmsFfnWeight'
                     }
            }
        | layerIdx <- [0..nLayers' - 1]
        ]
      -- Construct the TransformerArchitecture
      decoder = TransformerDecoderComponent
        { modelEmbedding = embedding,
          modelLayers = layers
        }
  return $
    ModelConfig
      { modelDim = modelDim',
        hiddenDim = hiddenDim',
        numLayers = nLayers',
        numAttentionHeads = numAttentionHeads',
        numKeyValueHeads = numKeyValueHeads',
        vocabSize = abs vocabSize',
        seqLen = seqLen',
        headDimension = headDim,
        decoder = decoder
      }

getHeadArray2D :: Int      -- ^ layer index
               -> Int      -- ^ head index
               -> Int      -- ^ headDim (rows per head)
               -> Array3D  -- ^ the big 3D tensor (layers × rows × cols)
               -> Array2D  -- ^ the resulting (headDim × ncols) matrix
getHeadArray2D layerIdx headIdx headDim (Array3D vec _ sizeY sizeZ) =
  let
    -- sizeY is the total number of rows = numHeads * headDim
    ncols    = sizeZ
    startRow = headIdx * headDim
    -- Offset to this layer’s slice:
    layerOffset = layerIdx * sizeY * sizeZ
    -- Grab each row belonging to this head:
    newItems = V.concat
      [ V.slice (layerOffset + (row * ncols)) ncols vec
      | row <- [startRow .. startRow + headDim - 1]
      ]
  in Array2D { items2D = newItems, nrows = headDim, ncols = ncols }

initDecoderCaches :: ModelConfig -> IO DecoderCache
initDecoderCaches ModelConfig{numLayers, seqLen, headDimension, decoder} = do
  let numHeads = length (heads (multiHeadAttention (head (modelLayers decoder))))
      cacheSize = seqLen * headDimension
  layerCachesList <- replicateM numLayers $ do
    headCachesList <- replicateM numHeads $ do
      kc <- MV.new cacheSize
      vc <- MV.new cacheSize
      return $ HeadCache { headKeyCache = kc, headValueCache = vc }
    return $ LayerAttentionCache { multiHeadCache = headCachesList }
  return $ DecoderCache { layerCache = layerCachesList }

--------------------------------------------------------------------------------
-- Tokenizer
--------------------------------------------------------------------------------

parseTokens :: BS.ByteString -> Int -> (Vocabulary, VocabularyScores)
parseTokens fileContent size = (vocab, vocabScores)
  where
    scoresTokens = BG.runGet scoresAndTokens fileContent
    vocabScores = map fst scoresTokens
    vocab = map snd scoresTokens

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

applyBPEMerges :: [Token] -> Vocabulary -> VocabularyScores -> PromptTokens
applyBPEMerges tokens vocab vocabScores = case findBestPair tokens of
  Just (bestIndex, bestToken) ->
    applyBPEMerges (mergePair bestIndex bestToken tokens) vocab vocabScores
  Nothing ->
    tokens
  where
    findBestPair :: [Token] -> Maybe (Int, Token)
    findBestPair tokens' = foldr checkPair Nothing (zip [0 ..] (zip tokens' (drop 1 tokens')))
      where
        checkPair :: (Int, (Token, Token)) -> Maybe (Int, Token) -> Maybe (Int, Token)
        checkPair (count, (Token tokenPrev, Token tokenNext)) acc =
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
   in applyBPEMerges (map fromIntegral tokens) vocab vocabScores

--------------------------------------------------------------------------------
-- LLM Structure
--------------------------------------------------------------------------------

-- Vector multiplication by a Matrix
matrixVectorMult :: Array2D -> V.Vector Float -> V.Vector Float
matrixVectorMult (Array2D items rows cols) vec = V.generate rows $ \i ->
      let rowStart = i * cols
          rowElements = V.slice rowStart cols items
      in V.sum $ V.zipWith (*) rowElements vec

-- RMS Norm
rmsNorm :: TokenVector -> V.Vector Float -> TokenVector
rmsNorm (TokenVector vector) weights =
  let squareNorm = V.foldl' (\acc v -> acc + v * v) 0.0 vector
      ss = (squareNorm / fromIntegral (V.length vector)) + 1e-5
      normalized = V.map (* (1.0 / sqrt ss)) vector
   in TokenVector $ V.zipWith (*) weights normalized

-- Activation
sigmoidLinearUnit :: Float -> Float
sigmoidLinearUnit x = x / (1.0 + exp (-x))

-- Softmax
softmax :: V.Vector Float -> Int -> V.Vector Float
softmax values size = V.concat [softmaxValues, V.slice size (V.length values - size) values]
  where
    maxVal = V.maximum (V.take size values)
    expValues = V.map (\x -> exp (x - maxVal)) (V.take size values)
    sumExpValues = V.sum expValues
    softmaxValues = V.map (/ sumExpValues) expValues

embed :: EmbeddingComponent -> Token -> TransformerResult TokenVector
embed embedding (Token tokenCode) = do
  let
    vocab = vocabulary embedding
    rowStart = fromIntegral tokenCode * ncols vocab
    tokenVector = TokenVector $ V.slice rowStart (ncols vocab) (items2D vocab)
  return tokenVector

applyRotation :: RotaryEncodingComponent -> StepCount -> TokenVector -> TransformerResult TokenVector
applyRotation (RotaryEncodingComponent freqCos freqSin) (StepCount step) inputToken = do
  network <- ask
  let
      cosFrequencies = getRow step freqCos
      sinFrequencies = getRow step freqSin
      headDim = headDimension network

  -- Apply rotation per head
  return $ applyRotaryPositionEncoding headDim cosFrequencies sinFrequencies inputToken

applyRotaryPositionEncoding :: Int -> V.Vector Float -> V.Vector Float -> TokenVector -> TokenVector
applyRotaryPositionEncoding headDim cosFrequencies sinFrequencies (TokenVector inputVector) = let

  processedPairs = map (\pairIndex ->
    let realComponent = inputVector V.! pairIndex
        imagComponent = inputVector V.! (pairIndex + 1)
        cosValue = cosFrequencies V.! (pairIndex `div` 2)
        sinValue = sinFrequencies V.! (pairIndex `div` 2)
        rotatedReal = realComponent * cosValue - imagComponent * sinValue
        rotatedImag = realComponent * sinValue + imagComponent * cosValue
    in [(pairIndex, rotatedReal), (pairIndex + 1, rotatedImag)]
    ) [0, 2 .. headDim - 2]

  updates = concat processedPairs
  rotated = inputVector V.// updates
  result = inputVector V.// zip [0..headDim-1] (V.toList rotated)
  in TokenVector result

runFeedForward :: FeedForwardNetworkComponent -> TokenVector -> TransformerResult (V.Vector Float)
runFeedForward feedForwardNetwork inputToken = do
  let
    rmsFfnWeights = fRMSFfn feedForwardNetwork
    w1 = fW1 feedForwardNetwork
    w2 = fW2 feedForwardNetwork
    w3 = fW3 feedForwardNetwork
    TokenVector normalizedInput = rmsNorm inputToken rmsFfnWeights
    gateOutput' = matrixVectorMult w1 normalizedInput
    upProjectionOutput' = matrixVectorMult w3 normalizedInput
    gateOutput = V.map sigmoidLinearUnit gateOutput'
    feedforwardNetworkOutput' = matrixVectorMult w2 (V.zipWith (*) gateOutput upProjectionOutput')
  return feedforwardNetworkOutput'

-- Per-head attention computation
singleHeadAttention :: HeadCache -> TokenVector -> StepCount -> Int -> TransformerResult TokenVector
singleHeadAttention (HeadCache kc vc) (TokenVector qHead) (StepCount stepInt) headDim = do
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
      result = DL.foldl' addScaled zeroVec [0 .. sequenceLength - 1]
  return $ TokenVector result

-- QKV per head
runSingleHeadQKV :: SingleHeadComponent -> TokenVector -> TransformerResult (TokenVector, TokenVector, TokenVector)
runSingleHeadQKV headComp (TokenVector normalizedInput) = do
  let q = matrixVectorMult (wqHead headComp) normalizedInput
      k = matrixVectorMult (wkHead headComp) normalizedInput
      v = matrixVectorMult (wvHead headComp) normalizedInput
  return (TokenVector q, TokenVector k, TokenVector v)

-- Rotary application
applyRotaryToHead :: SingleHeadComponent -> StepCount -> (TokenVector , TokenVector ) -> TransformerResult (TokenVector , TokenVector )
applyRotaryToHead headComp step (q, k) = do
  let rot = rotary headComp
  q' <- applyRotation rot step q
  k' <- applyRotation rot step k
  return (q', k')

-- Per-head cache update
updateCache :: MVectorFloat -> StepCount -> Int -> TokenVector -> TransformerResult ()
updateCache cache (StepCount step) headDim (TokenVector slice) = do
  let offset = step * headDim
  hsm <- V.thaw slice
  liftIO $ MV.copy (MV.slice offset headDim cache) hsm

runLayer :: MultiHeadAttentionComponent -> FeedForwardNetworkComponent -> LayerAttentionCache -> TokenVector -> StepCount -> TransformerResult (TokenVector, LayerAttentionCache)
runLayer mha feedForwardNetwork layerCache inputToken step = do
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
  let headCaches = multiHeadCache layerCache
  forM_ (zip headCaches qkvList) $ \(hc, (_, k', v')) -> do
    let (HeadCache kc vc) = hc
    updateCache kc step headDim k'
    updateCache vc step headDim v'
    return ()

  -- Compute per-head attention outputs
  headOutputs <- forM (zip qList headCaches) $ \(qHead, hc) ->
    singleHeadAttention hc qHead step headDim

  let multiHeadOut = V.concat [vec | TokenVector vec <- headOutputs]
      attentionDelta = matrixVectorMult outputProjectionWeights multiHeadOut
      TokenVector inputVector = inputToken
      tokenAfterAttention = V.zipWith (+) inputVector attentionDelta

  -- Apply FFN
  ffnOut <- runFeedForward feedForwardNetwork (TokenVector tokenAfterAttention)
  let finalToken = V.zipWith (+) tokenAfterAttention ffnOut

  return (TokenVector finalToken, layerCache)

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

-- classifier logits for a given token vector
transformerLogits :: TransformerDecoderComponent -> TokenVector -> TransformerResult TokenVector
transformerLogits dec tokenVector = do
  let
      vocab = vocabulary (modelEmbedding dec)
      rmsWeight = rmsFinalWeight (modelEmbedding dec)
      TokenVector tokenWithRms = rmsNorm tokenVector rmsWeight
      logits = V.generate (nrows vocab) $ \row ->
        let start = row * ncols vocab
            end = start + ncols vocab
            dot = sum [(items2D vocab V.! i) * (tokenWithRms V.! (i - start)) | i <- [start .. end - 1]]
         in dot
  return $ TokenVector logits

--------------------------------------------------------------------------------
-- Transformer runtime
--------------------------------------------------------------------------------

-- Transformer step
transformer :: Token -> StepCount -> TransformerResult TokenVector
transformer inputTokenCode stepCount = do
  network <- ask
  decoderCache <- get
  let
    dec = decoder network
    embeddingLayer = modelEmbedding dec
    layers = modelLayers dec
  inputTokenVector <- embed embeddingLayer inputTokenCode
  (outputTokenVector, updatedCache) <- runLayers layers (layerCache decoderCache) inputTokenVector stepCount
  put $ DecoderCache { layerCache = updatedCache }
  -- Classifier into logits
  transformerLogits dec outputTokenVector

-- Sampling
drawSample :: Int -> V.Vector Float -> IO Token
drawSample randomSeed probabilities = do
  let gen = R.mkStdGen randomSeed
      (randomValue, _) = R.random gen :: (Float, R.StdGen)
      cumulativeDistribution = V.scanl1 (+) probabilities
      selectedIndex = V.length (V.takeWhile (< randomValue) cumulativeDistribution)
  return $ fromIntegral (min selectedIndex (V.length probabilities - 1))

generateNextToken :: StepCount -> PromptTokens -> Float -> Vocabulary -> Token -> Int -> TransformerResult (BS.ByteString, Token)
generateNextToken timestep promptTokens temperature vocab tokenCode seedValue = do
  network <- ask
  TokenVector logits <- transformer tokenCode timestep
  let StepCount step = timestep
  Token nextToken <-
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
  return (tokenStr, Token nextToken)

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

runModel :: BS.ByteString -> BS.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
runModel modelFileContent tokenizerFileContent temperature steps prompt seed = do
  currentTime <- getPOSIXTime
  let
    seedValue = fromMaybe (round currentTime) seed
    initModel :: BS.ByteString -> ModelConfig
    initModel = BG.runGet parseModelConfigFile
    config = initModel modelFileContent
    prompt' = fromMaybe "" prompt
    (promptTokens, vocab) = tokenizerInit tokenizerFileContent (vocabSize config) (BSC.pack prompt')
  attentionKV <- initDecoderCaches config
  putStrLn "<s>"
  startTime <- getPOSIXTime
  (_, StepCount countTokens) <- evalStateT (runReaderT (generateTokens (StepCount steps) promptTokens temperature vocab seedValue) config) attentionKV
  endTime <- getPOSIXTime
  let duration :: Integer
      duration = round (endTime - startTime)
      tokensPerSec :: Float
      tokensPerSec = fromIntegral countTokens / fromIntegral duration
  printf "\nduration: %ds - (%.02f tokens/s)\n" duration tokensPerSec
  return ()
