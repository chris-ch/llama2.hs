{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE LambdaCase #-}
module Main (main) where

import qualified Options.Applicative as OA
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.Binary.Get as BG
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV
import qualified Data.List as DL
import qualified System.Random as R

import Control.Monad (replicateM, foldM, forM_, forM)
import Control.Monad.State ( StateT, evalStateT, gets )
import Control.Monad.Reader ( MonadIO(liftIO), ReaderT(runReaderT), MonadReader(ask) )
import Data.Maybe (fromMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)
import GHC.Unicode (isSpace)
import Text.Printf (printf)
import Data.Binary.Get (getInt32le, getFloatle)
import Data.Int (Int32)
import System.IO (hFlush, stdout)
import Control.DeepSeq (deepseq)

--------------------------------------------------------------------------------
-- Array2D Data Structure and Class
--------------------------------------------------------------------------------

data Array2D = Array2D
    { vec :: V.Vector Float
    , nrows :: Int
    , ncols :: Int
    } deriving (Show)

class Array2DOps a where
    getRow :: Int -> a -> V.Vector Float
    readArray2D :: Int -> Int -> BG.Get a  -- rows -> cols -> parser

instance Array2DOps Array2D where

    getRow :: Int -> Array2D -> V.Vector Float
    getRow i Array2D {vec, ncols} = V.slice (i * ncols) ncols vec

    readArray2D :: Int -> Int -> BG.Get Array2D
    readArray2D rows cols = do
        vec <- readVector (rows * cols) >>= \v -> v `deepseq` return v
        return $ Array2D {vec = vec, nrows = rows, ncols = cols}

--------------------------------------------------------------------------------
-- Options
--------------------------------------------------------------------------------

data Options = Options
    { seed :: Maybe Int
    , tokenizerFile :: FilePath
    , modelFile :: FilePath
    , temperature :: Double
    , steps :: Int
    , prompt :: Maybe String
    }

-- Parser for command-line options
optionsParser :: OA.Parser Options
optionsParser = Options
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

-- Flat Key/Value caches
data AttentionKV = AttentionKV
    { keyCache :: MVectorFloat
    , valueCache :: MVectorFloat
    , ffnBuf1    :: MVectorFloat   -- scratch for hidden1
    , ffnBuf2    :: MVectorFloat   -- scratch for hidden3
    , ffnBufOut  :: MVectorFloat   -- scratch for result
    , qBuf       :: MVectorFloat   -- scratch for query projection
    , kBuf       :: MVectorFloat   -- scratch for key projection
    , vBuf       :: MVectorFloat   -- scratch for value projection
    }

data TransformerWeighting = TransformerWeighting
    { tokenEmbeddingTable :: V.Vector Float
    , tokenEmbeddingTableRows :: Int
    , tokenEmbeddingTableCols :: Int
    , rmsAttWeight :: Array2D
    , wq :: Array2D
    , wk :: Array2D
    , wv :: Array2D
    , wo :: Array2D
    , rmsFfnWeight :: Array2D
    , w1 :: Array2D
    , w2 :: Array2D
    , w3 :: Array2D
    , rmsFinalWeight :: V.Vector Float
    , freqCisReal :: Array2D
    , freqCisImag :: Array2D
    } deriving (Show)

data NetworkConfig = NetworkConfig
    { dim :: Int
    , hiddenDim :: Int
    , nLayers :: Int
    , numAttentionHeads :: Int
    , numKeyValueHeads :: Int
    , vocabSize :: Int
    , seqLen :: Int
    , headDimension :: Int
    , weighting :: TransformerWeighting
    } deriving (Show)

--------------------------------------------------------------------------------
-- Binary Parsing
--------------------------------------------------------------------------------

readVector :: Int -> BG.Get (V.Vector Float)
readVector count = V.replicateM count getFloatle

parseNetworkConfigFile :: BG.Get NetworkConfig
parseNetworkConfigFile = do
        dim' <- fromIntegral <$> getInt32le
        hiddenDim' <- fromIntegral <$> getInt32le
        nLayers' <- fromIntegral <$> getInt32le
        numAttentionHeads' <- fromIntegral <$> getInt32le
        numKeyValueHeads' <- fromIntegral <$> getInt32le
        vocabSize' <- fromIntegral <$> getInt32le
        seqLen' <- fromIntegral <$> getInt32le
        tokenEmbeddingTable' <- readVector (vocabSize' * dim') >>= \v -> v `deepseq` return v
        rmsAttWeight' <- readArray2D nLayers' dim'
        wq' <- readArray2D nLayers' (dim' * dim')
        wk' <- readArray2D nLayers' (dim' * dim')
        wv' <- readArray2D nLayers' (dim' * dim')
        wo' <- readArray2D nLayers' (dim' * dim')
        rmsFfnWeight' <- readArray2D nLayers' dim'
        w1' <- readArray2D nLayers' (hiddenDim' * dim')
        w2' <- readArray2D nLayers' (dim' * hiddenDim')
        w3' <- readArray2D nLayers' (hiddenDim' * dim')
        rmsFinalWeight' <- readVector dim' >>= \v -> v `deepseq` return v
        freqCisReal' <- readArray2D seqLen' ((dim' `div` numAttentionHeads') `div` 2)
        freqCisImag' <- readArray2D seqLen' ((dim' `div` numAttentionHeads') `div` 2)

        let
            headDim = dim' `div` numAttentionHeads'
            weights = TransformerWeighting
                { tokenEmbeddingTable = tokenEmbeddingTable'
                , tokenEmbeddingTableRows = vocabSize'
                , tokenEmbeddingTableCols = dim'
                , rmsAttWeight = rmsAttWeight'
                , wq = wq'
                , wk = wk'
                , wv = wv'
                , wo = wo'
                , rmsFfnWeight = rmsFfnWeight'
                , w1 = w1'
                , w2 = w2'
                , w3 = w3'
                , rmsFinalWeight = rmsFinalWeight'
                , freqCisReal = freqCisReal'
                , freqCisImag = freqCisImag'
                }
        return $ NetworkConfig
            { dim = dim'
            , hiddenDim = hiddenDim'
            , nLayers = nLayers'
            , numAttentionHeads = numAttentionHeads'
            , numKeyValueHeads = numKeyValueHeads'
            , vocabSize = abs vocabSize'
            , seqLen = seqLen'
            , headDimension = headDim
            , weighting = weights
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
      findBestPair tokens' = foldr checkPair Nothing (zip [0..] (zip tokens' (drop 1 tokens')))
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
cacheIndex NetworkConfig {numAttentionHeads, seqLen, headDimension} layer numHead step headIndex =
    (((layer * numAttentionHeads + numHead) * seqLen) + step) * headDimension + headIndex

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
drawSample seedValue probabilities = do
  let gen = R.mkStdGen seedValue
      (r, _) = R.random gen :: (Float, R.StdGen)
      cdf = V.scanl1 (+) probabilities
      idx = V.length (V.takeWhile (< r) cdf)
  return $ fromIntegral (min idx (V.length probabilities - 1))

-- Helper: write single head vector from mutable slice to the flat cache
writeHeadToCacheMV :: NetworkConfig -> MVectorFloat -> Int -> Int -> Int -> MVectorFloat -> IO ()
writeHeadToCacheMV net cache layer numHead step slice = do
  let headDim = headDimension net
  let offset = cacheIndex net layer numHead step 0
  MV.copy (MV.slice offset headDim cache) slice

-- Dot product on mutable vectors
dotProductMV :: MVectorFloat -> MVectorFloat -> IO Float
dotProductMV vec1 vec2 = do
  let len = min (MV.length vec1) (MV.length vec2)
  foldM (\acc i -> do
    a <- MV.read vec1 i
    b <- MV.read vec2 i
    return $ acc + a * b
    ) 0.0 [0 .. len - 1]

-- Rotary application to mutable buffer slice
applyRotationsMV :: MVectorFloat -> Int -> V.Vector Float -> V.Vector Float -> IO ()
applyRotationsMV buf offset freqCisRealRow freqCisImagRow = do
  let headDim = V.length freqCisRealRow * 2
  forM_ [0, 2 .. headDim - 2] $ \i -> do
    let idx = offset + i
    v <- MV.read buf idx
    v' <- MV.read buf (idx + 1)
    let real = freqCisRealRow V.! (i `div` 2)
        imag = freqCisImagRow V.! (i `div` 2)
        newv = v * real - v' * imag
        newv' = v * imag + v' * real
    MV.write buf idx newv
    MV.write buf (idx + 1) newv'

-- Apply rotations to entire buffer (per head)
applyRotationsToBuf :: MVectorFloat -> V.Vector Float -> V.Vector Float -> Int -> Int -> IO ()
applyRotationsToBuf buf freqCisRealRow freqCisImagRow numHeads headDim = do
  forM_ [0 .. numHeads - 1] $ \h -> do
    applyRotationsMV buf (h * headDim) freqCisRealRow freqCisImagRow

-- Build activation directly into a provided mutable vector
buildActivationInPlace
  :: NetworkConfig
  -> Int          -- ^ Layer index
  -> Int          -- ^ Head index
  -> Int          -- ^ Head dimension
  -> [Float]      -- ^ Attention scores
  -> MVectorFloat -- ^ Value cache
  -> MVectorFloat -- ^ Output buffer (must be preallocated to headDim)
  -> IO ()
buildActivationInPlace net indexLayer indexHead headDim headScores vCache out = do
  -- Zero the output buffer
  MV.set out 0.0
  let numPos = length headScores
  forM_ [0 .. numPos - 1] $ \pos -> do
    let offset = cacheIndex net indexLayer indexHead pos 0
    let vSlice = MV.slice offset headDim vCache
    let scale = headScores !! pos
    forM_ [0 .. headDim - 1] $ \j -> do
      v <- MV.read vSlice j
      MV.modify out (+ (scale * v)) j

-- Attention scores
computeScores :: NetworkConfig -> Int -> Int -> Int -> MVectorFloat -> MVectorFloat -> Int -> IO (V.Vector Float)
computeScores net headDim indexLayer indexHead qBuf kCache stepCount = do
  let sqrtHead = sqrt (fromIntegral headDim)
  let numPos = stepCount + 1
  V.generateM numPos $ \pos -> do
    let offset = cacheIndex net indexLayer indexHead pos 0
    let kSlice = MV.slice offset headDim kCache
    let qSlice = MV.slice (indexHead * headDim) headDim qBuf
    dot <- dotProductMV qSlice kSlice
    return $ dot / sqrtHead

-- Multihead
multiheadActivation
  :: NetworkConfig
  -> Int          -- ^ Number of heads
  -> Int          -- ^ Head dimension
  -> Int          -- ^ Current step
  -> MVectorFloat -- ^ Q buffer
  -> MVectorFloat -- ^ Key cache
  -> MVectorFloat -- ^ Value cache
  -> Int          -- ^ Layer index
  -> IO [V.Vector Float]
multiheadActivation net numHeads headDim stepCount qBuf kCache vCache indexLayer = do
  forM [0 .. numHeads - 1] $ \indexHead -> do
    rawScores <- computeScores net headDim indexLayer indexHead qBuf kCache stepCount
    let softValues = softmax rawScores (V.length rawScores)
        headScores = V.toList softValues

    -- Reuse a mutable vector for accumulation
    outBuf <- MV.new headDim
    buildActivationInPlace net indexLayer indexHead headDim headScores vCache outBuf
    V.freeze outBuf

-- Math utils
matrixVectorMult :: V.Vector Float -> Int -> Int -> V.Vector Float -> V.Vector Float
matrixVectorMult flatMat nrows ncols vec = V.generate nrows $ \row ->
    let rowStart = row * ncols
        rowVec = V.slice rowStart ncols flatMat
    in dotProduct rowVec vec

-- Multiply a flat matrix by a mutable input vector, writing into an output mutable vector
matrixVectorMultInPlaceMV
  :: V.Vector Float    -- ^ Flat matrix (nrows * ncols)
  -> Int               -- ^ nrows
  -> Int               -- ^ ncols
  -> MVectorFloat      -- ^ Input vector (mutable)
  -> MVectorFloat      -- ^ Output vector (mutable)
  -> IO ()
matrixVectorMultInPlaceMV flatMat nrows ncols vecM result = do
  -- For each row
  forM_ [0 .. nrows - 1] $ \row -> do
    let rowStart = row * ncols
        -- local references to avoid repeated lookups
        flat = flatMat
        rs = rowStart
    -- accumulate dot product in a strict loop
    let loop !i !acc
          | i >= ncols = return acc
          | otherwise = do
              v <- MV.unsafeRead vecM i
              let m = V.unsafeIndex flat (rs + i)
              loop (i + 1) (acc + v * m)
    s <- loop 0 0.0
    MV.unsafeWrite result row s

dotProduct :: V.Vector Float -> V.Vector Float -> Float
dotProduct vec1 vec2 = V.sum $ V.zipWith (*) vec1 vec2

rmsNorm :: V.Vector Float -> V.Vector Float -> V.Vector Float
rmsNorm vector weights =
  let squareNorm = V.foldl' (\acc v -> acc + v*v) 0.0 vector
      ss = (squareNorm / fromIntegral (V.length vector)) + 1e-5
      normalized = V.map (* (1.0 / sqrt ss)) vector
  in V.zipWith (*) weights normalized

-- FFN
computeDeltaFFN :: TransformerWeighting -> Int -> V.Vector Float -> TransformerResult (V.Vector Float)
computeDeltaFFN weights indexLayer token = do
  network <- ask
  AttentionKV {ffnBuf1, ffnBuf2, ffnBufOut} <- gets id
  let hid = hiddenDim network
      d   = dim network
      silu v = v / (1.0 + exp (-v))
      rmsFFNWeight = getRow indexLayer $ rmsFfnWeight weights
      rba = rmsNorm token rmsFFNWeight

  -- thaw rba once (rba is a V.Vector), reuse it
  rbaM <- liftIO $ V.thaw rba
  -- W1 * rba  -> ffnBuf1 (length hid)
  liftIO $ matrixVectorMultInPlaceMV (getRow indexLayer (w1 weights)) hid d rbaM ffnBuf1

  -- hidden1 = silu(W1 * rba)
  liftIO $ forM_ [0 .. hid - 1] $ \i -> MV.unsafeModify ffnBuf1 silu i

  -- hidden3 = W3 * rba  -> ffnBuf2
  liftIO $ matrixVectorMultInPlaceMV (getRow indexLayer (w3 weights)) hid d rbaM ffnBuf2

  -- hidden1 *= hidden3   (in-place on ffnBuf1)
  liftIO $ forM_ [0 .. hid - 1] $ \i -> do
    h3 <- MV.unsafeRead ffnBuf2 i
    MV.unsafeModify ffnBuf1 (* h3) i

  -- result = W2 * hidden1
  -- Use ffnBuf1 directly (no freeze/thaw)
  liftIO $ matrixVectorMultInPlaceMV (getRow indexLayer (w2 weights)) d hid ffnBuf1 ffnBufOut
  liftIO $ V.freeze ffnBufOut

-- QKV
computeQKV :: TransformerWeighting -> Int -> V.Vector Float -> V.Vector Float -> V.Vector Float -> Int -> TransformerResult ()
computeQKV weights indexLayer freqCisRealRow freqCisImagRow token stepCount = do
  network <- ask
  AttentionKV {qBuf, kBuf, vBuf, keyCache, valueCache} <- gets id
  let
    d = dim network
    headDim = headDimension network
    numHeads = numAttentionHeads network
    rba = rmsNorm token (getRow indexLayer (rmsAttWeight weights))

  rbaM <- liftIO $ V.thaw rba               -- mutable copy of rba (one allocation)
  liftIO $ matrixVectorMultInPlaceMV (getRow indexLayer (wq weights)) d d rbaM qBuf
  liftIO $ matrixVectorMultInPlaceMV (getRow indexLayer (wk weights)) d d rbaM kBuf
  liftIO $ matrixVectorMultInPlaceMV (getRow indexLayer (wv weights )) d d rbaM vBuf

  liftIO $ applyRotationsToBuf qBuf freqCisRealRow freqCisImagRow numHeads headDim
  liftIO $ applyRotationsToBuf kBuf freqCisRealRow freqCisImagRow numHeads headDim
  liftIO $ forM_ [0 .. numHeads - 1] $ \h -> do
    let kSlice = MV.slice (h * headDim) headDim kBuf
    let vSlice = MV.slice (h * headDim) headDim vBuf
    writeHeadToCacheMV network keyCache indexLayer h stepCount kSlice
    writeHeadToCacheMV network valueCache indexLayer h stepCount vSlice

-- Layer
createLayerToken :: Int -> Int -> V.Vector Float -> V.Vector Float -> V.Vector Float -> TransformerResult (V.Vector Float)
createLayerToken stepCount indexLayer freqCisRealRow freqCisImagRow token = do
    network <- ask
    AttentionKV {qBuf, keyCache, valueCache} <- gets id
    let
      weights = weighting network
      headDim = headDimension network
      numHeads = numAttentionHeads network
    computeQKV weights indexLayer freqCisRealRow freqCisImagRow token stepCount
    activations <- liftIO $ multiheadActivation network numHeads headDim stepCount qBuf keyCache valueCache indexLayer
    let
      deltaTokenQKV = matrixVectorMult (getRow indexLayer (wo weights)) (dim network) (dim network) (V.concat activations)
      token' = V.zipWith (+) token deltaTokenQKV

    deltaTokenFFN <- computeDeltaFFN weights indexLayer token'
    return $ V.zipWith (+) token' deltaTokenFFN

-- Transformer step
transformer :: Token -> Int -> TransformerResult (V.Vector Float)
transformer tokenCode stepCount = do
    network <- ask

    -- Getting the token embedding
    let weights = weighting network
        rowStart = fromIntegral tokenCode * tokenEmbeddingTableCols weights
        token = V.slice rowStart (tokenEmbeddingTableCols weights) (tokenEmbeddingTable weights)

    -- Plucking out the current row of freq_cis_real and freq_cis_imag
    let freqCisRealRow = getRow stepCount (freqCisReal weights)
    let freqCisImagRow = getRow stepCount (freqCisImag weights)

    -- Forwarding all the layers
    finalToken <- foldM (\accToken indexLayer -> createLayerToken stepCount indexLayer freqCisRealRow freqCisImagRow accToken)
                  token
                  [0..nLayers network - 1]

    -- Final rmsnorm
    let tokenWithRms = rmsNorm finalToken (rmsFinalWeight weights)

    -- Classifier into logits
    let logits = matrixVectorMult (tokenEmbeddingTable weights) (tokenEmbeddingTableRows weights) (tokenEmbeddingTableCols weights) tokenWithRms

    return logits

generateNextToken :: Int -> PromptTokens -> Float -> Vocabulary -> Token -> Int -> TransformerResult (BS.ByteString, Token)
generateNextToken timestep promptTokens temperature vocab tokenCode seedValue = do
  network <- ask
  logits <- transformer tokenCode timestep
  nextToken <- if timestep < length promptTokens
    then return (promptTokens !! timestep)
    else if temperature == 0.0
      then return $ fromIntegral (V.maxIndex logits)
    else do
      liftIO $ drawSample seedValue $ softmax (V.map (/ temperature) logits) (vocabSize network)
  let
    word = vocab !! fromIntegral nextToken :: BS.ByteString
    firstChar = BSC.head word :: Char
    tokenStr = if tokenCode == 1 && isSpace firstChar
          then BSC.tail (vocab !! fromIntegral nextToken)
          else vocab !! fromIntegral nextToken
  return (tokenStr, nextToken)

generateTokens :: Int -> PromptTokens -> Float -> Vocabulary -> Int -> TransformerResult ([BS.ByteString], Int)
generateTokens maxSteps promptTokens temperature vocab seedValue = do
  network <- ask
  go network 0 [] 1 where
    go network timestep result token
      | timestep >= maxSteps || (timestep /= 0 && token == 1) = return (result, timestep)
      | otherwise = do
        (tokenStr, nextToken) <- generateNextToken timestep promptTokens temperature vocab token seedValue
        liftIO $ putStr $ BSC.unpack tokenStr
        liftIO $ hFlush stdout
        go network (timestep + 1) (result ++ [tokenStr]) nextToken

-- Initialize flat attention KV caches (flattened to one MVector each)
initAttentionKV :: NetworkConfig -> IO AttentionKV
initAttentionKV NetworkConfig {nLayers, numAttentionHeads, seqLen, headDimension, hiddenDim, dim} = do
  let size = nLayers * numAttentionHeads * seqLen * headDimension
  keyCache   <- MV.new size
  valueCache <- MV.new size
  ffnBuf1    <- MV.new hiddenDim
  ffnBuf2    <- MV.new hiddenDim
  ffnBufOut  <- MV.new dim
  qBuf       <- MV.new dim
  kBuf       <- MV.new dim
  vBuf       <- MV.new dim
  return AttentionKV { keyCache, valueCache, ffnBuf1, ffnBuf2, ffnBufOut, qBuf, kBuf, vBuf }

run :: BS.ByteString -> BS.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
run modelFileContent tokenizerFileContent temperature steps prompt seed = do
  currentTime <- getPOSIXTime
  let
    seedValue = fromMaybe (round currentTime) seed
    config = initModel modelFileContent
    prompt' = fromMaybe "" prompt
    (promptTokens, vocab) = tokenizerInit tokenizerFileContent (vocabSize config) (BSC.pack prompt')
  attentionKV <- initAttentionKV config
  putStrLn "<s>"
  startTime <- getPOSIXTime
  (_, countTokens) <- evalStateT (runReaderT (generateTokens steps promptTokens temperature vocab seedValue) config) attentionKV
  endTime <- getPOSIXTime
  let
    duration :: Integer
    duration = round (endTime - startTime)
    tokensPerSec :: Float
    tokensPerSec = fromIntegral countTokens / fromIntegral duration
  printf "\nduration: %ds - (%.02f tokens/s)\n" duration tokensPerSec
  return ()
