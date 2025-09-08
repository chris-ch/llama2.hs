{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE NamedFieldPuns #-}
module Main (main) where

import qualified Options.Applicative as OA
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.Binary.Get as BG
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV
import qualified Data.List as DL
import qualified Data.List.Split as DLS
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
import Data.Vector.Unboxed (Vector)
import System.IO (hFlush, stdout)

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
    { tokenEmbeddingTable :: Vector Float
    , tokenEmbeddingTableRows :: Int
    , tokenEmbeddingTableCols :: Int
    , rmsAttWeight :: [Vector Float]
    , wq :: [Vector Float]
    , wk :: [Vector Float]
    , wv :: [Vector Float]
    , wo :: [Vector Float]
    , rmsFfnWeight :: [Vector Float]
    , w1 :: [Vector Float]
    , w2 :: [Vector Float]
    , w3 :: [Vector Float]
    , rmsFinalWeight :: Vector Float
    , freqCisReal :: [Vector Float]
    , freqCisImag :: [Vector Float]
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

readVector :: Int -> BG.Get (Vector Float)
readVector count = do
    values <- replicateM count getFloatle
    return $ V.fromList values

readVectors :: Int -> Int -> BG.Get [Vector Float]
readVectors nrows ncols = replicateM nrows (readVector ncols)

readMatrices :: Int -> Int -> Int -> BG.Get [[Vector Float]]
readMatrices ndepth nrows ncols = replicateM ndepth (readVectors nrows ncols)

parseNetworkConfigFile :: BG.Get NetworkConfig
parseNetworkConfigFile = do
        dim' <- fromIntegral <$> getInt32le
        hiddenDim' <- fromIntegral <$> getInt32le
        nLayers' <- fromIntegral <$> getInt32le
        numAttentionHeads' <- fromIntegral <$> getInt32le
        numKeyValueHeads' <- fromIntegral <$> getInt32le
        vocabSize' <- fromIntegral <$> getInt32le
        seqLen' <- fromIntegral <$> getInt32le
        tokenEmbeddingTable' <- readVectors vocabSize' dim'
        let tokenEmbeddingTableFlat = V.concat tokenEmbeddingTable'
        rmsAttWeight' <- readVectors nLayers' dim'
        wq' <- readMatrices nLayers' dim' dim'
        let wqFlat = map V.concat wq'
        wk' <- readMatrices nLayers' dim' dim'
        let wkFlat = map V.concat wk'
        wv' <- readMatrices nLayers' dim' dim'
        let wvFlat = map V.concat wv'
        wo' <- readMatrices nLayers' dim' dim'
        let woFlat = map V.concat wo'
        rmsFfnWeight' <- readVectors nLayers' dim'
        w1' <- readMatrices nLayers' hiddenDim' dim'
        let w1Flat = map V.concat w1'
        w2' <- readMatrices nLayers' dim' hiddenDim'
        let w2Flat = map V.concat w2'
        w3' <- readMatrices nLayers' hiddenDim' dim'
        let w3Flat = map V.concat w3'
        rmsFinalWeight' <- readVector dim'
        freqCisReal' <- readVectors seqLen' ((dim' `div` numAttentionHeads') `div` 2)
        freqCisImag' <- readVectors seqLen' ((dim' `div` numAttentionHeads') `div` 2)

        let
            headDim = dim' `div` numAttentionHeads'
            weights = TransformerWeighting
              { tokenEmbeddingTable = tokenEmbeddingTableFlat
              , tokenEmbeddingTableRows = vocabSize'
              , tokenEmbeddingTableCols = dim'
              , rmsAttWeight = rmsAttWeight'
              , wq = wqFlat
              , wk = wkFlat
              , wv = wvFlat
              , wo = woFlat
              , rmsFfnWeight = rmsFfnWeight'
              , w1 = w1Flat
              , w2 = w2Flat
              , w3 = w3Flat
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

-- Helper: write/read single head vector to/from the flat cache
writeHeadToCache :: NetworkConfig -> MVectorFloat -> Int -> Int -> Int -> V.Vector Float -> IO ()
writeHeadToCache net cache layer numHead step headVec = do
  let headDim = headDimension net
      offset = cacheIndex net layer numHead step 0
  mk <- V.thaw headVec
  MV.copy (MV.slice offset headDim cache) mk

-- Build activation from value cache
buildActivation :: NetworkConfig -> Int -> Int -> Int -> [Float] -> MVectorFloat -> IO (V.Vector Float)
buildActivation net indexLayer indexHead headDim headScores vCache = do
  let numPos = length headScores
      zero = V.replicate headDim 0.0
      addScaled acc pos = do
        let offset = cacheIndex net indexLayer indexHead pos 0
        vVec <- V.freeze (MV.slice offset headDim vCache)
        let scaled = V.map (* (headScores !! pos)) vVec
        return $ V.zipWith (+) acc scaled
  foldM addScaled zero [0 .. numPos - 1]

-- Rotary
applyRotations :: V.Vector Float -> V.Vector Float -> V.Vector Float -> V.Vector Float
applyRotations headVector freqCisRealRow freqCisImagRow =
  V.fromList $ concatMap applyRotation [0,2..V.length headVector - 2]
  where
    applyRotation :: Int -> [Float]
    applyRotation headItemIndex = [v * real - v' * imag, v * imag + v' * real]
      where
        real = freqCisRealRow V.! (headItemIndex `div` 2)
        imag = freqCisImagRow V.! (headItemIndex `div` 2)
        v = headVector V.! headItemIndex
        v' = headVector V.! (headItemIndex + 1)

-- Math utils
matrixVectorMult :: Vector Float -> Int -> Int -> V.Vector Float -> V.Vector Float
matrixVectorMult flatMat nrows ncols vec = V.generate nrows $ \row ->
    let rowStart = row * ncols
        rowVec = V.slice rowStart ncols flatMat
    in dotProduct rowVec vec

matrixVectorMultInPlace :: V.Vector Float -> Int -> Int -> V.Vector Float -> MVectorFloat -> IO ()
matrixVectorMultInPlace flatMat nrows ncols vec result = do
  forM_ [0 .. nrows - 1] $ \row -> do
    let rowStart = row * ncols
    let rowVec = V.slice rowStart ncols flatMat
    let s = dotProduct rowVec vec
    MV.write result row s

splitVector :: Int -> V.Vector Float -> [V.Vector Float]
splitVector m vec = V.fromList <$> DLS.chunksOf (V.length vec `div` m) (V.toList vec)

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
      rmsFFNWeight = rmsFfnWeight weights !! indexLayer
      rba = rmsNorm token rmsFFNWeight

  -- hidden1 = silu(W1 * rba)
  liftIO $ matrixVectorMultInPlace (w1 weights !! indexLayer) hid d rba ffnBuf1
  liftIO $ forM_ [0 .. hid - 1] $ \i -> MV.modify ffnBuf1 silu i

  -- hidden3 = W3 * rba
  liftIO $ matrixVectorMultInPlace (w3 weights !! indexLayer) hid d rba ffnBuf2

  -- hidden1 *= hidden3
  liftIO $ forM_ [0 .. hid - 1] $ \i -> do
    h3 <- MV.read ffnBuf2 i
    MV.modify ffnBuf1 (* h3) i

  -- result = W2 * hidden1
  productV <- liftIO $ V.freeze ffnBuf1
  liftIO $ matrixVectorMultInPlace (w2 weights !! indexLayer) d hid productV ffnBufOut
  liftIO $ V.freeze ffnBufOut

-- QKV
computeQKV :: TransformerWeighting -> Int -> Int -> V.Vector Float -> V.Vector Float -> V.Vector Float -> TransformerResult ([V.Vector Float], [V.Vector Float], [V.Vector Float])
computeQKV weights numHeads indexLayer freqCisRealRow freqCisImagRow token = do
  network <- ask
  AttentionKV {qBuf, kBuf, vBuf} <- gets id
  let
    d = dim network
    rba = rmsNorm token (rmsAttWeight weights !! indexLayer)
  liftIO $ matrixVectorMultInPlace (wq weights !! indexLayer) d d rba qBuf
  liftIO $ matrixVectorMultInPlace (wk weights !! indexLayer) d d rba kBuf
  liftIO $ matrixVectorMultInPlace (wv weights !! indexLayer) d d rba vBuf
  wQ <- liftIO $ V.freeze qBuf
  wK <- liftIO $ V.freeze kBuf
  wV <- liftIO $ V.freeze vBuf
  let
    headsQ = splitVector numHeads wQ
    headsQ' = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) headsQ
    headsK = splitVector numHeads wK
    headsK' = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) headsK
    headsV = splitVector numHeads wV
  return (headsQ', headsK', headsV)

-- Attention scores
computeScores :: NetworkConfig -> Int -> Int -> Int -> [V.Vector Float] -> MVectorFloat -> Int -> IO (V.Vector Float)
computeScores net headDim indexLayer indexHead headsQ kCache stepCount = do
  let
    qHead = headsQ !! indexHead
    sqrtHead = sqrt (fromIntegral headDim)
    numPos = stepCount + 1
  V.generateM numPos $ \pos -> do
    let offset = cacheIndex net indexLayer indexHead pos 0
    kVec <- V.freeze (MV.slice offset headDim kCache)
    return $ dotProduct kVec qHead / sqrtHead

-- Multihead
multiheadActivation :: NetworkConfig -> Int -> Int -> Int -> [V.Vector Float] -> MVectorFloat -> MVectorFloat -> Int -> IO [V.Vector Float]
multiheadActivation net numHeads headDim stepCount headsQ kCache vCache indexLayer = do
  forM [0 .. numHeads - 1] $ \indexHead -> do
    rawScores <- computeScores net headDim indexLayer indexHead headsQ kCache stepCount
    let
      softValues = softmax rawScores (V.length rawScores)
      headScores = V.toList softValues
    buildActivation net indexLayer indexHead headDim headScores vCache

-- Layer
createLayerToken :: Int -> Int -> V.Vector Float -> V.Vector Float -> V.Vector Float -> TransformerResult (V.Vector Float)
createLayerToken stepCount indexLayer freqCisRealRow freqCisImagRow token = do
    network <- ask
    AttentionKV {keyCache, valueCache} <- gets id
    let
      weights = weighting network
      headDim = headDimension network
      numHeads = numAttentionHeads network
    (headsQ, headsK, headsV) <- computeQKV weights (numAttentionHeads network) indexLayer freqCisRealRow freqCisImagRow token
    -- write K and V into the flat caches
    liftIO $ forM_ [0 .. numHeads - 1] $ \h -> do
      writeHeadToCache network keyCache indexLayer h stepCount (headsK !! h)
      writeHeadToCache network valueCache indexLayer h stepCount (headsV !! h)

    activations <- liftIO $ multiheadActivation network numHeads headDim stepCount headsQ keyCache valueCache indexLayer
    let
      deltaTokenQKV = matrixVectorMult (wo weights !! indexLayer) (dim network) (dim network) (V.concat activations)
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
    let freqCisRealRow = freqCisReal weights !! stepCount
    let freqCisImagRow = freqCisImag weights !! stepCount

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