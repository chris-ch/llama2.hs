{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}
module Main (main) where

import qualified Options.Applicative as OA
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.Binary.Get as BG
import qualified Data.Vector.Unboxed as V
import qualified Data.List as DL
import qualified Data.List.Split as DLS
import qualified System.Random as R

import Control.Monad (replicateM, foldM)
import Control.Monad.State ( StateT, evalStateT, MonadState(put), gets )
import Control.Monad.Reader ( MonadIO(liftIO), ReaderT(runReaderT), MonadReader(ask) )
import Data.Maybe (fromMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)
import GHC.Unicode (isSpace)
import Text.Printf (printf)
import Data.Binary.Get (getInt32le, getFloatle)
import Data.Int (Int32)
import Data.Vector.Unboxed (Vector)
import System.IO (hFlush, stdout)


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

type Matrix a = [Vector a] -- Matrix as row vectors
type KeyCache = [[Matrix Float]]
type ValueCache = [[Matrix Float]]
type Vocabulary = [BS.ByteString]
type VocabularyScores = [Float]
type Token = Int32
type PromptTokens = [Token]

data AttentionKV = AttentionKV
    { keyCache :: KeyCache
    , valueCache :: ValueCache
    } deriving (Show)

data TransformerWeighting = TransformerWeighting
    { tokenEmbeddingTable :: Matrix Float
    , rmsAttWeight :: [Vector Float]
    , wq :: [Matrix Float]
    , wk :: [Matrix Float]
    , wv :: [Matrix Float]
    , wo :: [Matrix Float]
    , rmsFfnWeight :: [Vector Float]
    , w1 :: [Matrix Float]
    , w3 :: [Matrix Float]
    , w2 :: [Matrix Float]
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

readVector :: Int -> BG.Get (Vector Float)
readVector count = do
    values <- replicateM count getFloatle
    return $ V.fromList values

readVectors :: Int -> Int -> BG.Get [Vector Float]
readVectors nrows ncols = replicateM nrows (readVector ncols)

readMatrices :: Int -> Int -> Int -> BG.Get [Matrix Float]
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
        rmsAttWeight' <- readVectors nLayers' dim'
        wq' <- readMatrices nLayers' dim' dim'
        wk' <- readMatrices nLayers' dim' dim'
        wv' <- readMatrices nLayers' dim' dim'
        wo' <- readMatrices nLayers' dim' dim'
        rmsFfnWeight' <- readVectors nLayers' dim'
        w1' <- readMatrices nLayers' hiddenDim' dim'
        w2' <- readMatrices nLayers' dim' hiddenDim'
        w3' <- readMatrices nLayers' hiddenDim' dim'
        rmsFinalWeight' <- readVector dim'
        freqCisReal' <- readVectors seqLen' ((dim' `div` numAttentionHeads') `div` 2)
        freqCisImag' <- readVectors seqLen' ((dim' `div` numAttentionHeads') `div` 2)

        let
            headDim = dim' `div` numAttentionHeads'
            weights = TransformerWeighting
              { tokenEmbeddingTable = tokenEmbeddingTable'
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

type TransformerResult a = ReaderT NetworkConfig (StateT AttentionKV IO) a

softmax :: V.Vector Float -> Int -> V.Vector Float
softmax values size = V.concat [softmaxValues, V.slice size (V.length values - size) values]
  where
    maxVal = V.maximum (V.take size values)
    expValues = V.map (\x -> exp (x - maxVal)) (V.take size values)
    sumExpValues = V.sum expValues
    softmaxValues = V.map (/ sumExpValues) expValues

drawSample :: Int -> V.Vector Float -> IO Token
drawSample seedValue probabilities = do
  let
    gen = R.mkStdGen seedValue
    (r, _) = R.random gen :: (Float, R.StdGen)

    indexHighestCDF :: Float -> V.Vector Float -> Int
    indexHighestCDF rand vec = min (V.ifoldl' (indexHighest rand) 0 cdf) (V.length vec - 1)
        where
          cdf = V.scanl1 (+) vec
          indexHighest :: Float -> Int -> Int -> Float -> Int
          indexHighest rand' acc i v = if v <= rand' then i + 1 else acc

  return $ fromIntegral $ indexHighestCDF r probabilities

buildActivation :: Int -> Int -> KeyCache -> Int -> [Float] -> V.Vector Float
buildActivation dimension indexLayer vC indexHead headScores =
  DL.foldl' accumulate zeroVector zippedValues
  where
    accumulate :: V.Vector Float -> (V.Vector Float, Float) -> V.Vector Float
    accumulate acc (valueVector, attentionWeight) = V.zipWith (+) acc (scale attentionWeight valueVector)
    zeroVector = V.replicate dimension 0.0
    zippedValues = zip (map (\count -> vC !! count !! indexLayer !! indexHead) [0..]) headScores
    scale w = V.map (w *)

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

matrixVectorMult :: Matrix Float -> V.Vector Float -> V.Vector Float
matrixVectorMult mat vec = V.fromList $ map (`dotProduct` vec) mat

splitVector :: Int -> V.Vector Float -> [V.Vector Float]
splitVector m vec = V.fromList <$> DLS.chunksOf (V.length vec `div` m) (V.toList vec)

dotProduct :: V.Vector Float -> V.Vector Float -> Float
dotProduct vec1 vec2 = V.sum $ V.zipWith (*) vec1 vec2

rmsNorm :: V.Vector Float -> V.Vector Float -> V.Vector Float
rmsNorm vector weights =
  let
    squareNorm :: V.Vector Float -> Float
    squareNorm = V.foldl' cumSumSquare 0.0
      where
        cumSumSquare :: Float -> Float -> Float
        cumSumSquare acc v = acc + v ^ (2::Int)

    ss = (squareNorm vector / fromIntegral (V.length vector)) + 1e-5
    normalized = V.map (* (1.0 / sqrt ss)) vector
  in V.zipWith (*) weights normalized

computeDeltaFFN :: TransformerWeighting -> Int -> V.Vector Float -> V.Vector Float
computeDeltaFFN weights indexLayer token =
    let
      sigmoidLinearUnit :: Float -> Float
      sigmoidLinearUnit v = v / (1.0 + exp (-v))

      rmsFFNWeight = rmsFfnWeight weights !! indexLayer
      weight1 = w1 weights !! indexLayer
      weight2 = w2 weights !! indexLayer
      weight3 = w3 weights !! indexLayer
      rba = rmsNorm token rmsFFNWeight
      hiddenDimensionBuffer1 = matrixVectorMult weight1 rba
      hiddenDimensionBuffer2 = matrixVectorMult weight3 rba
      sigmoided = V.map sigmoidLinearUnit hiddenDimensionBuffer1
    in matrixVectorMult weight2 (V.zipWith (*) sigmoided hiddenDimensionBuffer2)

computeQKV :: TransformerWeighting -> Int -> Int -> V.Vector Float -> V.Vector Float -> V.Vector Float -> ([V.Vector Float], [V.Vector Float], [V.Vector Float])
computeQKV weights numHeads indexLayer freqCisRealRow freqCisImagRow token =
  let
    rba = rmsNorm token (rmsAttWeight weights !! indexLayer)
    wQ = splitVector numHeads (matrixVectorMult (wq weights !! indexLayer) rba)
    headsQ = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) wQ
    wK = splitVector numHeads (matrixVectorMult (wk weights !! indexLayer) rba)
    headsK = map (\vector -> applyRotations vector freqCisRealRow freqCisImagRow) wK
    headsV = splitVector numHeads (matrixVectorMult (wv weights !! indexLayer) rba)
  in
    (headsQ, headsK, headsV)

computeScores :: Int -> KeyCache -> Int -> Int -> [V.Vector Float] -> V.Vector Float
computeScores headDim kC indexLayer indexHead headsQ = V.fromList $ map calculateScore kC
  where
    calculateScore :: [[V.Vector Float]] -> Float
    calculateScore keyVectors =
      let keyVector = ((keyVectors !! indexLayer) !! indexHead)
      in dotProduct (headsQ !! indexHead) keyVector / sqrt (fromIntegral headDim)

multiheadActivation :: Int -> Int -> Int -> KeyCache-> ValueCache -> [V.Vector Float] -> Matrix Float
multiheadActivation numHeads headDim indexLayer kC vC headsQ =
    [buildActivation headDim indexLayer vC indexHead (scores indexHead) | indexHead <- [0 .. numHeads - 1]]
    where
      scores indexHead = V.toList $ softmax rawScores (V.length rawScores)
        where
          rawScores = computeScores headDim kC indexLayer indexHead headsQ

createLayerToken :: Int -> Int -> V.Vector Float -> V.Vector Float -> V.Vector Float -> TransformerResult (V.Vector Float)
createLayerToken stepCount indexLayer freqCisRealRow freqCisImagRow token = do
    network <- ask
    (kC, vC) <- gets (\cache -> (keyCache cache, valueCache cache))
    let
        (headsQ, headsK, headsV) = computeQKV (weighting network) (numAttentionHeads network) indexLayer freqCisRealRow freqCisImagRow token
        keyCacheStep = (kC !! stepCount) ++ [headsK]
        valueCacheStep = (vC !! stepCount) ++ [headsV]
        keyCache' = take stepCount kC ++ [keyCacheStep]
        valueCache' = take stepCount vC ++ [valueCacheStep]
        activations = multiheadActivation (numAttentionHeads network) (headDimension network) indexLayer keyCache' valueCache' headsQ
        wO = wo (weighting network)
        deltaTokenQKV = matrixVectorMult (wO !! indexLayer) (V.concat activations)
        token' = V.zipWith (+) token deltaTokenQKV
        deltaTokenFFN = computeDeltaFFN (weighting network) indexLayer token'
        result = V.zipWith (+) token' deltaTokenFFN
    put (AttentionKV {keyCache = keyCache', valueCache = valueCache'})
    return result

transformer :: Token -> Int -> TransformerResult (V.Vector Float)
transformer tokenCode stepCount = do
    network <- ask

    -- Getting the token embedding
    let token = tokenEmbeddingTable (weighting network) !! fromIntegral tokenCode

    -- Plucking out the current row of freq_cis_real and freq_cis_imag
    let freqCisRealRow = freqCisReal (weighting network) !! stepCount
    let freqCisImagRow = freqCisImag (weighting network) !! stepCount

    -- Forwarding all the layers
    finalToken <- foldM (\accToken indexLayer -> createLayerToken stepCount indexLayer freqCisRealRow freqCisImagRow accToken)
                  token
                  [0..nLayers network - 1]

    -- Final rmsnorm
    let tokenWithRms = rmsNorm finalToken (rmsFinalWeight $ weighting network)

    -- Classifier into logits
    let logits = matrixVectorMult (tokenEmbeddingTable (weighting network)) tokenWithRms

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
        (kC, vC) <- gets (\cache -> (keyCache cache, valueCache cache))
        put (AttentionKV {keyCache = take timestep kC ++ [[]], valueCache = take timestep vC ++ [[]]})
        (tokenStr, nextToken) <- generateNextToken timestep promptTokens temperature vocab token seedValue
        liftIO $ putStr $ BSC.unpack tokenStr
        liftIO $ hFlush stdout
        go network (timestep + 1) (result ++ [tokenStr]) nextToken

run :: BS.ByteString -> BS.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
run modelFileContent tokenizerFileContent temperature steps prompt seed = do
  currentTime <- getPOSIXTime
  let
    seedValue = fromMaybe (round currentTime) seed
    config = initModel modelFileContent
    prompt' = fromMaybe "" prompt
    (promptTokens, vocab) = tokenizerInit tokenizerFileContent (vocabSize config) (BSC.pack prompt')
    initStateAttentionKV :: AttentionKV
    initStateAttentionKV = AttentionKV { keyCache = [], valueCache = [] }
  putStrLn "<s>"
  startTime <- getPOSIXTime
  (_, countTokens) <- evalStateT (runReaderT (generateTokens steps promptTokens temperature vocab seedValue) config) initStateAttentionKV
  endTime <- getPOSIXTime
  let
    duration :: Integer
    duration = round (endTime - startTime)
    tokensPerSec :: Float
    tokensPerSec = fromIntegral countTokens / fromIntegral duration
  printf "\nduration: %ds - (%.02f tokens/s)\n" duration tokensPerSec
  return ()
