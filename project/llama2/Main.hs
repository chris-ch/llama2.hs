module Main (main) where

import Control.Monad (replicateM)
import Control.Monad.Reader (ReaderT (runReaderT))
import Control.Monad.State (evalStateT)
import qualified Data.Binary.Get as BG
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.List as DL
import Data.Maybe (fromMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)
import qualified Data.Vector.Unboxed.Mutable as MV
import qualified Options.Applicative as OA
import Text.Printf (printf)
import Transformer (generateTokens)
import Types (PromptTokens, StepCount (..), Token (..), Vocabulary, VocabularyScores)
import Architecture (NetworkConfig (..), TransformerLayerComponent (..),
  MultiHeadAttentionComponent (..), TransformerDecoderComponent (..), DecoderCache (..), LayerAttentionCache (..), HeadCache (..), parseNetworkConfigFile, TransformerResult (..),
  )

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
  runModel modelFileContent tokenizerFileContent (realToFrac temperature) steps prompt seed

--------------------------------------------------------------------------------
-- Binary Parsing
--------------------------------------------------------------------------------

initDecoderCaches :: NetworkConfig -> IO DecoderCache
initDecoderCaches NetworkConfig{numLayers, seqLen, headDimension, decoder} = do
  let numHeads = length (heads (multiHeadAttention (head (modelLayers decoder))))
      cacheSize = seqLen * headDimension
  layerCachesList <- replicateM numLayers $ do
    headCachesList <- replicateM numHeads $ do
      kc <- MV.new cacheSize
      vc <- MV.new cacheSize
      return $ HeadCache { headKeyCache = kc, headValueCache = vc }
    return $ LayerAttentionCache { headCaches = headCachesList }
  return $ DecoderCaches { layerCaches = layerCachesList }

initModel :: BS.ByteString -> NetworkConfig
initModel = BG.runGet parseNetworkConfigFile

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

runModel :: BS.ByteString -> BS.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
runModel modelFileContent tokenizerFileContent temperature steps prompt seed = do
  currentTime <- getPOSIXTime
  let seedValue = fromMaybe (round currentTime) seed
      config = initModel modelFileContent
      prompt' = fromMaybe "" prompt
      (promptTokens, vocab) = tokenizerInit tokenizerFileContent (vocabSize config) (BSC.pack prompt')
  attentionKV <- initDecoderCaches config
  putStrLn "<s>"
  startTime <- getPOSIXTime
  (_, StepCount countTokens) <- evalStateT (runReaderT (runTransformerResult (generateTokens (StepCount steps) promptTokens temperature vocab seedValue)) config) attentionKV
  endTime <- getPOSIXTime
  let duration :: Integer
      duration = round (endTime - startTime)
      tokensPerSec :: Float
      tokensPerSec = fromIntegral countTokens / fromIntegral duration
  printf "\nduration: %ds - (%.02f tokens/s)\n" duration tokensPerSec
  return ()
