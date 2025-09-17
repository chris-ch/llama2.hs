module Main (main) where

import qualified Data.Binary.Get as BG
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.List as DL
import qualified Options.Applicative as OA
import qualified Foreign as F
import Control.Monad.Reader (ReaderT(runReaderT), MonadIO(liftIO), MonadReader(ask))
import Control.Monad.State (evalStateT )
import Data.Maybe (fromMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)
import GHC.Unicode (isSpace)
import System.IO (hFlush, stdout)
import Control.Monad (replicateM)
import Text.Printf (printf)

import Helpers
import Model
import qualified Clash.Explicit.Prelude as C
import qualified Clash.Explicit.SimIO as CES
import qualified Clash.Clocks as CC
import qualified Clash.Signal as CS (simulate, withClockResetEnable)

type Vocabulary = [BS.ByteString]
type VocabularyScores = [Float]
newtype Token = Token F.Int32 deriving (Show, Eq, Ord, Num)
type PromptTokens = [Helpers.Token]


-- example model config 110M: 
-- modelDim = 768
-- hiddenDim = 2048
--numLayers = 12
-- numAttentionHeads = 12
-- numKeyValueHeads = 12
-- vocabSize = 32000
-- seqLen = 1024
-- headDimension = 64

--------------------------------------------------------------------------------
-- Data Structure and Class
--------------------------------------------------------------------------------

-- Generic data definitions

--------------------------------------------------------------------------------
-- LLM cache
--------------------------------------------------------------------------------


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
    <*> OA.strOption (OA.long "model-file" <> OA.value "./data/stories110M.bin" <> OA.metavar "MODEL_FILE" <> OA.help "Model binary file")
    <*> OA.option OA.auto (OA.long "temperature" <> OA.value 0.0 <> OA.metavar "TEMPERATURE" <> OA.help "Temperature")
    <*> OA.option OA.auto (OA.long "steps" <> OA.value 256 <> OA.metavar "STEPS" <> OA.help "Number of steps")
    <*> OA.optional (OA.strArgument (OA.metavar "PROMPT" <> OA.help "Initial prompt"))


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

applyBPEMerges :: [Helpers.Token] -> Vocabulary -> VocabularyScores -> PromptTokens
applyBPEMerges tokens vocab vocabScores = case findBestPair tokens of
  Just (bestIndex, bestToken) ->
    applyBPEMerges (mergePair bestIndex bestToken tokens) vocab vocabScores
  Nothing ->
    tokens
  where
    findBestPair :: [Helpers.Token] -> Maybe (Int, Helpers.Token)
    findBestPair tokens' = foldr checkPair Nothing (zip [0 ..] (zip tokens' (drop 1 tokens')))
      where
        checkPair :: (Int, (Helpers.Token, Helpers.Token)) -> Maybe (Int, Helpers.Token) -> Maybe (Int, Helpers.Token)
        checkPair (count, (Helpers.Token tokenPrev, Helpers.Token tokenNext)) acc =
          case strLookup ((vocab !! fromIntegral tokenPrev) `BS.append` (vocab !! fromIntegral tokenNext)) vocab of
            pos | pos /= -1 && vocabScores !! pos > bestScore -> Just (count, fromIntegral pos)
            _ -> acc

        bestScore :: Float
        bestScore = -1e10

    mergePair :: Int -> Helpers.Token -> [Helpers.Token] -> [Helpers.Token]
    mergePair count token tokens' =
      take count tokens' ++ [token] ++ drop (count + 2) tokens'

bpeEncode :: BS.ByteString -> Vocabulary -> VocabularyScores -> PromptTokens
bpeEncode prompt vocab vocabScores =
  let tokens = map (\char -> fromMaybe (error "Character not found in vocabulary") (DL.elemIndex (BS.pack [char]) vocab)) (BS.unpack prompt)
   in applyBPEMerges (map fromIntegral tokens) vocab vocabScores

runModel :: BS.ByteString -> BS.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
runModel modelFileContent tokenizerFileContent temperature steps prompt seed = do
  currentTime <- getPOSIXTime
  let
    seedValue = fromMaybe (round currentTime) seed
    initModel :: BS.ByteString -> TransformerDecoderComponent
    initModel = BG.runGet parseModelConfigFile
    config = initModel modelFileContent
    prompt' = fromMaybe "" prompt
    (promptTokens, vocab) = tokenizerInit tokenizerFileContent (fromIntegral vocabSize) (BSC.pack prompt')

  putStrLn "<s>"
  startTime <- getPOSIXTime

  (_, StepCount countTokens) <- generateTokens config (StepCount (fromIntegral steps)) promptTokens temperature vocab seedValue

  endTime <- getPOSIXTime
  let duration :: Integer
      duration = round (endTime - startTime)
      tokensPerSec :: Float
      tokensPerSec = fromIntegral countTokens / fromIntegral duration
  printf "\nduration: %ds - (%.02f tokens/s)\n" duration tokensPerSec
  return ()

--------------------------------------------------------------------------------
-- Token Generation with Clash Simulation
--------------------------------------------------------------------------------

generateTokens :: (MonadIO m) => TransformerDecoderComponent -> StepCount -> PromptTokens -> Float -> Vocabulary -> Int -> m ([BSC.ByteString], StepCount)
generateTokens decoder (StepCount maxSteps) promptTokens temperature vocab seedValue = do
  -- Initialize attention caches for all layers with System domain
  let initCaches = C.repeat initAttentionCache :: C.Vec NumLayers (AttentionCache C.System)
      -- Wrap transformer to take a Signal System (Unsigned 32, Helpers.Token) as input
      transformerWrapper (timestep, token) =
        C.sampleN 1 (transformer decoder initCaches (fromIntegral timestep)
                                token temperature seedValue promptTokens) !! 0
      -- Generate tokens iteratively
      go timestep result token
        | timestep >= maxSteps || (timestep /= 0 && token == Helpers.Token 1) = pure (result, StepCount timestep)
        | otherwise = do
            -- Simulate transformer for one step
            let input = (timestep :: C.Unsigned 32, token)
                [nextToken] = C.simulate transformerWrapper [input]
                tokenStr = vocab !! fromIntegral nextToken
            -- Output token
            liftIO $ putStr $ BSC.unpack tokenStr
            liftIO $ hFlush stdout
            go (timestep + 1) (result ++ [tokenStr]) nextToken
  (tokens, finalStep) <- go 0 [] (Helpers.Token 1)
  pure (tokens, finalStep)
