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
import qualified Clash.Signal as CS
import qualified Clash.Sized.Vector as CSV

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
    (promptTokens, vocab) = tokenizerInit tokenizerFileContent vocabSize (BSC.pack prompt')

  putStrLn "<s>"
  startTime <- getPOSIXTime

  (_, StepCount countTokens) <- generateTokensSimAutoregressive config (fromIntegral steps) promptTokens temperature seedValue

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

-- This version feeds the output of each step as input to the next step
generateTokensSimAutoregressive
  :: TransformerDecoderComponent
  -> C.Unsigned 32                      -- ^ number of steps to generate
  -> [Helpers.Token]                   -- ^ prompt tokens
  -> Float                     -- ^ temperature
  -> Int                       -- ^ seed
  -> IO ([Helpers.Token], StepCount)   -- ^ produced tokens and token count
generateTokensSimAutoregressive decoder nSteps promptTokens temperature seed = do
  let promptLen = length promptTokens
      totalSteps = promptLen + fromIntegral nSteps
      
      -- For autoregressive generation, we need to carefully construct the input sequence
      -- The sequence position increases, and each input token is either from prompt or previously generated
      seqPositions = [0..totalSteps-1]
      
      -- Create temperature and seed signals
      tempSig = replicate totalSteps temperature
      rngSeeds = [seed + i | i <- [0..totalSteps-1]]
      
      -- Create prompt vector (same for all steps in this simple version)
      promptPadded = take seqLen (promptTokens ++ repeat (Helpers.Token 0))
      promptVec = replicate totalSteps (CSV.unsafeFromList promptPadded)
      
      -- For the input tokens, we need to simulate the autoregressive process
      -- This is a simplified version - in practice you'd need to feed outputs back as inputs
      inputTokens = take totalSteps (promptTokens ++ repeat (Helpers.Token 0))

      -- Bundle all inputs together as tuples for CS.simulate
      bundledInputs = zip5 seqPositions inputTokens tempSig rngSeeds promptVec

  -- Run simulation with bundled inputs
  let allOutputs = CS.simulate (topEntityBundled @CS.System decoder) bundledInputs

  -- Extract only the generated tokens (after the prompt)
  let generatedTokens = drop promptLen (take totalSteps allOutputs)
      resultTokens = take (fromIntegral nSteps) generatedTokens

  -- Print generation progress
  putStrLn $ "Prompt: " ++ show promptTokens
  putStr "Generated: "
  mapM_ (\t -> putStr (show t ++ " ") >> hFlush stdout) resultTokens
  putStrLn ""

  return (resultTokens, StepCount nSteps)

zip5 :: [a] -> [b] -> [c] -> [d] -> [e] -> [(a, b, c, d, e)]
zip5 (a:as) (b:bs) (c:cs) (d:ds) (e:es) = (a, b, c, d, e) : zip5 as bs cs ds es
zip5 _ _ _ _ _ = []

-- Helper function to create a bundled version of topEntity
topEntityBundled
  :: CS.HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> C.Signal dom (Int, Helpers.Token, Float, Int, C.Vec SeqLen Helpers.Token)
  -> C.Signal dom Helpers.Token
topEntityBundled decoder bundledInputs = 
  let (seqPos, inputToken, temp, rngSeed, promptVec) = C.unbundle bundledInputs
  in topEntity decoder seqPos inputToken temp rngSeed promptVec
