module Main (main) where

import qualified Data.Binary.Get as BG
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.List as DL
import qualified Options.Applicative as OA
import qualified Foreign as F
import Control.Monad.Reader (ReaderT(runReaderT), MonadIO(liftIO), MonadReader(ask))
import Control.Monad.State (evalStateT, MonadState (..), runStateT )
import Data.Maybe (fromMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)
import GHC.Unicode (isSpace)
import System.IO (hFlush, stdout)
import Control.Monad (replicateM)
import Text.Printf (printf)
import qualified Control.Monad.Reader as MR

import Helpers
import Model
import qualified Clash.Prelude as C
import qualified Clash.Explicit.Prelude as CEP
import qualified Clash.Explicit.SimIO as CES
import qualified Clash.Clocks as CC
import qualified Clash.Explicit.SimIO as SimIO
import Control.Monad.State (StateT, get, MonadState (put))

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

-- | Generate tokens step by step using Clash's SimIO API
generateTokens
    :: StepCount
    -> PromptTokens
    -> Float
    -> Vocabulary
    -> Int
    -> TransformerResult C.System ([BS.ByteString], StepCount)
generateTokens maxSteps promptTokens temperature vocab seedValue = do
    sigTokens :: C.Signal C.System (C.Unsigned 32) <- tokenSignal maxSteps promptTokens temperature vocab seedValue

    liftIO $ do
        simSig <- SimIO.runSim sigTokens   -- turn the pure signal into a runnable simulation

        let go :: Int -> [BS.ByteString] -> IO ([BS.ByteString], StepCount)
            go step acc
              | step >= fromIntegral (C.unNat maxSteps) = return (reverse acc, maxSteps)
              | otherwise = do
                  [tokId] <- SimIO.sampleN simSig 1   -- pull exactly one token ID
                  let word = vocab !! fromIntegral tokId
                      tokenStr =
                        if tokId == 1 && not (BS.null word) && isSpace (BSC.head word)
                           then BSC.tail word
                           else word
                  BSC.putStr tokenStr
                  hFlush stdout
                  go (step + 1) (tokenStr : acc)

        go 0 []

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

  -- Create actual clock/reset/enable values for implicit parameters
  let clk :: C.Clock C.System
      clk = C.clockGen
      rst :: C.Reset C.System
      rst = C.resetGen
      en  :: C.Enable C.System
      en  = C.enableGen

  -- Bind implicit parameters for initDecoderCaches
  let ?clock = clk
      ?reset = rst
      ?enable = en

  attentionKV :: DecoderCache C.System <- initDecoderCaches

  (_, StepCount countTokens) <- evalStateT
    (runReaderT (generateTokens (StepCount steps) promptTokens temperature vocab seedValue) config)
    attentionKV

  endTime <- getPOSIXTime
  let duration :: Integer
      duration = round (endTime - startTime)
      tokensPerSec :: Float
      tokensPerSec = fromIntegral countTokens / fromIntegral duration
  printf "\nduration: %ds - (%.02f tokens/s)\n" duration tokensPerSec
  return ()

-- | Pure signal that produces one token ID per cycle.
tokenSignal
    :: StepCount                 -- ^ maximum number of steps (maxSteps)
    -> PromptTokens              -- ^ pre‑tokenised prompt
    -> Float                     -- ^ temperature (passed to transformer)
    -> Vocabulary                -- ^ vocab for lookup (only needed later)
    -> Int                       -- ^ seed value
    -> MR.ReaderT TransformerDecoderComponent (StateT (DecoderCache C.System) SimIO.SimIO)
         (C.Signal C.System (C.Unsigned 32))   -- ^ Signal of token IDs
tokenSignal (StepCount maxSteps) promptTokens temperature vocab seedValue = do
    cfg   <- ask               -- the model configuration (Reader env)
    cache <- get               -- current decoder cache (State)

    -- Build the *pure* signal that runs the transformer step‑by‑step.
    let stepFn :: C.Signal C.System (C.Unsigned 32)
        stepFn = C.mealy (transformerStep cfg temperature promptTokens seedValue) cache
                 (C.fromList [0..maxSteps-1])   -- dummy clock ticks, one per step
    return stepFn
  where
    -- | One‑step transition function for the transformer.
    --   It receives the previous cache and the current tick (ignored),
    --   and returns the next token ID together with the updated cache.
    transformerStep
        :: TransformerDecoderComponent
        -> Float
        -> PromptTokens
        -> Int
        -> DecoderCache C.System
        -> C.Unsigned 32                -- ^ dummy input (tick index)
        -> (DecoderCache C.System, C.Unsigned 32)   -- ^ (new cache, token ID)
    transformerStep cfg temp pTokens seedVal oldCache _tick =
        let (nextTok, newCache) = MR.runReader (runStateT (transformer (promptTokenAt stepIdx) (StepCount stepIdx) temp pTokens seedVal) oldCache) cfg
            stepIdx = fromIntegral (C.unNat (C.length (C.unbundle oldCache)))   -- just a placeholder; adjust if you keep an explicit step counter
        in (newCache, nextTok)

    -- Helper to fetch the appropriate prompt token (same as before)
    promptTokenAt :: Int -> Helpers.Token
    promptTokenAt s
      | s < length promptTokens = promptTokens !! s
      | otherwise               = 1   -- EOS / dummy token after prompt
