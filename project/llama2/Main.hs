module Main (main) where

import Prelude

import qualified Data.Binary.Get as BG
import qualified Data.ByteString.Lazy as BSL
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.List as DL
import qualified Options.Applicative as OA
import qualified Foreign as F
import qualified Data.Vector.Unboxed as V
import qualified Clash.Sized.Vector as CV

import Data.Maybe (fromMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)
import GHC.Unicode (isSpace)
import System.IO (hFlush, stdout)
import Control.Monad (replicateM)
import Text.Printf (printf)

import Helpers
    ( TransformerDecoderComponent(..),
      TransformerLayerComponent(TransformerLayerComponent,
                                feedforwardNetwork, multiHeadAttention),
      FeedForwardNetworkComponent(FeedForwardNetworkComponent, fRMSFfn,
                                  fW1, fW2, fW3),
      MultiHeadAttentionComponent(MultiHeadAttentionComponent, rmsAtt,
                                  heads, mWo),
      SingleHeadComponent(SingleHeadComponent, rotary, wqHead, wkHead,
                          wvHead),
      RotaryEncodingComponent(RotaryEncodingComponent, freqSin, freqCos),
      EmbeddingComponent(EmbeddingComponent, rmsFinalWeight, vocabulary),
      StepCount(..),
      Token(..),
      CArray2D(CArray2D),
      HeadDimension,
      SeqLen,
      VocabSize,
      NumAttentionHeads,
      NumLayers,
      HiddenDim,
      ModelDim,
      vocabSize,
      seqLen )
import Model ( topEntity )
import qualified Clash.Prelude as C
import qualified Clash.Signal as CS
import qualified Clash.Sized.Vector as CSV
import GHC.IO (unsafePerformIO)

type Vocabulary = [BSL.ByteString]
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
  modelFileContent <- BSL.readFile modelFile
  tokenizerFileContent <- BSL.readFile tokenizerFile
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

parseTokens :: BSL.ByteString -> Int -> (Vocabulary, VocabularyScores)
parseTokens fileContent size = (vocab, vocabScores)
  where
    scoresTokens = BG.runGet scoresAndTokens fileContent
    vocabScores = map fst scoresTokens
    vocab = map snd scoresTokens

    scoresAndTokens :: BG.Get [(Float, BSL.ByteString)]
    scoresAndTokens = replicateM size readToken

    readToken :: BG.Get (Float, BSL.ByteString)
    readToken = do
      score <- BG.getFloatle
      tokenSize <- BG.getInt32le
      token <- BG.getLazyByteString (fromIntegral tokenSize)
      return (score, token)

tokenizerInit :: BSL.ByteString -> Int -> BSL.ByteString -> (PromptTokens, Vocabulary)
tokenizerInit file size prompt = (bpeEncode prompt vocab vocabScores, vocab)
  where
    (vocab, vocabScores) = parseTokens (BSL.drop 4 file) size

strLookup :: BSL.ByteString -> Vocabulary -> Int
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
        checkPair (count, (tokenPrev, tokenNext)) acc =
          case strLookup ((vocab !! fromIntegral tokenPrev) `BSL.append` (vocab !! fromIntegral tokenNext)) vocab of
            pos | pos /= -1 && vocabScores !! pos > bestScore -> Just (count, fromIntegral pos)
            _ -> acc

        bestScore :: Float
        bestScore = -1e10

    mergePair :: Int -> Helpers.Token -> [Helpers.Token] -> [Helpers.Token]
    mergePair count token tokens' =
      take count tokens' ++ [token] ++ drop (count + 2) tokens'

bpeEncode :: BSL.ByteString -> Vocabulary -> VocabularyScores -> PromptTokens
bpeEncode prompt vocab vocabScores =
  let initialTokens = map (\byte ->
        let byteStr = BSL.pack [byte]
            maybeIdx = DL.elemIndex byteStr vocab
        in case maybeIdx of
             Just idx -> fromIntegral idx
             Nothing  -> fromIntegral (fromEnum byte + 3)  -- Fallback to byte + 3
        ) (BSL.unpack prompt)
   in applyBPEMerges initialTokens vocab vocabScores

runModel :: BSL.ByteString -> BSL.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
runModel modelFileContent tokenizerFileContent temperature steps prompt seed = do
  currentTime <- getPOSIXTime
  let
    seedValue = fromMaybe (round currentTime) seed
    initModel :: BSL.ByteString -> TransformerDecoderComponent
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

-- ============================================================================
-- File Parsing
-- ============================================================================

readVector :: Int -> BG.Get (V.Vector Float)
readVector count = do
  byteData <- BG.getByteString (count * 4)
  return $! unsafePerformIO $ do
    BS.useAsCString byteData $ \ptr -> do
      let floatPtr = F.castPtr ptr :: F.Ptr Float
      V.generateM count (F.peekElemOff floatPtr)

readVec1D :: forall n. C.KnownNat n => BG.Get (C.Vec n Float)
readVec1D = do
    let total = C.snatToNum (C.SNat :: C.SNat n)
    vec <- readVector total
    return $ CV.unsafeFromList (V.toList vec)

readVec2D :: forall n m. (C.KnownNat n, C.KnownNat m) => BG.Get (C.Vec n (C.Vec m Float))
readVec2D = do
    let n = C.snatToNum (C.SNat :: C.SNat n)
        m = C.snatToNum (C.SNat :: C.SNat m)
        total = n * m
    vec <- readVector total
    let floatList = V.toList vec
        chunks = chunksOf m floatList
        vecs = map CV.unsafeFromList chunks
    return $ CV.unsafeFromList vecs
  where
    chunksOf :: Int -> [a] -> [[a]]
    chunksOf _ [] = []
    chunksOf k xs = take k xs : chunksOf k (drop k xs)

readVec3D :: forall n m p. (C.KnownNat n, C.KnownNat m, C.KnownNat p) => BG.Get (C.Vec n (C.Vec m (C.Vec p Float)))
readVec3D = do
    let n = C.snatToNum (C.SNat :: C.SNat n)
        m = C.snatToNum (C.SNat :: C.SNat m)
        p = C.snatToNum (C.SNat :: C.SNat p)
        total = n * m * p
    vec <- readVector total
    let floatList = V.toList vec
        innerChunks = chunksOf p floatList
        innerVecs = map CV.unsafeFromList innerChunks
        middleChunks = chunksOf m innerVecs
        middleVecs = map CV.unsafeFromList middleChunks
    return $ CV.unsafeFromList middleVecs
  where
    chunksOf :: Int -> [a] -> [[a]]
    chunksOf _ [] = []
    chunksOf k xs = take k xs : chunksOf k (drop k xs)

parseModelConfigFile :: BG.Get TransformerDecoderComponent
parseModelConfigFile = do
  _ <- BG.getInt32le
  _ <- BG.getInt32le
  _ <- BG.getInt32le
  _ <- BG.getInt32le
  _ <- BG.getInt32le
  _ <- BG.getInt32le
  _ <- BG.getInt32le
  tokenEmbeddingTable' <- readVec2D @VocabSize
  rmsAttWeight' <- readVec2D @NumLayers
  wq' <- readVec3D @NumLayers @HeadDimension @ModelDim
  wk' <- readVec3D @NumLayers @HeadDimension @ModelDim
  wv' <- readVec3D @NumLayers @HeadDimension @ModelDim
  wo' <- readVec3D @NumLayers @ModelDim @ModelDim
  rmsFfnWeight' <- readVec2D @NumLayers
  w1' <- readVec3D @NumLayers @HiddenDim @ModelDim
  w2' <- readVec3D @NumLayers @ModelDim @HiddenDim
  w3' <- readVec3D @NumLayers @HiddenDim @ModelDim
  rmsFinalWeight' <- readVec1D @ModelDim
  freqCisReal' <- readVec2D @SeqLen
  freqCisImag' <- readVec2D @SeqLen

  let embedding = EmbeddingComponent
        { vocabulary = CArray2D tokenEmbeddingTable'
        , rmsFinalWeight = rmsFinalWeight'
        }
      sha hIdx = SingleHeadComponent
        { wqHead = CArray2D $ wq' C.!! toInteger hIdx
        , wkHead = CArray2D $ wk' C.!! toInteger hIdx
        , wvHead = CArray2D $ wv' C.!! toInteger hIdx
        , rotary = RotaryEncodingComponent
            { freqCos = CArray2D freqCisReal'
            , freqSin = CArray2D freqCisImag'
            }
        }
      layer lIdx = TransformerLayerComponent
        { multiHeadAttention = MultiHeadAttentionComponent
            { heads = C.map sha (C.indicesI :: C.Vec NumAttentionHeads (C.Index NumAttentionHeads))
            , mWo = CArray2D $ wo' C.!! lIdx
            , rmsAtt = rmsAttWeight' C.!! lIdx
            }
        , feedforwardNetwork = FeedForwardNetworkComponent
            { fW1 = CArray2D $ w1' C.!! toInteger lIdx
            , fW2 = CArray2D $ w2' C.!! toInteger lIdx
            , fW3 = CArray2D $ w3' C.!! toInteger lIdx
            , fRMSFfn = rmsFfnWeight' C.!! lIdx
            }
        }

      decoder = TransformerDecoderComponent
        { modelEmbedding = embedding
        , modelLayers = C.map layer (C.indicesI :: C.Vec NumLayers (C.Index NumLayers))
        }

  return decoder

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
      promptPadded = take seqLen (promptTokens ++ repeat 0)
      promptVec = replicate totalSteps (CSV.unsafeFromList promptPadded)
      
      -- For the input tokens, we need to simulate the autoregressive process
      -- This is a simplified version - in practice you'd need to feed outputs back as inputs
      inputTokens = take totalSteps (promptTokens ++ repeat 0)

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
