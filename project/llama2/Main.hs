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
import qualified Clash.Prelude as C
import qualified Clash.Signal as CS

import GHC.IO (unsafePerformIO)
import GHC.Base (when)
import Data.Maybe (fromMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)
import GHC.Unicode (isSpace)
import System.IO (hFlush, stdout)
import Control.Monad (replicateM, replicateM_)
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
      Token,
      CArray2D(CArray2D),
      HeadDimension,
      SeqLen,
      VocabSize,
      NumQueryHeads,
      NumLayers,
      NumKeyValueHeads,
      HiddenDim,
      ModelDim,
      vocabSize,
      FreqDim )
import Model ( multiCycleTransformer, initAttentionCache, ProcessingState (..), topEntity )
import Data.List (findIndex)
import Debug.Trace (trace)

type Vocabulary = [BSL.ByteString]
type VocabularyScores = [Float]
type PromptTokens = [Token]

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
        checkPair (count, (tokenPrev, tokenNext)) acc =
          case strLookup ((vocab !! fromIntegral tokenPrev) `BSL.append` (vocab !! fromIntegral tokenNext)) vocab of
            pos | pos /= -1 && vocabScores !! pos > bestScore -> Just (count, fromIntegral pos)
            _ -> acc

        bestScore :: Float
        bestScore = -1e10

    mergePair :: Int -> Token -> [Token] -> [Token]
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

  putStrLn "✅ model loaded successfully"
  putStrLn "<s>"
  startTime <- getPOSIXTime

  (_, StepCount countTokens) <- generateTokensSimAutoregressive config vocab (fromIntegral steps) promptTokens temperature seedValue

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
    chunksOf _ [] = []
    chunksOf k xs = take k xs : chunksOf k (drop k xs)

readVec4D :: forall m n p q. (C.KnownNat m, C.KnownNat n, C.KnownNat p, C.KnownNat q) => BG.Get (C.Vec m (C.Vec n (C.Vec p (C.Vec q Float))))
readVec4D = do
    let m = C.snatToNum (C.SNat :: C.SNat m)
        n = C.snatToNum (C.SNat :: C.SNat n)
        p = C.snatToNum (C.SNat :: C.SNat p)
        q = C.snatToNum (C.SNat :: C.SNat q)
        total = m * n * p * q
    vec <- readVector total
    let floatList = V.toList vec
        -- First chunk into q-sized vectors
        innerChunks = chunksOf q floatList
        innerVecs = map CV.unsafeFromList innerChunks
        -- Then chunk into p groups of q-sized vectors
        middleChunks = chunksOf p innerVecs
        middleVecs = map CV.unsafeFromList middleChunks
        -- Then chunk into n groups of p×q-sized tensors
        outerChunks = chunksOf n middleVecs
        outerVecs = map CV.unsafeFromList outerChunks
        -- Finally chunk into m groups
    return $ CV.unsafeFromList outerVecs
  where
    chunksOf _ [] = []
    chunksOf k xs = take k xs : chunksOf k (drop k xs)

parseModelConfigFile :: BG.Get TransformerDecoderComponent
parseModelConfigFile = do
  replicateM_ 7 BG.getInt32le
  tokenEmbeddingTable' <- readVec2D @VocabSize @ModelDim
  let tokenShape = (length (C.toList tokenEmbeddingTable'), length (C.toList (C.head tokenEmbeddingTable')))
      expectedTokenShape = (C.natToNum @VocabSize, C.natToNum @ModelDim)
  trace ("parseModelConfigFile: tokenEmbeddingTable shape=" ++ show tokenShape ++ ", expected=" ++ show expectedTokenShape) return ()
  rmsAttWeight' <- readVec2D @NumLayers @ModelDim
  let rmsAttShape = (length (C.toList rmsAttWeight'), length (C.toList (C.head rmsAttWeight')))
      expectedRmsAttShape = (C.natToNum @NumLayers, C.natToNum @ModelDim)
  trace ("parseModelConfigFile: rmsAttWeight shape=" ++ show rmsAttShape ++ ", expected=" ++ show expectedRmsAttShape) return ()
  wq' <- readVec4D @NumLayers @NumQueryHeads @HeadDimension @ModelDim
  let wqShape = (length (C.toList wq'), length (C.toList (C.head wq')), length (C.toList (C.head (C.head wq'))), length (C.toList (C.head (C.head (C.head wq')))))
      expectedWqShape = (C.natToNum @NumLayers, C.natToNum @NumQueryHeads, C.natToNum @HeadDimension, C.natToNum @ModelDim)
  trace ("parseModelConfigFile: wq shape=" ++ show wqShape ++ ", expected=" ++ show expectedWqShape) return ()
  wk' <- readVec4D @NumLayers @NumKeyValueHeads @HeadDimension @ModelDim
  let wkShape = (length (C.toList wk'), length (C.toList (C.head wk')), length (C.toList (C.head (C.head wk'))), length (C.toList (C.head (C.head (C.head wk')))))
      expectedWkShape = (C.natToNum @NumLayers, C.natToNum @NumKeyValueHeads, C.natToNum @HeadDimension, C.natToNum @ModelDim)
  trace ("parseModelConfigFile: wk shape=" ++ show wkShape ++ ", expected=" ++ show expectedWkShape) return ()
  wv' <- readVec4D @NumLayers @NumKeyValueHeads @HeadDimension @ModelDim
  let wvShape = (length (C.toList wv'), length (C.toList (C.head wv')), length (C.toList (C.head (C.head wv'))), length (C.toList (C.head (C.head (C.head wv')))))
      expectedWvShape = (C.natToNum @NumLayers, C.natToNum @NumKeyValueHeads, C.natToNum @HeadDimension, C.natToNum @ModelDim)
  trace ("parseModelConfigFile: wv shape=" ++ show wvShape ++ ", expected=" ++ show expectedWvShape) return ()
  wo' <- readVec4D @NumLayers @NumQueryHeads @ModelDim @HeadDimension
  let woShape = (length (C.toList wo'), length (C.toList (C.head wo')), length (C.toList (C.head (C.head wo'))), length (C.toList (C.head (C.head (C.head wo')))))
      expectedWoShape = (C.natToNum @NumLayers, C.natToNum @NumQueryHeads, C.natToNum @ModelDim, C.natToNum @HeadDimension)
  trace ("parseModelConfigFile: wo shape=" ++ show woShape ++ ", expected=" ++ show expectedWoShape) return ()
  rmsFfnWeight' <- readVec2D @NumLayers @ModelDim
  let rmsFfnShape = (length (C.toList rmsFfnWeight'), length (C.toList (C.head rmsFfnWeight')))
      expectedRmsFfnShape = (C.natToNum @NumLayers, C.natToNum @ModelDim)
  trace ("parseModelConfigFile: rmsFfnWeight shape=" ++ show rmsFfnShape ++ ", expected=" ++ show expectedRmsFfnShape) return ()
  w1' <- readVec3D @NumLayers @HiddenDim @ModelDim
  let w1Shape = (length (C.toList w1'), length (C.toList (C.head w1')), length (C.toList (C.head (C.head w1'))))
      expectedW1Shape = (C.natToNum @NumLayers, C.natToNum @HiddenDim, C.natToNum @ModelDim)
  trace ("parseModelConfigFile: w1 shape=" ++ show w1Shape ++ ", expected=" ++ show expectedW1Shape) return ()
  w2' <- readVec3D @NumLayers @ModelDim @HiddenDim
  let w2Shape = (length (C.toList w2'), length (C.toList (C.head w2')), length (C.toList (C.head (C.head w2'))))
      expectedW2Shape = (C.natToNum @NumLayers, C.natToNum @ModelDim, C.natToNum @HiddenDim)
  trace ("parseModelConfigFile: w2 shape=" ++ show w2Shape ++ ", expected=" ++ show expectedW2Shape) return ()
  w3' <- readVec3D @NumLayers @HiddenDim @ModelDim
  let w3Shape = (length (C.toList w3'), length (C.toList (C.head w3')), length (C.toList (C.head (C.head w3'))))
      expectedW3Shape = (C.natToNum @NumLayers, C.natToNum @HiddenDim, C.natToNum @ModelDim)
  trace ("parseModelConfigFile: w3 shape=" ++ show w3Shape ++ ", expected=" ++ show expectedW3Shape) return ()
  rmsFinalWeight' <- readVec1D @ModelDim
  let rmsFinalShape = length (C.toList rmsFinalWeight')
      expectedRmsFinalShape = C.natToNum @ModelDim
  trace ("parseModelConfigFile: rmsFinalWeight shape=" ++ show rmsFinalShape ++ ", expected=" ++ show expectedRmsFinalShape) return ()
  freqCisReal' <- readVec2D @SeqLen @FreqDim
  let freqCisRealShape = (length (C.toList freqCisReal'), length (C.toList (C.head freqCisReal')))
      expectedFreqCisRealShape = (C.natToNum @SeqLen, C.natToNum @FreqDim)
  trace ("parseModelConfigFile: freqCisReal shape=" ++ show freqCisRealShape ++ ", expected=" ++ show expectedFreqCisRealShape) return ()
  freqCisImag' <- readVec2D @SeqLen @FreqDim
  let freqCisImagShape = (length (C.toList freqCisImag'), length (C.toList (C.head freqCisImag')))
      expectedFreqCisImagShape = (C.natToNum @SeqLen, C.natToNum @FreqDim)
  trace ("parseModelConfigFile: freqCisImag shape=" ++ show freqCisImagShape ++ ", expected=" ++ show expectedFreqCisImagShape) return ()
  let
    embedding = EmbeddingComponent
      { vocabulary     = CArray2D tokenEmbeddingTable'
      , rmsFinalWeight = rmsFinalWeight'
      }

    layer :: C.Index NumLayers -> TransformerLayerComponent
    layer lIdx =
      let
        sha :: C.Index NumQueryHeads -> SingleHeadComponent
        sha hIdx =
          let nQ  = C.snatToNum (C.SNat @NumQueryHeads) :: Int
              nKV = C.snatToNum (C.SNat @NumKeyValueHeads)  :: Int
              kvMul = max 1 (nQ `div` nKV)
              kvIdxInt = fromIntegral hIdx `div` kvMul
              kvIdx :: C.Index NumKeyValueHeads
              kvIdx = fromInteger (toInteger kvIdxInt)
          in SingleHeadComponent
               { wqHead = CArray2D $ (wq' C.!! lIdx) C.!! hIdx
               , wkHead = CArray2D $ (wk' C.!! lIdx) C.!! kvIdx
               , wvHead = CArray2D $ (wv' C.!! lIdx) C.!! kvIdx
               , rotary  = RotaryEncodingComponent
                   { freqCos = CArray2D freqCisReal'
                   , freqSin = CArray2D freqCisImag'
                   }
               }

        mWoVec :: C.Vec NumQueryHeads (CArray2D ModelDim HeadDimension)
        mWoVec = C.map
            (\hIdx -> CArray2D $ (wo' C.!! lIdx) C.!! hIdx)
            (C.indicesI @NumQueryHeads)

      in TransformerLayerComponent
           { multiHeadAttention = MultiHeadAttentionComponent
               { heads  = C.map sha (C.indicesI :: C.Vec NumQueryHeads (C.Index NumQueryHeads))
               , mWo    = mWoVec
               , rmsAtt = rmsAttWeight' C.!! lIdx
               }
           , feedforwardNetwork = FeedForwardNetworkComponent
               { fW1     = CArray2D $ w1' C.!! toInteger lIdx
               , fW2     = CArray2D $ w2' C.!! toInteger lIdx
               , fW3     = CArray2D $ w3' C.!! toInteger lIdx
               , fRMSFfn = rmsFfnWeight' C.!! lIdx
               }
           }

    decoder = TransformerDecoderComponent
      { modelEmbedding = embedding
      , modelLayers    = C.map layer (C.indicesI :: C.Vec NumLayers (C.Index NumLayers))
      }

  return decoder

--------------------------------------------------------------------------------
-- Token Generation with Clash Simulation
--------------------------------------------------------------------------------

-- | Autoregressive token generation, one token at a time.
generateTokensSimAutoregressive
  :: TransformerDecoderComponent
  -> Vocabulary
  -> C.Unsigned 32                      -- ^ number of steps to generate
  -> [Token]                    -- ^ prompt tokens
  -> Float                              -- ^ temperature
  -> Int                                -- ^ seed
  -> IO ([Token], StepCount)   -- ^ produced tokens and token count
generateTokensSimAutoregressive decoder vocab nSteps promptTokens temperature seed = do
  let promptLen = length promptTokens
  let totalSteps = promptLen + fromIntegral nSteps

  putStrLn $ "✅ Prompt: " ++ show promptTokens
  putStr "Generated: "
  hFlush stdout

  let
    go step acc
      | step >= totalSteps = return (drop promptLen acc)
      | otherwise = do
          let seqPos = step
              inputToken | step < promptLen = promptTokens !! step
                        | otherwise = last acc
              bundledIn = (seqPos, inputToken, temperature, seed)
          putStrLn $ "Step " ++ show step ++ ": seqPos=" ++ show seqPos ++ ", inputToken=" ++ show inputToken ++ ", rngSeed=" ++ show seed
          let outputs = CS.simulate (bundledOutputs decoder) (repeat bundledIn)
              (outToken, readyFlags) = unzip outputs
              readyIdx = fromMaybe 0 $ findIndex id readyFlags
              tokenProduced = outToken !! readyIdx
          putStrLn $ "Output token: " ++ show tokenProduced ++ ", readyIdx: " ++ show readyIdx
          when (step >= promptLen) $ do
            putStr (show tokenProduced ++ " ")
            hFlush stdout
          go (step+1) (acc ++ [tokenProduced])

  generated <- go 0 promptTokens
  putStrLn ""
  return (take (fromIntegral nSteps) generated, StepCount nSteps)

bundledOutputs :: TransformerDecoderComponent -> CS.Signal CS.System (Int, Token, Float, Int) -> CS.Signal C.System (C.Unsigned 32, Bool)
bundledOutputs decoder = CS.bundle . CS.exposeClockResetEnable
    (topEntityBundled @CS.System decoder)
    CS.systemClockGen
    CS.resetGen
    CS.enableGen

-- Helper function to create a bundled version of topEntity
topEntityBundled
  :: CS.HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> C.Signal dom (Int, Token, Float, Int)
  -> (C.Signal dom Token, C.Signal dom Bool)
topEntityBundled decoder bundledInputs = topEntity decoder seqPos inputToken temp rngSeed
  where
    (seqPos, inputToken, temp, rngSeed) = C.unbundle bundledInputs
