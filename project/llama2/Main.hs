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
      NumAttentionHeads,
      NumLayers,
      NumKeyValueHeads,
      HiddenDim,
      ModelDim,
      vocabSize,
      FreqDim )
import Model ( topEntity )
import qualified Clash.Prelude as C
import qualified Clash.Signal as CS
import GHC.IO (unsafePerformIO)
import GHC.Base (when)

type Vocabulary = [BSL.ByteString]
type VocabularyScores = [Float]
type PromptTokens = [Helpers.Token]


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
    initModel :: BSL.ByteString -> Helpers.TransformerDecoderComponent
    initModel = BG.runGet parseModelConfigFile
    config = initModel modelFileContent
    prompt' = fromMaybe "" prompt
    (promptTokens, vocab) = tokenizerInit tokenizerFileContent Helpers.vocabSize (BSC.pack prompt')

  putStrLn "<s>"
  startTime <- getPOSIXTime

  (_, Helpers.StepCount countTokens) <- generateTokensSimAutoregressive config vocab (fromIntegral steps) promptTokens temperature seedValue

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

readVec4D :: forall m n p q. (C.KnownNat m, C.KnownNat n, C.KnownNat p, C.KnownNat q) => BG.Get (C.Vec m (C.Vec n (C.Vec p (C.Vec q Float))))
readVec4D = do
    let m = C.snatToNum (C.SNat :: C.SNat m)
        n = C.snatToNum (C.SNat :: C.SNat n)
        p = C.snatToNum (C.SNat :: C.SNat p)
        q = C.snatToNum (C.SNat :: C.SNat q)
        total = m * n * p * q
    vec <- readVector total
    let floatList = V.toList vec
        innerChunks = chunksOf p floatList
        innerVecs = map CV.unsafeFromList innerChunks
        middleChunks = chunksOf n innerVecs
        middleVecs = map CV.unsafeFromList middleChunks
        externChunks = chunksOf m middleVecs
        externVecs = map CV.unsafeFromList externChunks
    return $ CV.unsafeFromList externVecs
  where
    chunksOf :: Int -> [a] -> [[a]]
    chunksOf _ [] = []
    chunksOf k xs = take k xs : chunksOf k (drop k xs)

parseModelConfigFile :: BG.Get Helpers.TransformerDecoderComponent
parseModelConfigFile = do
  replicateM_ 7 BG.getInt32le

  tokenEmbeddingTable' <- readVec2D @Helpers.VocabSize @Helpers.ModelDim
  rmsAttWeight'        <- readVec2D @Helpers.NumLayers @Helpers.ModelDim
  wq'                  <- readVec4D @Helpers.NumLayers @Helpers.NumAttentionHeads @Helpers.HeadDimension @Helpers.ModelDim
  wk'                  <- readVec4D @Helpers.NumLayers @Helpers.NumKeyValueHeads @Helpers.HeadDimension @Helpers.ModelDim
  wv'                  <- readVec4D @Helpers.NumLayers @Helpers.NumKeyValueHeads @Helpers.HeadDimension @Helpers.ModelDim
  wo'                  <- readVec4D @Helpers.NumLayers @Helpers.NumAttentionHeads @Helpers.ModelDim @Helpers.HeadDimension 
  rmsFfnWeight'        <- readVec2D @Helpers.NumLayers @Helpers.ModelDim
  w1'                  <- readVec3D @Helpers.NumLayers @Helpers.HiddenDim @Helpers.ModelDim
  w2'                  <- readVec3D @Helpers.NumLayers @Helpers.ModelDim @Helpers.HiddenDim
  w3'                  <- readVec3D @Helpers.NumLayers @Helpers.HiddenDim @Helpers.ModelDim
  rmsFinalWeight'      <- readVec1D @Helpers.ModelDim
  freqCisReal'         <- readVec2D @Helpers.SeqLen @Helpers.FreqDim
  freqCisImag'         <- readVec2D @Helpers.SeqLen @Helpers.FreqDim

  let
    embedding = Helpers.EmbeddingComponent
      { vocabulary     = Helpers.CArray2D tokenEmbeddingTable'
      , rmsFinalWeight = rmsFinalWeight'
      }

    layer :: C.Index Helpers.NumLayers -> Helpers.TransformerLayerComponent
    layer lIdx =
      let
        sha :: C.Index Helpers.NumAttentionHeads -> Helpers.SingleHeadComponent
        sha hIdx =
          let nQ  = C.snatToNum (C.SNat @Helpers.NumAttentionHeads) :: Int
              nKV = C.snatToNum (C.SNat @Helpers.NumKeyValueHeads)  :: Int
              kvMul = max 1 (nQ `div` nKV)
              kvIdxInt = fromIntegral hIdx `div` kvMul
              kvIdx :: C.Index Helpers.NumKeyValueHeads
              kvIdx = fromInteger (toInteger kvIdxInt)
          in Helpers.SingleHeadComponent
               { wqHead = Helpers.CArray2D $ (wq' C.!! lIdx) C.!! hIdx
               , wkHead = Helpers.CArray2D $ (wk' C.!! lIdx) C.!! kvIdx
               , wvHead = Helpers.CArray2D $ (wv' C.!! lIdx) C.!! kvIdx
               , rotary  = Helpers.RotaryEncodingComponent
                   { freqCos = Helpers.CArray2D freqCisReal'
                   , freqSin = Helpers.CArray2D freqCisImag'
                   }
               }

        mWoVec :: C.Vec NumAttentionHeads (CArray2D ModelDim HeadDimension)
        mWoVec = C.map
            (\hIdx -> Helpers.CArray2D $ (wo' C.!! lIdx) C.!! hIdx)
            (C.indicesI @Helpers.NumAttentionHeads)

      in Helpers.TransformerLayerComponent
           { multiHeadAttention = Helpers.MultiHeadAttentionComponent
               { heads  = C.map sha (C.indicesI :: C.Vec Helpers.NumAttentionHeads (C.Index Helpers.NumAttentionHeads))
               , mWo    = mWoVec
               , rmsAtt = rmsAttWeight' C.!! lIdx
               }
           , feedforwardNetwork = Helpers.FeedForwardNetworkComponent
               { fW1     = Helpers.CArray2D $ w1' C.!! toInteger lIdx
               , fW2     = Helpers.CArray2D $ w2' C.!! toInteger lIdx
               , fW3     = Helpers.CArray2D $ w3' C.!! toInteger lIdx
               , fRMSFfn = rmsFfnWeight' C.!! lIdx
               }
           }

    decoder = Helpers.TransformerDecoderComponent
      { modelEmbedding = embedding
      , modelLayers    = C.map layer (C.indicesI :: C.Vec Helpers.NumLayers (C.Index Helpers.NumLayers))
      }

  return decoder

--------------------------------------------------------------------------------
-- Token Generation with Clash Simulation
--------------------------------------------------------------------------------

-- | Autoregressive token generation, one token at a time.
generateTokensSimAutoregressive
  :: Helpers.TransformerDecoderComponent
  -> Vocabulary
  -> C.Unsigned 32                      -- ^ number of steps to generate
  -> [Helpers.Token]                    -- ^ prompt tokens
  -> Float                              -- ^ temperature
  -> Int                                -- ^ seed
  -> IO ([Helpers.Token], Helpers.StepCount)   -- ^ produced tokens and token count
generateTokensSimAutoregressive decoder vocab nSteps promptTokens temperature seed = do
  let promptLen  = length promptTokens
      totalSteps = promptLen + fromIntegral nSteps

  putStrLn $ "Prompt: " ++ show promptTokens
  putStr "Generated: "
  hFlush stdout

  -- We keep an accumulating list of all tokens seen so far
  let go :: Int -> [Helpers.Token] -> IO [Helpers.Token]
      go step acc
        | step >= totalSteps = return (drop promptLen acc)  -- return only generated part
        | otherwise = do
            let seqPos    = step
                -- last token produced or prompt token
                inputToken | step < promptLen = promptTokens !! step
                           | otherwise        = last acc
                rngSeed    = seed + step
                -- Build promptVec padded/truncated to seqLen
                bundledIn   = (seqPos, inputToken, temperature, rngSeed)

                -- run topEntity for exactly one clock tick
                outToken    = head $ CS.simulate (topEntityBundled @CS.System decoder) [bundledIn]

            -- print as we go
            when (step >= promptLen) $ do
              putStr (show outToken ++ " ")
              hFlush stdout

            go (step + 1) (acc ++ [outToken])

  generated <- go 0 promptTokens
  putStrLn ""
  return (take (fromIntegral nSteps) generated, Helpers.StepCount nSteps)

-- Helper function to create a bundled version of topEntity
topEntityBundled
  :: CS.HiddenClockResetEnable dom
  => Helpers.TransformerDecoderComponent
  -> C.Signal dom (Int, Helpers.Token, Float, Int)
  -> C.Signal dom Helpers.Token
topEntityBundled decoder bundledInputs = 
  let (seqPos, inputToken, temp, rngSeed) = C.unbundle bundledInputs
  in topEntity decoder seqPos inputToken temp rngSeed
