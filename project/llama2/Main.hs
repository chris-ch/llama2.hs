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
import Data.Maybe (fromMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)
import System.IO (hFlush, stdout)
import Control.Monad (replicateM_)
import Text.Printf (printf)
import Helpers
    (
      SingleHeadComponent(SingleHeadComponent, rotary, wqHead, wkHead,
                          wvHead),
      RotaryEncodingComponent(RotaryEncodingComponent, freqSin, freqCos),
      EmbeddingComponent(EmbeddingComponent, rmsFinalWeight, vocabulary),
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
      FreqDim, Temperature, Seed )
import Model ( topEntity )
import qualified Tokenizer as T (buildTokenizer, encodeTokens, Tokenizer, decodePiece)
import Model.Layers.TransformerLayer (TransformerDecoderComponent (..), TransformerLayerComponent (..))
import qualified Model.Layer as Layer
import qualified Model.Layers.FeedForward.FeedForwardNetwork as FeedForwardNetwork
import qualified Model.Layers.Attention.MultiHeadAttention as MultiHeadAttention

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

runModel :: BSL.ByteString -> BSL.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
runModel modelFileContent tokenizerFileContent temperature steps prompt seed = do
  currentTime <- getPOSIXTime
  let
    seedValue = fromIntegral $ fromMaybe (round currentTime) seed
    initModel = BG.runGet parseModelConfigFile
    config    = initModel modelFileContent
    tokenizer = T.buildTokenizer tokenizerFileContent (C.natToNum @VocabSize)
    prompt'   = fromMaybe "" prompt
  let
    promptTokensI = T.encodeTokens tokenizer (BSC.pack prompt') True False
    promptTokens  = map fromIntegral promptTokensI :: [Token]

  -- After promptTokens are built:
  case promptTokens of
    (t0:t1:_) -> do
      putStrLn $ "[TRACE] token0=" ++ show t0
      putStrLn $ "[TRACE] token1=" ++ show t1
      -- Run pure tracer: compares directly to your C prints for pos 0 and pos 1
    _ -> putStrLn "Need at least two tokens in the prompt to run pos0/pos1 tracer."

  putStrLn "✅ model loaded successfully"
  putStrLn "<s>"
  startTime <- getPOSIXTime

  (_, Layer.StepCount countTokens) <- generateTokensSimAutoregressive config tokenizer (fromIntegral steps) promptTokens temperature seedValue

  endTime <- getPOSIXTime
  let duration :: Integer
      duration = round (endTime - startTime)
      tokensPerSec :: Float
      tokensPerSec = fromIntegral countTokens / fromIntegral duration
  printf "\nduration: %ds - (%.02f tokens/s)\n" duration tokensPerSec
  return ()

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
  rmsAttWeight' <- readVec2D @NumLayers @ModelDim
  wq' <- readVec4D @NumLayers @NumQueryHeads @HeadDimension @ModelDim
  wk' <- readVec4D @NumLayers @NumKeyValueHeads @HeadDimension @ModelDim
  wv' <- readVec4D @NumLayers @NumKeyValueHeads @HeadDimension @ModelDim
  wo' <- readVec3D @NumLayers @ModelDim @ModelDim
  rmsFfnWeight' <- readVec2D @NumLayers @ModelDim
  w1' <- readVec3D @NumLayers @HiddenDim @ModelDim
  w2' <- readVec3D @NumLayers @ModelDim @HiddenDim
  w3' <- readVec3D @NumLayers @HiddenDim @ModelDim
  rmsFinalWeight' <- readVec1D @ModelDim
  freqCisReal' <- readVec2D @SeqLen @FreqDim
  freqCisImag' <- readVec2D @SeqLen @FreqDim
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
        woLayer :: C.Vec ModelDim (C.Vec ModelDim Float)
        woLayer = wo' C.!! lIdx
        headBlock :: C.Index NumQueryHeads -> CArray2D ModelDim HeadDimension
        headBlock hIdx =
          let base :: Int
              base = fromIntegral hIdx * C.snatToNum (C.SNat @HeadDimension)
              -- for each output row (0..modelDim-1), pick the headDim columns:
              rowSlice :: C.Vec ModelDim Float -> C.Vec HeadDimension Float
              rowSlice row =
                C.map
                  (\off -> row C.!! (toEnum (base + fromIntegral off) :: C.Index ModelDim))
                  (C.indicesI @HeadDimension)
          in CArray2D (C.map rowSlice woLayer)

        mWoVec :: C.Vec NumQueryHeads (CArray2D ModelDim HeadDimension)
        mWoVec = C.map headBlock (C.indicesI @NumQueryHeads)

      in TransformerLayerComponent
           { multiHeadAttention = MultiHeadAttention.MultiHeadAttentionComponent
               { heads  = C.map sha (C.indicesI :: C.Vec NumQueryHeads (C.Index NumQueryHeads))
               , mWo    = mWoVec
               , rmsAtt = rmsAttWeight' C.!! lIdx
               }
           , feedforwardNetwork = FeedForwardNetwork.FeedForwardNetworkComponent
               { fW1     = CArray2D $ w1' C.!! lIdx
               , fW2     = CArray2D $ w2' C.!! lIdx
               , fW3     = CArray2D $ w3' C.!! lIdx
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
  -> T.Tokenizer
  -> C.Unsigned 32
  -> [Token]
  -> Temperature
  -> Seed
  -> IO ([Token], Layer.StepCount)
generateTokensSimAutoregressive decoder tokenizer nSteps promptTokens temperature seed = do

  putStrLn $ "✅ Prompt: " ++ show promptTokens
  putStr "<s>\n"
  putStr "Generated: "
  hFlush stdout

  -- Constant input streams
  let temps   = repeat temperature
      seeds   = repeat seed

  -- One simulation run; inputs depend on the token stream,
  -- which depends on outputs (knot-tying).
  let outputs :: [(Token, Bool)]
      outputs = CS.simulate (bundledOutputs decoder) (DL.zip3 tokenStream temps seeds)

      (outTokens, readyFlags) = unzip outputs

      -- Drive the input token:
      -- Hold current token until a ready pulse.
      -- On ready: consume next prompt token if any, else feed back the last sampled token.
      tokenStream :: [Token]
      tokenStream =
        let (cur0, restPrompt) =
              case promptTokens of
                (t0:ts) -> (t0, ts)
                []      -> (1, []) -- start from BOS if no prompt
        in drive cur0 restPrompt readyFlags outTokens
        where
          drive :: Token -> [Token] -> [Bool] -> [Token] -> [Token]
          drive cur ps (r:rs) (o:os) =
            cur : if not r
                    then drive cur ps rs os
                    else case ps of
                           (p:ps') -> drive p  ps'  rs os
                           []      -> drive o  []    rs os
          drive cur _ _ _ = repeat cur  -- totality guard

      -- Tokens produced by the DUT at ready pulses
      sampledAll :: [Token]
      sampledAll = [ t | (t,r) <- outputs, r ]

  -- We want to EMIT: first the entire prompt (forced), then the samples.
  let promptLen = length promptTokens
      forcedEmitted = promptTokens                         -- emit full prompt
      sampledAfterPrompt = drop promptLen sampledAll       -- then model outputs
      emittedAll = forcedEmitted ++ sampledAfterPrompt

      totalWanted = promptLen + fromIntegral nSteps
      emittedLimited = take totalWanted emittedAll

  -- Print
  -- Build (prev,next) transitions like llama2.c
  let emitted = emittedLimited
      trans   = zip emitted (drop 1 emitted)

  mapM_
    (\(prev, nxt) ->
        BSC.putStr (T.decodePiece tokenizer (fromIntegral prev) (fromIntegral nxt))
        >> hFlush stdout)
    trans
  putStrLn ""
  -- Only the generated tokens (exclude the prompt portion)
  let generated = take (fromIntegral nSteps) (drop promptLen emittedLimited)
  pure (generated, Layer.StepCount nSteps)

bundledOutputs :: TransformerDecoderComponent -> CS.Signal CS.System (Token, Temperature, Seed) -> CS.Signal C.System (Token, Bool)
bundledOutputs decoder = CS.bundle . CS.exposeClockResetEnable
    (topEntityBundled @CS.System decoder)
    CS.systemClockGen
    CS.resetGen
    CS.enableGen

-- Helper function to create a bundled version of topEntity
topEntityBundled
  :: CS.HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> C.Signal dom (Token, Temperature, Seed)
  -> (C.Signal dom Token, C.Signal dom Bool)
topEntityBundled decoder bundledInputs = topEntity decoder inputToken temp rngSeed
  where
    (inputToken, temp, rngSeed) = C.unbundle bundledInputs
