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
      FreqDim, Temperature, Seed, runSingleHeadQKV, applyRotaryToHead, matrixVectorMult, rmsNorm, dotProduct, computeAttentionScores, computeAttentionWeights, computeAttentionOutput )
import Model ( topEntity )
import qualified Tokenizer as T (buildTokenizer, encodeTokens, Tokenizer, decodePiece)

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
    tokenizer = T.buildTokenizer tokenizerFileContent vocabSize
    prompt'   = fromMaybe "" prompt
    promptTokensI = T.encodeTokens tokenizer (BSC.pack prompt') True False
    promptTokens  = map fromIntegral promptTokensI :: [Token]

  case promptTokens of
    (t0:_) -> traceL0P0 config t0
    []     -> pure ()

  putStrLn "✅ model loaded successfully"
  putStrLn "<s>"
  startTime <- getPOSIXTime

  (_, StepCount countTokens) <- generateTokensSimAutoregressive config tokenizer (fromIntegral steps) promptTokens temperature seedValue

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
           { multiHeadAttention = MultiHeadAttentionComponent
               { heads  = C.map sha (C.indicesI :: C.Vec NumQueryHeads (C.Index NumQueryHeads))
               , mWo    = mWoVec
               , rmsAtt = rmsAttWeight' C.!! lIdx
               }
           , feedforwardNetwork = FeedForwardNetworkComponent
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
  -> IO ([Token], StepCount)
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
  pure (generated, StepCount nSteps)

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

dump8 :: String -> C.Vec n Float -> IO ()
dump8 tag v = putStrLn $ tag ++ " " ++ show (take 8 (C.toList v))

dump8L :: String -> [Float] -> IO ()
dump8L tag xs = putStrLn $ tag ++ " " ++ show (take 8 xs)

flattenHeads
  :: C.Vec NumQueryHeads (C.Vec HeadDimension Float)
  -> C.Vec ModelDim Float
flattenHeads = C.concat

traceL0P0 :: TransformerDecoderComponent -> Token -> IO ()
traceL0P0 dec tok = do

  let EmbeddingComponent{vocabulary = CArray2D vocabRows, rmsFinalWeight = rmsFin} = modelEmbedding dec
      l0 :: TransformerLayerComponent
      l0 = C.head (modelLayers dec)

      MultiHeadAttentionComponent{heads = hs, mWo = woSlices, rmsAtt = rmsAtt0} = multiHeadAttention l0

      ffn0 = feedforwardNetwork l0
      FeedForwardNetworkComponent{ fRMSFfn = fRMSFfn0, fW1 = w1, fW2 = w2, fW3 = w3 } = ffn0

      x   = vocabRows C.!! tok
      xHat = rmsNorm x rmsAtt0

      -- Per-head QKV from xHat; apply RoPE to Q & K only. At pos=0, attn output == V.
      qkv h = runSingleHeadQKV (hs C.!! h) xHat
      qkRot h =
        let (q0,k0,_) = qkv h
            (qR,kR)   = applyRotaryToHead (hs C.!! h) (StepCount 0) (q0,k0)
        in (qR,kR)

      vHead h =
        let (_,_,v0) = qkv h
        in v0

      vHeads :: C.Vec NumQueryHeads (C.Vec HeadDimension Float)
      vHeads = C.map vHead C.indicesI

      concatHead :: C.Vec ModelDim Float
      concatHead = flattenHeads vHeads

      -- WO @ concat_head: do it equivalently to C: sum_h (WO_slice_h @ v_h)
      perHeadProj :: C.Vec NumQueryHeads (C.Vec ModelDim Float)
      perHeadProj = C.zipWith matrixVectorMult woSlices vHeads

      yAttn :: C.Vec ModelDim Float
      yAttn = C.foldl1 (C.zipWith (+)) perHeadProj

      xAfterAttn = C.zipWith (+) x yAttn
      xHatFfn   = rmsNorm xAfterAttn fRMSFfn0
      gatePre   = matrixVectorMult w1 xHatFfn
      upPre     = matrixVectorMult w3 xHatFfn
      gate      = C.map (\z -> z / (1 + exp (-z))) gatePre  -- SiLU
      ffnCore   = matrixVectorMult w2 (C.zipWith (*) gate upPre)
      xAfterFfn = C.zipWith (+) xAfterAttn ffnCore
      xFin   = rmsNorm xAfterFfn rmsFin
      logits = C.map (`dotProduct` xFin) vocabRows
      top5   = take 5
             $ reverse
             $ DL.sortOn snd
             $ zip [0..] (C.toList logits)

  dump8 "[L0 P0] (a) xHat:" xHat
  dump8 "[L0 P0] (b) concat_head:" concatHead
  dump8 "[L0 P0] (c) WO@head_concat:" yAttn
  dump8 "[L0 P0] (d) x_after_attn:" xAfterAttn

  dump8 "[L0 P0] (e) xHat_ffn:" xHatFfn
  dump8 "[L0 P0] (f) W1*xHat_ffn:" gatePre
  dump8 "[L0 P0] (g) W3*xHat_ffn:" upPre
  dump8 "[L0 P0] (h) ffn_core:" ffnCore
  dump8 "[L0 P0] (i) x_after_ffn:" xAfterFfn

  putStrLn $ "[P0] top5 logits (id, val): " ++ show top5

-- Pure L0, P1 check: recompute K/V for t=0 and t=1, then attention over both
traceL0P1 :: TransformerDecoderComponent -> Token -> Token -> IO ()
traceL0P1 dec t0 t1 = do
  let EmbeddingComponent{vocabulary=CArray2D vocab} = modelEmbedding dec
      l0 = C.head (modelLayers dec)
      mha = multiHeadAttention l0
      rmsA = rmsAtt mha

      x0    = vocab C.!! fromIntegral t0
      x1    = vocab C.!! fromIntegral t1
      xHat0 = rmsNorm x0 rmsA
      xHat1 = rmsNorm x1 rmsA

      -- per-head QKV at t=0 and t=1
      qkv h v = runSingleHeadQKV (heads mha C.!! h) v
      rot h s (q,k) = applyRotaryToHead (heads mha C.!! h) (StepCount s) (q,k)

      -- build per-head sequences of K,V: [t0, t1]
      kSeq h = let (_,k0,_) = qkv h xHat0
                   (_,k1,_) = qkv h xHat1
                   (_,k0r)  = rot h 0 (C.repeat 0, k0)
                   (_,k1r)  = rot h 1 (C.repeat 0, k1)
               in C.replace 0 k0r $ C.replace 1 k1r $ C.repeat (C.repeat 0)

      vSeq h = let (_,_,v0) = qkv h xHat0
                   (_,_,v1) = qkv h xHat1
               in C.replace 0 v0  $ C.replace 1 v1  $ C.repeat (C.repeat 0)

      q1 h   = let (q,_,_) = qkv h xHat1
                   (qr,_)  = rot h 1 (q, C.repeat 0)
               in qr

      -- compute per-head outputs at pos=1 using your pure helpers
      headOut h =
        let
          scores  = computeAttentionScores (q1 h) (kSeq h) -- keys[t]
          weights = computeAttentionWeights (C.replace 2 (negate 1e9) scores) -- only t<=1 matter; rest masked
          out     = computeAttentionOutput weights (vSeq h)
        in out

      vHeadsOut = C.map headOut C.indicesI
      yAttn     = C.foldl1 (C.zipWith (+))
                $ C.zipWith matrixVectorMult (mWo mha) vHeadsOut
      xAfterAttn = C.zipWith (+) x1 yAttn

  dump8 "[L0 P1] (a) xHat1:" xHat1
  let
    attnScoresL0P1 :: C.Vec SeqLen Float
    attnScoresL0P1 = computeAttentionScores (q1 0) (kSeq 0)
    attnScoresL0P1List = C.toList attnScoresL0P1
  dump8L "[L0 P1] (b) head0 scores t0..t1:" $ take 2 attnScoresL0P1List
  dump8 "[L0 P1] (c) WO@concat:" yAttn
  dump8 "[L0 P1] (d) x_after_attn:" xAfterAttn
