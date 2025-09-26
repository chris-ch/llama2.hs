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
import Model.Core.Types
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
import qualified Model.Top as Top ( topEntity )
import qualified Tokenizer as T (buildTokenizer, encodeTokens, Tokenizer, decodePiece)
import Model.Layers.TransformerLayer (TransformerDecoderComponent (..), TransformerLayerComponent (..))
import qualified Model.Layers.FeedForward.FeedForwardNetwork as FeedForwardNetwork
import qualified Model.Layers.Attention.MultiHeadAttention as MultiHeadAttention
import GHC.Base (when)

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
    
  -- Handle prompt tokenization more carefully
  promptTokens <- case prompt of
    Nothing -> do
      -- No prompt provided, start with BOS token only
      putStrLn "No prompt provided, starting with BOS token (1)"
      return [1]
    Just promptStr -> do
      let promptTokensI = T.encodeTokens tokenizer (BSC.pack promptStr) True False
          promptTokens' = map fromIntegral promptTokensI :: [Token]
      case promptTokens' of
        [] -> do
          -- Empty tokenization, ensure we have BOS
          putStrLn "Empty tokenization, adding BOS token (1)"
          return [1]
        (1:_) -> do
          -- Already starts with BOS, good
          putStrLn $ "Prompt already starts with BOS: " ++ show promptTokens'
          return promptTokens'
        _ -> do
          -- Doesn't start with BOS, prepend it
          putStrLn $ "Prepending BOS to prompt tokens: " ++ show (1 : promptTokens')
          return (1 : promptTokens')

  putStrLn "✅ model loaded successfully"

  let CArray2D embedding' = vocabulary $ modelEmbedding config
  let rmsAtt0  = MultiHeadAttention.rmsAtt $ multiHeadAttention $ modelLayers config C.!! (1 :: Int)    
  let rmsFfn0  = FeedForwardNetwork.fRMSFfn $ feedforwardNetwork $ modelLayers config C.!! (1 :: Int)
  let rmsFinal = rmsFinalWeight $ modelEmbedding config       -- final RMS weight
  putStrLn $ "embedding vector 0 (first 10): " ++ show (take 10 $ C.toList $ embedding' C.!! (0::Int))
  putStrLn $ "embedding vector 1 (first 10): " ++ show (take 10 $ C.toList $ embedding' C.!! (1::Int))
  putStrLn $ "Layer0 RMS-Attn weight (first 10): "
          ++ show (take 10 (C.toList rmsAtt0))

  putStrLn $ "Layer0 RMS-FFN weight (first 10): "
          ++ show (take 10 (C.toList rmsFfn0))

  putStrLn $ "Final RMS weight (first 10): "
          ++ show (take 10 (C.toList rmsFinal))
  
  putStrLn "<s>"
  startTime <- getPOSIXTime

  (_, MultiHeadAttention.StepCount countTokens) <- generateTokensSimAutoregressive config tokenizer (fromIntegral steps) promptTokens temperature seedValue

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

printToken :: Token -> T.Tokenizer -> IO ()
printToken tok tokenizer = do
    BSC.putStr (T.decodePiece tokenizer (fromIntegral tok) (fromIntegral tok))
    hFlush stdout

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
  -> IO ([Token], MultiHeadAttention.StepCount)
generateTokensSimAutoregressive decoder tokenizer nSteps promptTokens temperature seed = do
  putStrLn $ "✅ Prompt: " ++ show promptTokens
  putStr "<s>\n"
  putStr "Generated: "
  hFlush stdout

  let temps = repeat temperature
      seeds = repeat seed

      (cur0, restPrompt0) =
        case promptTokens of
          (t0:ts) -> (t0, ts)
          []      -> (1,  [])  -- BOS fallback

      -- Run DUT; inputs depend on outputs only through tails (breaks <<loop>>)
      outputs :: [( Token, Bool, Bool
                  , C.Index NumLayers, C.Index SeqLen
                  , C.Vec ModelDim Float
                  , C.Vec ModelDim Float
                  , C.Vec ModelDim Float
                  , C.Vec ModelDim Float
                  , C.Vec ModelDim Float
                  , C.Vec ModelDim Float)]
      outputs =
        CS.simulate (bundledOutputs decoder)
                    (DL.zip4 tokenStream inputValids temps seeds)

      outTokens  = [ t | (t,_,_,_,_,_,_,_,_,_,_) <- outputs ]
      readyFlags = [ r | (_,r,_,_,_,_,_,_,_,_,_) <- outputs ]

      -- State per cycle n>=0: (currentToken, remainingPrompt, usingPrompt?)
      -- Cycle 0 is seeded; tail states are built from (ready, sampled) tails
      rTail = drop 1 readyFlags
      oTail = drop 1 outTokens

      step :: (Token, [Token], Bool) -> (Bool, Token) -> (Token, [Token], Bool)
      step (cur, ps, usingP) (r, o)
        | not r     = (cur, ps, usingP)                  -- hold until ready
        | otherwise = case ps of
            (p:ps') -> (p, ps', True)                    -- still consuming prompt
            []      -> (o, [],  False)                   -- switch to sampled

      statesTail = Prelude.scanl step (cur0, restPrompt0, True) (Prelude.zip rTail oTail)

      tokenStream  = cur0  : [ cur  | (cur,_,_) <- statesTail ]
      inputValids  = True  : [ useP | (_,_,useP) <- statesTail ]

      tapFlags       = [ tp | (_,_,tp,_,_,_,_,_,_,_,_) <- outputs ]
      tapLayers      = [ l  | (_,_,_,l,_,_,_,_,_,_,_) <- outputs ]
      tapSeqs        = [ p  | (_,_,_,_,p,_,_,_,_,_,_) <- outputs ]
      dbgXHats       = [ xv | (_,_,_,_,_,xv,_,_,_,_,_) <- outputs ]
      dbgConcatHeads = [ ch | (_,_,_,_,_,_,ch,_,_,_,_) <- outputs ]
      dbgWOs         = [ wv | (_,_,_,_,_,_,_,wv,_,_,_) <- outputs ]
      dbgXAfters     = [ av | (_,_,_,_,_,_,_,_,av,_,_) <- outputs ]
      dbgKAtPos      = [ k  | (_,_,_,_,_,_,_,_,_,k,_) <- outputs ]
      dbgVAtPos      = [ v  | (_,_,_,_,_,_,_,_,_,_,v) <- outputs ]

      sampledAll :: [Token]
      sampledAll = [ t | (t,r,_,_,_,_,_,_,_,_,_) <- outputs, r ]

  -- Pretty-print taps
  let fmt8 :: C.Vec n Float -> String
      fmt8 v = unwords (Prelude.map show (Prelude.take 8 (C.toList v)))

      succIdx :: forall n. (Bounded (C.Index n), Enum (C.Index n)) => C.Index n -> C.Index n
      succIdx p = if p == maxBound then p else succ p

      showPos :: C.Index NumLayers -> C.Index SeqLen -> (Int, Int)
      showPos l p =
        let lInt = fromEnum l
            pAdj = if l == maxBound then succIdx p else p
        in (lInt, fromEnum pAdj)

  let tokens = Prelude.map (\(t,_,_,_,_,_,_,_,_,_,_) -> t) outputs
  mapM_
    (\(tok, tp, l, p, xhat, ch, woh, xaa, k, v) -> do
        when tp $ do
          let (lI, pI) = showPos l p
          let decoded = T.decodePiece tokenizer (fromIntegral tok) (fromIntegral tok)
          putStr $ "[L" ++ show lI ++ " P" ++ show pI ++ "] "
          putStr $ "READY=" ++ show tp ++ " "
          putStr $ "token=" ++ show tok ++ " (" ++ BSC.unpack decoded ++ ") "
          putStrLn $ "xHat=" ++ fmt8 xhat
          putStr $ "[L" ++ show lI ++ " P" ++ show pI ++ "] "
          putStrLn $ "k=" ++ fmt8 k
          putStr $ "[L" ++ show lI ++ " P" ++ show pI ++ "] "
          putStrLn $ "v=" ++ fmt8 v
          putStr $ "[L" ++ show lI ++ " P" ++ show pI ++ "] "
          putStrLn $ "Concat@heads=" ++ fmt8 ch
          putStr $ "[L" ++ show lI ++ " P" ++ show pI ++ "] "
          putStrLn $ "WO@heads=" ++ fmt8 woh
          putStr $ "[L" ++ show lI ++ " P" ++ show pI ++ "] "
          putStrLn $ "x_after_attn=" ++ fmt8 xaa
          hFlush stdout
    )
    (zip10 tokens tapFlags tapLayers tapSeqs dbgXHats dbgConcatHeads dbgWOs dbgXAfters dbgKAtPos dbgVAtPos)

  -- Emit prompt verbatim, then sampled tokens after the prompt
  let promptLen          = length promptTokens
      forcedEmitted      = promptTokens
      sampledAfterPrompt = drop promptLen sampledAll
      emittedAll         = forcedEmitted ++ sampledAfterPrompt
      totalWanted        = promptLen + fromIntegral nSteps
      emittedLimited     = take totalWanted emittedAll

  let streamTokens :: [Token] -> IO ()
      streamTokens []     = pure ()
      streamTokens (t:ts) = printToken t tokenizer >> streamTokens ts

  streamTokens emittedLimited
  putStrLn ""

  let generated = take (fromIntegral nSteps) (drop promptLen emittedLimited)
  pure (generated, MultiHeadAttention.StepCount nSteps)

-- | Zip eight lists together.
--   The resulting list is as long as the shortest input list.
zip10 :: [a] -> [b] -> [c] -> [d] -> [e] -> [f] -> [g] -> [h] -> [i] -> [j] -> [(a,b,c,d,e,f,g,h,i,j)]
zip10 (a:as) (b:bs) (c:cs) (d:ds) (e:es) (f:fs) (g:gs) (h:hs) (i:is) (j:js) =
    (a,b,c,d,e,f,g,h,i,j) : zip10 as bs cs ds es fs gs hs is js
zip10 _ _ _ _ _ _ _ _ _ _ = []   -- stop when any list runs out

bundledOutputs
  :: TransformerDecoderComponent
  -> C.Signal C.System (Token, Bool, Temperature, Seed)
  -> C.Signal C.System  ( Token
                         , Bool
                         , Bool
                         , C.Index NumLayers
                         , C.Index SeqLen
                         , C.Vec ModelDim Float
                         , C.Vec ModelDim Float
                         , C.Vec ModelDim Float
                         , C.Vec ModelDim Float 
                         , C.Vec ModelDim Float
                         , C.Vec ModelDim Float 
                         )
bundledOutputs decoder =
  C.bundle . C.exposeClockResetEnable
    (topEntityBundled @CS.System decoder)
    CS.systemClockGen
    CS.resetGen
    CS.enableGen

-- Helper function to create a bundled version of topEntity
topEntityBundled :: CS.HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> C.Signal dom (Token, Bool, Temperature, Seed)
  -> ( C.Signal dom Token
     , C.Signal dom Bool
     , C.Signal dom Bool
    , C.Signal dom (C.Index NumLayers)
    , C.Signal dom (C.Index SeqLen)
    , C.Signal dom (C.Vec ModelDim Float)
    , C.Signal dom (C.Vec ModelDim Float)
    , C.Signal dom (C.Vec ModelDim Float)
    , C.Signal dom (C.Vec ModelDim Float )
    , C.Signal dom (C.Vec ModelDim Float) -- dbgKAtPos
    , C.Signal dom (C.Vec ModelDim Float) -- dbgVAtPos
    )
topEntityBundled decoder bundledInputs =
  Top.topEntity decoder inputToken inputTokenValid temp rngSeed
  where
    (inputToken, inputTokenValid, temp, rngSeed) = C.unbundle bundledInputs
