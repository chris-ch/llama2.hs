module Model where

import qualified Prelude as P
import Clash.Prelude
import qualified Clash.Sized.Vector as CV
import qualified Data.Binary.Get as BG
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV
import qualified Foreign as F
import qualified Control.Monad.Reader as MR
import Control.Monad.State (StateT, get, MonadState (put))
import GHC.IO (unsafePerformIO)
import Data.ByteString (useAsCString)
import Helpers
import qualified GHC.TypeNats
import Data.Functor ((<&>), ($>))
import qualified System.Random as R
import qualified Clash.Explicit.SimIO as SimIO

type MVectorFloat = MV.MVector (MV.PrimState IO) Float

type TransformerResult' dom a = MR.ReaderT TransformerDecoderComponent (StateT (DecoderCache dom) IO) a
type TransformerResult dom a = MR.ReaderT TransformerDecoderComponent (StateT (DecoderCache dom) SimIO.SimIO) a

data HeadCache dom = HeadCache
  { headKeyCache :: Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float
  , headValueCache :: Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float
  }

newtype LayerAttentionCache dom = LayerAttentionCache
  { multiHeadCache :: Vec NumAttentionHeads (HeadCache dom)
  }

newtype DecoderCache dom = DecoderCache
  { layerCache :: Vec NumLayers (LayerAttentionCache dom)
  }

type CacheDepth = NumLayers GHC.TypeNats.* NumAttentionHeads GHC.TypeNats.* SeqLen GHC.TypeNats.* HeadDimension
type CacheAddr  = Index CacheDepth

keyCacheRam
  :: HiddenClockResetEnable dom
  => Signal dom CacheAddr
  -> Signal dom (Maybe (CacheAddr, Float))
  -> Signal dom Float
keyCacheRam = blockRam (replicate (SNat @CacheDepth) 0)

valueCacheRam
  :: HiddenClockResetEnable dom
  => Signal dom CacheAddr
  -> Signal dom (Maybe (CacheAddr, Float))
  -> Signal dom Float
valueCacheRam = blockRam (replicate (SNat @CacheDepth) 0)

readRowRAM
  :: forall dom
   . (KnownNat HeadDimension)
  => (Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float)
  -> Index NumLayers
  -> Index NumAttentionHeads
  -> Index SeqLen
  -> Vec HeadDimension (Signal dom Float)
readRowRAM ram l h s = map (\d -> ram (pure $ cacheAddr l h s d) (pure Nothing)) indicesI

writeRowRAM
  :: forall dom
   . (Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float)
  -> Index NumLayers
  -> Index NumAttentionHeads
  -> Index SeqLen
  -> Vec HeadDimension Float
  -> Vec HeadDimension (Maybe (CacheAddr, Float))
writeRowRAM _ram l h s = zipWith (\d val -> Just (cacheAddr l h s d, val)) indicesI

cacheAddr
  :: Index NumLayers
  -> Index NumAttentionHeads
  -> Index SeqLen
  -> Index HeadDimension
  -> Index CacheDepth
cacheAddr l h s d =
  let
    headSize = natToNum @HeadDimension
    seqSize  = natToNum @SeqLen * headSize
    layerSize = natToNum @NumAttentionHeads * seqSize
    addr =
      fromIntegral d
      + fromIntegral s * headSize
      + fromIntegral h * seqSize
      + fromIntegral l * layerSize
  in addr

readVector :: Int -> BG.Get (V.Vector Float)
readVector count = do
  byteData <- BG.getByteString (count * 4)
  return $! unsafePerformIO $ do
    useAsCString byteData $ \ptr -> do
      let floatPtr = F.castPtr ptr :: F.Ptr Float
      V.generateM count (F.peekElemOff floatPtr)

readVec1D :: forall n. KnownNat n => BG.Get (Vec n Float)
readVec1D = do
    let total = snatToNum (SNat :: SNat n)
    vec <- readVector total
    return $ CV.unsafeFromList (V.toList vec)


-- Fixed readVec2D function
readVec2D :: forall n m. (KnownNat n, KnownNat m) => BG.Get (Vec n (Vec m Float))
readVec2D = do
    let n = snatToNum (SNat :: SNat n)
        m = snatToNum (SNat :: SNat m)
        total = n * m
    vec <- readVector total  -- this returns V.Vector Float (unboxed)
    let -- Convert to list and then chunk it
        floatList = V.toList vec
        chunks = chunksOf m floatList
        vecs = P.map CV.unsafeFromList chunks
    return $ CV.unsafeFromList vecs
  where
    chunksOf :: Int -> [a] -> [[a]]
    chunksOf _ [] = []
    chunksOf k xs = P.take k xs : chunksOf k (P.drop k xs)

readVec3D :: forall n m p. (KnownNat n, KnownNat m, KnownNat p) => BG.Get (Vec n (Vec m (Vec p Float)))
readVec3D = do
    let n = snatToNum (SNat :: SNat n)
        m = snatToNum (SNat :: SNat m)
        p = snatToNum (SNat :: SNat p)
        total = n * m * p
    vec <- readVector total
    let
      floatList = V.toList vec
      innerChunks = chunksOf p floatList
      innerVecs = P.map CV.unsafeFromList innerChunks
      middleChunks = chunksOf m innerVecs
      middleVecs = P.map CV.unsafeFromList middleChunks
    return $ CV.unsafeFromList middleVecs
  where
    chunksOf :: Int -> [a] -> [[a]]
    chunksOf _ [] = []
    chunksOf k xs = P.take k xs : chunksOf k (P.drop k xs)

initDecoderCaches
  :: forall dom. HiddenClockResetEnable dom
  => IO (DecoderCache dom)
initDecoderCaches = do
  let layerCaches = CV.repeat @NumLayers $
        LayerAttentionCache $
          CV.repeat @NumAttentionHeads $
            HeadCache
              { headKeyCache = keyCacheRam
              , headValueCache = valueCacheRam
              }
  return $ DecoderCache { layerCache = layerCaches }

vecToList :: CV.Vec n a -> [a]
vecToList = CV.toList

dotVec :: forall n. KnownNat n => Vec n Float -> Vec n Float -> Float
dotVec xs ys = sum (zipWith (*) xs ys)

softmaxVec :: forall n. KnownNat (n+1) => Vec (n+1) Float -> Vec (n+1) Float
softmaxVec xs =
  let m = maximum xs
      exps = map (\x -> P.exp (x - m)) xs
      s = sum exps
  in map (/ s) exps

readRowAsList :: MVectorFloat -> Int -> Int -> IO [Float]
readRowAsList mv pos headDim = do
  v <- V.freeze mv
  let offset = pos * headDim
  return $ V.toList $ V.slice offset headDim v

readRow
  :: forall dom.
     ( KnownNat HeadDimension )
  => (Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float)
  -> Index NumLayers
  -> Index NumAttentionHeads
  -> Index SeqLen
  -> Signal dom (Vec HeadDimension Float)
readRow ram l h s =
  P.traverse (\d -> ram (pure $ cacheAddr l h s d) (pure Nothing)) indicesI

readPrefixRowsRam
  :: forall dom.
     ( KnownNat SeqLen
     , KnownNat HeadDimension )
  =>(Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float)
  -> Index NumLayers
  -> Index NumAttentionHeads
  -> Signal dom (Vec SeqLen (Vec HeadDimension Float))
readPrefixRowsRam ram l h = do
  let seqIndices = indicesI :: Vec SeqLen (Index SeqLen)
      rows = map (readRow ram l h) seqIndices
  sequenceA rows

updateCache :: (Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float)
            -> StepCount
            -> Vec HeadDimension Float
            -> Signal dom ()
updateCache ram (StepCount step) vec =
  let writes = CV.map (\d ->
                let addr = cacheAddr 0 0 (fromIntegral step :: Index SeqLen) d
                in ram (pure addr) (pure $ Just (addr, vec !! d))
              ) indicesI
  in sequenceA writes $> ()  -- sequence into a signal and discard values

singleHeadAttention
  :: forall dom. HeadCache dom
  -> Index NumAttentionHeads        -- head index
  -> Vec HeadDimension Float       -- qHead (Vec, not list)
  -> StepCount
  -> Signal dom (Vec HeadDimension Float)
singleHeadAttention (HeadCache kcRam vcRam) hIdx qHead (StepCount stepInt) =
  let headDim = snatToNum (SNat @HeadDimension)
      scaling = P.sqrt (headDim :: Float)
      layerIdx = fromIntegral stepInt :: Index NumLayers

      -- readPrefixRowsRam already returns a Signal, so bind with let (not <-)
      kRowsSig :: Signal dom (Vec SeqLen (Vec HeadDimension Float))
      kRowsSig = readPrefixRowsRam kcRam layerIdx hIdx

      vRowsSig :: Signal dom (Vec SeqLen (Vec HeadDimension Float))
      vRowsSig = readPrefixRowsRam vcRam layerIdx hIdx

      -- compute scores as a Vec SeqLen Float inside the Signal
      scoresSig :: Signal dom (Vec SeqLen Float)
      scoresSig = fmap ((\q kRows -> map (\kRow -> dotVec q kRow / scaling) kRows) qHead) kRowsSig

      attWeightsSig :: Signal dom (Vec SeqLen Float)
      attWeightsSig = fmap softmaxVec scoresSig

      resultSig :: Signal dom (Vec HeadDimension Float)
      resultSig = liftA2 (\attWeights vRows ->
        -- weighted sum over sequence
        let zeroVec = repeat 0 :: Vec HeadDimension Float
            weightedAdd acc (w, vrow) = zipWith (+) acc (map (* w) vrow)
            pairs = zip attWeights vRows :: Vec SeqLen (Float, Vec HeadDimension Float)
        in foldl weightedAdd zeroVec pairs
        ) attWeightsSig vRowsSig
  in resultSig


flattenVec :: Vec n (Vec m a) -> Vec (n GHC.TypeNats.* m) a
flattenVec = concat

runLayer
  :: forall dom. (Monad (Signal dom))
  =>MultiHeadAttentionComponent
  -> FeedForwardNetworkComponent
  -> LayerAttentionCache dom
  -> Vec ModelDim Float
  -> StepCount
  -> Signal dom (Vec ModelDim Float)
runLayer mha feedForwardNetwork layerCache inputToken step = do
  let rmsWeights = rmsAtt mha
      normalizedInput = rmsNorm inputToken rmsWeights
      outputProjectionWeights = mWo mha
      headCaches = multiHeadCache layerCache

  -- Compute QKV and update caches for each head (Vec-based)
  let qkvResults :: Vec NumAttentionHeads (Vec HeadDimension Float, HeadCache dom)
      qkvResults = imap (\hIdx hc ->
                          let headComp = heads mha !! hIdx
                              (q, k, _) = runSingleHeadQKV headComp normalizedInput
                              (q', _) = applyRotaryToHead headComp step (q, k)
                          in (q', hc)
                        ) headCaches

  -- Write keys/values to cache (side-effecting signals)
  let cacheWrites :: Vec NumAttentionHeads (Signal dom ())
      cacheWrites = imap (\hIdx (_, hc) ->
                            let HeadCache kcRam vcRam = hc
                                (_, k, v) = runSingleHeadQKV (heads mha !! hIdx) normalizedInput
                                (_, k') = applyRotaryToHead (heads mha !! hIdx) step (k, k)
                                kcWrite = updateCache kcRam step k'
                                vcWrite = updateCache vcRam step v
                            in kcWrite *> vcWrite
                        ) qkvResults

  -- Combine all cache write signals into one (ignored output)
  _ <- sequenceA cacheWrites

  -- Compute attention output for each head
  let headOutputsSig :: Vec NumAttentionHeads (Signal dom (Vec HeadDimension Float))
      headOutputsSig = imap (\hIdx (qHead, hc) -> singleHeadAttention hc hIdx qHead step) qkvResults

      -- Sequence Vec of signals -> Signal of Vec
      multiHeadOutSig :: Signal dom (Vec NumAttentionHeads (Vec HeadDimension Float))
      multiHeadOutSig = sequenceA headOutputsSig

  -- Combine heads and apply output projection
  let attentionDeltaSig :: Signal dom (Vec ModelDim Float)
      attentionDeltaSig = multiHeadOutSig <&> \headOutputs ->
        let concatenatedHeads :: Vec ModelDim Float
            concatenatedHeads = flattenVec headOutputs
            headsOut = matrixVectorMult outputProjectionWeights concatenatedHeads
        in headsOut

  -- Residual connection after attention
  let tokenAfterAttentionSig = fmap (Clash.Prelude.zipWith (+) inputToken) attentionDeltaSig

  -- Apply feed-forward network
  let ffnOutSig = fmap (runFeedForward feedForwardNetwork) tokenAfterAttentionSig

  -- Final residual connection
  let finalTokenSig = liftA2 (Clash.Prelude.zipWith (+)) tokenAfterAttentionSig ffnOutSig

  finalTokenSig

runLayers
  :: forall dom. ( Monad (Signal dom))
  => Vec NumLayers TransformerLayerComponent
  -> DecoderCache dom
  -> Vec ModelDim Float
  -> StepCount
  -> Signal dom (Vec ModelDim Float)
runLayers layers decoderCache inputToken step = fmap P.fst (foldLayers inputToken layerCachePairs)
  where
    -- Extract layer caches from DecoderCache
    layerCaches = layerCache decoderCache

    -- Helper function to process a single layer
    processLayer
      :: TransformerLayerComponent
      -> LayerAttentionCache dom
      -> Vec ModelDim Float
      -> Signal dom (Vec ModelDim Float)
    processLayer layer layerCache token = runLayer (multiHeadAttention layer) (feedforwardNetwork layer) layerCache token step
    -- Fold over layers and caches to produce the final token and updated caches

    -- Fold over layers and caches to produce the final token and updated caches
    foldLayers
      :: Vec ModelDim Float
      -> Vec NumLayers (TransformerLayerComponent, LayerAttentionCache dom)
      -> Signal dom (Vec ModelDim Float, [LayerAttentionCache dom])
    foldLayers token = Clash.Prelude.foldr combineLayer (pure (token, []))
      where
        combineLayer
          :: (TransformerLayerComponent, LayerAttentionCache dom)
          -> Signal dom (Vec ModelDim Float, [LayerAttentionCache dom])
          -> Signal dom (Vec ModelDim Float, [LayerAttentionCache dom])
        combineLayer (layer, cache) accSig = do
          (nextToken, restCaches) <- accSig
          token' <- processLayer layer cache nextToken
          pure (token', restCaches)

    -- Pair layers with their corresponding caches
    layerCachePairs :: Vec NumLayers (TransformerLayerComponent, LayerAttentionCache dom)
    layerCachePairs = zip layers layerCaches

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
  let
      embedding = EmbeddingComponent
        { vocabulary = CArray2D tokenEmbeddingTable',
          rmsFinalWeight = rmsFinalWeight'
        }
      sha hIdx = SingleHeadComponent
                          { wqHead = CArray2D $ wq' !! toInteger hIdx
                          , wkHead = CArray2D $ wk' !! toInteger hIdx
                          , wvHead = CArray2D $ wv' !! toInteger hIdx
                          , rotary = RotaryEncodingComponent { freqCos = CArray2D freqCisReal'
                                                            , freqSin = CArray2D freqCisImag' }
                          }
      layer lIdx = TransformerLayerComponent {
          multiHeadAttention = MultiHeadAttentionComponent
                {
                  heads = map sha (indicesI :: Vec NumAttentionHeads (Index NumAttentionHeads)) :: Vec NumAttentionHeads SingleHeadComponent
                , mWo     = CArray2D $ wo' !! lIdx
                , rmsAtt = rmsAttWeight' !! lIdx
                },
            feedforwardNetwork = FeedForwardNetworkComponent
                     { fW1 = CArray2D $ w1' !! toInteger lIdx,
                       fW2 = CArray2D $ w2' !! toInteger lIdx,
                       fW3 = CArray2D $ w3' !! toInteger lIdx,
                       fRMSFfn = rmsFfnWeight' !! lIdx
                     }
            }
      decoder = TransformerDecoderComponent
              { modelEmbedding = embedding,
                modelLayers = map layer (indicesI :: Vec NumLayers (Index NumLayers)) :: Vec NumLayers TransformerLayerComponent
              }
  return decoder

-- Sampling
drawSample :: Int -> V.Vector Float -> IO Helpers.Token
drawSample randomSeed probabilities = do
  let gen = R.mkStdGen randomSeed
      (randomValue, _) = R.random gen :: (Float, R.StdGen)
      cumulativeDistribution = V.scanl1 (+) probabilities
      selectedIndex = V.length (V.takeWhile (< randomValue) cumulativeDistribution)
  return $ fromIntegral (min selectedIndex (V.length probabilities - 1))

-- Pure deterministic sampling from probabilities
drawSamplePure :: Int -> Vec VocabSize Float -> Int
drawSamplePure seed probabilities =
    let gen = R.mkStdGen seed
        (randomValue, _) = R.random gen :: (Float, R.StdGen)

        -- cumulative sum using scanl1'
        cumulativeDistribution :: Vec VocabSize Float
        cumulativeDistribution = CV.scanl1 (+) probabilities

        -- find first index where cumulative >= randomValue
        selectedIndex :: Index VocabSize
        selectedIndex = maybe maxBound id (findIndex (>= randomValue) cumulativeDistribution)

    in fromEnum selectedIndex

-- | Find the index of the maximum element in a non-empty vector
argMax :: forall n. (KnownNat n) => Vec n Float -> Int
argMax vec = fst $ foldl compareMax (0, vec !! 0) (imap (\i x -> (fromEnum i, x)) vec)
  where
    compareMax :: (Int, Float) -> (Int, Float) -> (Int, Float)
    compareMax (maxIdx, maxVal) (i, x)
      | x > maxVal = (i, x)
      | otherwise  = (maxIdx, maxVal)

-- Pure transformer function
transformer
  :: forall dom. Monad (Signal dom)
  => Helpers.Token        -- input token
  -> StepCount
  -> Float                -- temperature
  -> [Helpers.Token]      -- prompt tokens
  -> Int                  -- seed
  -> TransformerResult dom (Signal dom Helpers.Token)
transformer inputTokenCode stepCount temperature promptTokens seedValue = do
  decoder <- MR.ask
  decoderCache <- get

  let embeddingLayer = modelEmbedding decoder
      layers = modelLayers decoder
      inputTokenVector = embed (vocabulary embeddingLayer) inputTokenCode
      StepCount step = stepCount

      -- Run transformer layers
      outputTokenVectorSig :: Signal dom (Vec ModelDim Float)
      outputTokenVectorSig = runLayers layers decoderCache inputTokenVector stepCount

      -- Compute logits
      logitsSig :: Signal dom (Vec VocabSize Float)
      logitsSig = liftA2 (flip transformerLogits) outputTokenVectorSig (pure decoder)

      -- Select next token
      nextTokenSig :: Signal dom Helpers.Token
      nextTokenSig
        | fromIntegral step < P.length promptTokens
        = pure (promptTokens P.!! fromIntegral step)   -- use prompt token
        | temperature == 0.0
        = logitsSig <&> \logits -> Helpers.Token $ fromIntegral $ argMax logits
        | otherwise
        = logitsSig <&> \logits ->
            let scaled = map (/ temperature) logits
            in Helpers.Token $ fromIntegral $ drawSamplePure seedValue scaled

  -- Put back the unchanged cache
  put decoderCache
  return nextTokenSig
