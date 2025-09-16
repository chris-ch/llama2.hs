{-# LANGUAGE StarIsType #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use tuple-section" #-}
module Model where

import qualified Prelude as P
import Clash.Prelude

import qualified Clash.Sized.Vector as CV
import qualified Data.Binary.Get as BG
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV
import qualified Foreign as F
import Control.Monad.Reader (ReaderT(runReaderT), MonadIO(liftIO), ask)
import Control.Monad.State (StateT, get, MonadState (put) )
import GHC.IO (unsafePerformIO)
import Data.ByteString (useAsCString)

import Helpers
import Control.Monad (forM, replicateM, forM_)
import Clash.Prelude (SNat(..), snatToNum)
import qualified GHC.TypeNats
import Data.Data (Proxy (..))
import Control.Monad.Identity
import Data.Functor ((<&>))

type MVectorFloat = MV.MVector (MV.PrimState IO) Float

-- Runtime type alias
type TransformerResult dom a = ReaderT TransformerDecoderComponent (StateT (DecoderCache dom) Identity) a

-- Old cache
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

-- New cache to be used
type CacheDepth = NumLayers GHC.TypeNats.* NumAttentionHeads GHC.TypeNats.* SeqLen GHC.TypeNats.* HeadDimension
type CacheAddr  = Index CacheDepth

-- RAM for keys
keyCacheRam
  :: HiddenClockResetEnable dom
  => Signal dom CacheAddr                  -- address
  -> Signal dom (Maybe (CacheAddr, Float)) -- optional write
  -> Signal dom Float                      -- output
keyCacheRam = blockRam (replicate (SNat @CacheDepth) 0)

-- RAM for values
valueCacheRam
  :: HiddenClockResetEnable dom
  => Signal dom CacheAddr
  -> Signal dom (Maybe (CacheAddr, Float))
  -> Signal dom Float
valueCacheRam = blockRam (replicate (SNat @CacheDepth) 0)

-- read a single row (step) as Vec HeadDimension Float
readRowRAM
  :: forall dom
   . ( KnownNat HeadDimension)
  =>(Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float) -- RAM function
  -> Index NumLayers
  -> Index NumAttentionHeads
  -> Index SeqLen          -- step
  -> Vec HeadDimension (Signal dom Float)
readRowRAM ram l h s = map (\d -> ram (pure $ cacheAddr l h s d) (pure Nothing)) indicesI

-- write a row into RAM at step
writeRowRAM
  :: forall dom
   . (Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float)
  -> Index NumLayers
  -> Index NumAttentionHeads
  -> Index SeqLen        -- step
  -> Vec HeadDimension Float -- data to write
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

-- Read a Vec n Float from the binary stream
readVec1D :: forall n. KnownNat n => BG.Get (Vec n Float)
readVec1D = do
    let total = snatToNum (SNat :: SNat n)
    vec <- readVector total
    return $ CV.unsafeFromList (V.toList vec)

-- New function to read a 2D fixed-size vector
readVec2D :: forall n m. (KnownNat n, KnownNat m) => BG.Get (Vec n (Vec m Float))
readVec2D = do
    let n = snatToNum (SNat :: SNat n) -- Get the outer dimension n
        m = snatToNum (SNat :: SNat m) -- Get the inner dimension m
        total = n * m -- Total number of floats to read
    vec <- readVector total -- Read all floats into a flat vector
    -- Split the flat vector into n chunks of m floats each
    let
      chunks = V.toList $ V.unfoldrN n (Just . V.splitAt m) vec
      chunks2 = fmap V.toList chunks
      -- Convert each chunk into a Vec m Float
      vecs = fmap CV.unsafeFromList chunks2
    -- Convert the list of Vec m Float into a Vec n (Vec m Float)
    return $ CV.unsafeFromList vecs

-- Read a 3D fixed-size vector (Vec n (Vec m (Vec p Float))) from the binary stream
readVec3D :: forall n m p. (KnownNat n, KnownNat m, KnownNat p) => BG.Get (Vec n (Vec m (Vec p Float)))
readVec3D = do
    let n = snatToNum (SNat :: SNat n) -- Get the outer dimension n
        m = snatToNum (SNat :: SNat m) -- Get the middle dimension m
        p = snatToNum (SNat :: SNat p) -- Get the inner dimension p
        total = n * m * p -- Total number of floats to read
    vec <- readVector total -- Read all floats into a flat vector
    -- Split the flat vector into n*m chunks of p floats each
    let
      -- First split into n*m chunks of p floats
      innerChunks = V.toList $ V.unfoldrN (n * m) (Just . V.splitAt p) vec
      innerChunks2 = fmap V.toList innerChunks
      -- Convert each chunk into a Vec p Float
      innerVecs = fmap CV.unsafeFromList innerChunks2
      -- Group the inner vectors into groups of m to form the middle dimension
      middleChunks = chunksOf m innerVecs
      -- Convert each group into a Vec m (Vec p Float)
      middleVecs = fmap CV.unsafeFromList middleChunks
    -- Convert the list of Vec m (Vec p Float) into a Vec n (Vec m (Vec p Float))
    return $ CV.unsafeFromList middleVecs
  where
    chunksOf :: Int -> [a] -> [[a]]
    chunksOf _ [] = []
    chunksOf k xs = P.take k xs : chunksOf k (P.drop k xs)

--------------------------------------------------------------------------------
-- LLM Structure
--------------------------------------------------------------------------------

-- Allocate all caches (keys + values) for every layer/head
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

-- helper: convert a Clash Vec to a list
vecToList :: CV.Vec n a -> [a]
vecToList = CV.toList

-- helper: dot-product on lists
dotList :: [Float] -> [Float] -> Float
dotList xs ys = P.sum $ P.zipWith (*) xs ys

-- softmax for lists (numerically stable)
softmaxList :: [Float] -> [Float]
softmaxList xs =
  let m = P.maximum xs
      exps = P.map (\x -> P.exp (x - m)) xs
      s = P.sum exps
  in P.map (/ s) exps

-- Read one row (headDim floats) from a flat mutable vector at position `pos`
-- mv is the flat buffer of length (seqLen * headDim)
readRowAsList :: MVectorFloat -> Int -> Int -> IO [Float]
readRowAsList mv pos headDim = do
  -- freeze into an immutable unboxed vector, then slice
  v <- V.freeze mv
  let offset = pos * headDim
  return $ V.toList $ V.slice offset headDim v

-- Load the first (step+1) rows (as lists) from the flat key/value cache
readPrefixRowsRam
  :: forall dom. (Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float)
  -> Int -- ^ step
  -> Int -- ^ headDim
  -> Int -- ^ seqLength
  -> Identity (Signal dom [[Float]])
readPrefixRowsRam ram step headDim seqLength = do
  let maxPos = P.min step (seqLength - 1)
      positions = [0 .. maxPos]
      idxPositions = P.map (fromIntegral :: Int -> Index SeqLen) positions
      addrs = P.map (\s -> cacheAddr 0 0 s 0) idxPositions
      rowSigs = P.map (\baseAddr ->
        let addrRange = P.map (\d -> baseAddr + fromIntegral d) [0 .. headDim - 1]
            rowValues = P.map (\a -> ram (pure a) (pure Nothing)) addrRange
        in sequenceA rowValues) addrs
  return $ sequenceA rowSigs

--------------------------------------------------------------------------------
-- singleHeadAttention that reads from runtime MVector caches, computes attention
-- and returns a statically-sized Vec headDim Float
--------------------------------------------------------------------------------

singleHeadAttention
  :: forall dom. HeadCache dom
  -> Vec HeadDimension Float
  -> StepCount
  -> Identity (Signal dom (Vec HeadDimension Float))
singleHeadAttention (HeadCache kcRam vcRam) qHead (StepCount stepInt) = do
  let headDim = snatToNum (SNat @HeadDimension)
      scaling = P.sqrt (fromIntegral headDim :: Float)
      qHeadList = CV.toList qHead
      seqLength = snatToNum (SNat @SeqLen)
  kRowsSig <- readPrefixRowsRam kcRam stepInt headDim seqLength
  vRowsSig <- readPrefixRowsRam vcRam stepInt headDim seqLength
  let scoresSig = kRowsSig <&> \kRows ->
        P.map (\kRow -> dotList qHeadList kRow / scaling) kRows
      attWeightsSig = scoresSig <&> softmaxList
      resultSig = liftA2 (\attWeights vRows ->
        let zeroVecList = P.replicate headDim 0.0 :: [Float]
            addWeighted acc (w, vRow) = P.zipWith (+) acc (P.map (w *) vRow)
            resultList = P.foldl addWeighted zeroVecList (P.zip attWeights vRows)
        in CV.unsafeFromList resultList :: Vec HeadDimension Float
        ) attWeightsSig vRowsSig
  return resultSig

updateCache
  :: forall dom headDim. (HiddenClockResetEnable dom, KnownNat headDim)
  => (Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float)
  -> StepCount
  -> Vec headDim Float
  -> Identity (Signal dom ())
updateCache ram (StepCount step) vec = do
  let headDim = snatToNum (SNat @headDim)
      baseAddr = cacheAddr 0 0 step 0
      vecList = CV.toList vec
      -- Generate write signals for each element
      writeSigs = zipWith (\ d val
        -> (let addr = P.baseAddr P.+ P.fromIntegral d
            in pure (P.Just (P.addr, P.val)))) [0 .. headDim - 1] vecList
      -- Sequence writes (in simulation, this is instantaneous)
      writeAction = traverse_ (Control.Monad.void . ram (pure 0)) writeSigs
  return writeAction

runLayer
  :: forall dom. HiddenClockResetEnable dom
  => MultiHeadAttentionComponent
  -> FeedForwardNetworkComponent
  -> LayerAttentionCache dom
  -> Vec ModelDim Float
  -> StepCount
  -> Identity (Signal dom (Vec ModelDim Float, LayerAttentionCache dom))
runLayer mha feedForwardNetwork layerCache inputToken step = do
  let CVector rmsWeights = rmsAtt mha
      normalizedInput = rmsNorm inputToken rmsWeights
      outputProjectionWeights = mWo mha
  -- Compute per-head QKV
  qkvList <- forM (vecToList (heads mha)) $ \headComp -> do
    let (CVector q, CVector k, v) = runSingleHeadQKV headComp normalizedInput
        (q', k') = applyRotaryToHead headComp step (q, k)
    return (q', k', v)
  let (qList, _, _) = P.unzip3 qkvList
      headCaches = multiHeadCache layerCache
  -- Update caches
  writeSigs <- forM (P.zip (vecToList headCaches) qkvList) $ \(hc, (_, k', CVector v')) -> do
    kcWrite <- updateCache (headKeyCache hc) step k'
    vcWrite <- updateCache (headValueCache hc) step v'
    return (kcWrite >> vcWrite)
  let writeAction = sequenceA writeSigs
  -- Compute attention outputs
  headOutputsSig <- forM (P.zip qList (vecToList headCaches)) $ \(qHead, hc) ->
    singleHeadAttention hc qHead step
  let multiHeadOutSig = sequenceA headOutputsSig <&> concat
      attentionDeltaSig = multiHeadOutSig <&> \multiHeadOut ->
        matrixVectorMult outputProjectionWeights multiHeadOut
      tokenAfterAttentionSig = P.fmap (zipWith (+) inputToken) attentionDeltaSig
      ffnOutSig = tokenAfterAttentionSig <&> runFeedForward feedForwardNetwork
      finalTokenSig = liftA2 (zipWith (+)) tokenAfterAttentionSig ffnOutSig
      resultSig = finalTokenSig <&> \finalToken -> (finalToken, layerCache)
  return $ writeAction >> resultSig

runLayers
  :: forall dom. HiddenClockResetEnable dom
  => Vec NumLayers TransformerLayerComponent
  -> Vec NumLayers (LayerAttentionCache dom)
  -> Vec ModelDim Float
  -> StepCount
  -> Identity (Signal dom (Vec ModelDim Float, Vec NumLayers (LayerAttentionCache dom)))
runLayers layers caches tv step = do
  case (layers, caches) of
    (Nil, Nil) -> return $ pure (tv, Nil)
    (layer :> restLayers, cache :> restCaches) -> do
      let mha = multiHeadAttention layer
          ffn = feedforwardNetwork layer
      layerResultSig <- runLayer mha ffn cache tv step
      let (tvSig, cacheSig) = layerResultSig <&> \(tv', cache') -> (tv', cache')
      (restTvSig, restCachesSig) <- runLayers restLayers restCaches tv step
      return $ liftA2 (\(tv', cache') (restTv, restCaches') ->
        (restTv, cache' :> restCaches')) layerResultSig restCachesSig
    _ -> fail "Mismatch: Layers and caches must have the same length"

--------------------------------------------------------------------------------
-- Transformer runtime
--------------------------------------------------------------------------------

-- Transformer step
transformer
  :: forall dom. HiddenClockResetEnable dom
  => Helpers.Token
  -> StepCount
  -> TransformerResult dom (Signal dom (Vec VocabSize Float))
transformer inputTokenCode stepCount = do
  decoder <- ask
  decoderCache <- get
  let embeddingLayer = modelEmbedding decoder
      layers = modelLayers decoder
      inputTokenVector = embed (vocabulary embeddingLayer) inputTokenCode
  resultSig <- runLayers layers (layerCache decoderCache) inputTokenVector stepCount
  let (outputTokenVectorSig, updatedCacheSig) = resultSig <&> \(tv, caches) -> (tv, caches)
  put $ DecoderCache { layerCache = updatedCacheSig }
  return $ outputTokenVectorSig <&> transformerLogits decoder
--------------------------------------------------------------------------------
-- Binary Parsing
--------------------------------------------------------------------------------
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
  wq' <- readVec3D @NumLayers @ModelDim @ModelDim
  wk' <- readVec3D @NumLayers @ModelDim @ModelDim
  wv' <- readVec3D @NumLayers @ModelDim @ModelDim
  wo' <- readVec3D @NumLayers @ModelDim @ModelDim
  rmsFfnWeight' <- readVec2D @NumLayers
  w1' <- readVec3D @NumLayers @HiddenDim @ModelDim
  w2' <- readVec3D @NumLayers @ModelDim @HiddenDim
  w3' <- readVec3D @NumLayers @HiddenDim @ModelDim
  rmsFinalWeight' <- readVec1D @ModelDim
  freqCisReal' <- readVec2D @SeqLen
  freqCisImag' <- readVec2D @SeqLen

  let
      -- Construct the Embedding
      embedding = EmbeddingComponent
        { vocabulary =  CArray2D tokenEmbeddingTable',
          rmsFinalWeight = CVector rmsFinalWeight'
        }
      -- Construct the parameters for Transformer layers
      sha hIdx = SingleHeadComponent
                          { wqHead = CArray2D $ wq' !! fromIntegral (toInteger hIdx)
                          , wkHead = CArray2D $ wk' !! fromIntegral (toInteger hIdx)
                          , wvHead = CArray2D $ wv' !! fromIntegral (toInteger hIdx)
                          , rotary = RotaryEncodingComponent { freqCos = CArray2D freqCisReal'
                                                            , freqSin = CArray2D freqCisImag' }
                          }
      layer lIdx = TransformerLayerComponent {
          multiHeadAttention = MultiHeadAttentionComponent
                {
                  heads = map sha (indicesI :: Vec NumAttentionHeads (Index NumAttentionHeads)) :: Vec NumAttentionHeads SingleHeadComponent
                , mWo     = CArray2D $ wo' !! lIdx
                , rmsAtt = CVector $ rmsAttWeight' !! lIdx
                },
            feedforwardNetwork = FeedForwardNetworkComponent
                     { fW1 = CArray2D $ w1' !! fromIntegral (toInteger lIdx),
                       fW2 = CArray2D $ w2' !! fromIntegral (toInteger lIdx),
                       fW3 = CArray2D $ w3' !! fromIntegral (toInteger lIdx),
                       fRMSFfn = CVector $ rmsFfnWeight' !! lIdx
                     }
            }
      -- Construct the TransformerArchitecture
      decoder = TransformerDecoderComponent
              { modelEmbedding = embedding,
                modelLayers = map layer (indicesI :: Vec NumLayers (Index NumLayers)) :: Vec NumLayers TransformerLayerComponent
              }
  return decoder
