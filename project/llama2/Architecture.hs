module Architecture
  ( EmbeddingComponent (..),
    RotaryEncodingComponent (..),
    MultiHeadAttentionComponent (..),
    FeedForwardNetworkComponent (..),
    TransformerLayerComponent (..),
    TransformerDecoderComponent (..),
    SingleHeadComponent (..), Component(..),
    transformerLogits, TransformerResult(..)
    , NetworkConfig (..)
    , embed, runModel, parseNetworkConfigFile
    , DecoderCache (..), LayerAttentionCache (..), HeadCache(..), NormRMS(..)
  ) where

import Control.Monad.Reader (MonadIO (liftIO), MonadReader (ask), ReaderT)
import Control.Monad.State (StateT, get, put, MonadState)
import Control.Monad ( forM_, forM)
import qualified Data.Binary.Get as BG
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV

import Types
    (
      Array2D(..),
      Token (..),
      TokenVector(..),
      StepCount(..), getRow, MVectorFloat, Array3D (..), readArray2D, readArray3D, readVector, getArray2D )
import Primitives
    (
      softmax, rmsNorm, sigmoidLinearUnit, matrixVectorMult )
import Data.List (foldl')

-- Runtime type alias
newtype TransformerResult a =
  TransformerResult
    { runTransformerResult :: ReaderT NetworkConfig (StateT DecoderCache IO) a
    }
  deriving (Functor, Applicative, Monad, MonadIO, MonadReader NetworkConfig, MonadState DecoderCache)

-- | Any hardware-like component
class Monad m => Component m c where
  -- The type of input to the component
  type Input c
  -- The type of output from the component
  type Output c
  -- The type of internal state (registers/memory)
  type State c

  -- Initialize state once
  initState :: c -> m (State c)

  -- One simulation “tick”:
  tick :: c -> State c -> Input c -> m (State c, Output c)

-- Data definitions mirroring architectural components

data NormRMS = NormRMS

instance Component TransformerResult NormRMS where
  type Input NormRMS  = (V.Vector Float, V.Vector Float)
  type Output NormRMS = V.Vector Float
  type State NormRMS  = ()

  initState :: NormRMS -> TransformerResult ()
  initState _ = return ()

  tick :: NormRMS -> State NormRMS -> Input NormRMS -> TransformerResult (State NormRMS, Output NormRMS)
  tick _ () (inputToken, rmsWeights) = do
    let
      normalizedInput = rmsNorm inputToken rmsWeights
    return ((), normalizedInput)
    
data EmbeddingComponent = EmbeddingComponent
  { vocabulary :: Array2D,
    rmsFinalWeight :: V.Vector Float
  } deriving (Show)

data RotaryEncodingComponent = RotaryEncodingComponent
  { freqCos :: Array2D,
    freqSin :: Array2D
  } deriving (Show)

instance Component TransformerResult RotaryEncodingComponent where
  type Input RotaryEncodingComponent  = (RotaryEncodingComponent, StepCount, V.Vector Float, V.Vector Float)
  type Output RotaryEncodingComponent = (V.Vector Float, V.Vector Float)
  type State RotaryEncodingComponent  = ()

  initState :: RotaryEncodingComponent -> TransformerResult ()
  initState _ = return ()

  tick :: RotaryEncodingComponent -> State RotaryEncodingComponent -> Input RotaryEncodingComponent -> TransformerResult (State RotaryEncodingComponent, Output RotaryEncodingComponent)
  tick _ () (rotary, stepCount, q, k) = do
    q' <- applyRotary rotary stepCount q
    k' <- applyRotary rotary stepCount k
    return ((), (q', k'))

data SingleHeadComponent = SingleHeadComponent
  { wqHead :: Array2D    -- size headDim x modelDim
  , wkHead :: Array2D
  , wvHead :: Array2D
  , rotary :: RotaryEncodingComponent
  } deriving (Show)

data MultiHeadAttentionComponent = MultiHeadAttentionComponent
  { heads    :: [SingleHeadComponent]
  , mWo      :: Array2D         -- Output projection matrix
  , rmsAtt   :: V.Vector Float  -- RMSNorm before QKV projection
  , nrms     :: NormRMS
  }

data FeedForwardNetworkComponent = FeedForwardNetworkComponent
  { fW1 :: Array2D,
    fW2 :: Array2D,
    fW3 :: Array2D,
    fRMSFfn :: V.Vector Float
  } deriving (Show)

instance Component TransformerResult FeedForwardNetworkComponent where
  type Input FeedForwardNetworkComponent = V.Vector Float
  type Output FeedForwardNetworkComponent = V.Vector Float
  type State FeedForwardNetworkComponent = ()

  initState :: FeedForwardNetworkComponent -> TransformerResult ()
  initState _ = do
    return ()

  tick :: FeedForwardNetworkComponent -> State FeedForwardNetworkComponent -> Input FeedForwardNetworkComponent -> TransformerResult (State FeedForwardNetworkComponent, Output FeedForwardNetworkComponent)
  tick ffn () inputToken = do
    let
      -- (this is your runFeedForward logic, purely functional)
      rmsFfnWeights = fRMSFfn ffn
      w1 = fW1 ffn
      w2 = fW2 ffn
      w3 = fW3 ffn
      normalizedInput = rmsNorm inputToken rmsFfnWeights
      gateOutput' = matrixVectorMult w1 normalizedInput
      upProjectionOutput' = matrixVectorMult w3 normalizedInput
      gateOutput = V.map sigmoidLinearUnit gateOutput'
      feedforwardNetworkOutput' =
        matrixVectorMult w2 (V.zipWith (*) gateOutput upProjectionOutput')
    return ((), feedforwardNetworkOutput')

instance Component TransformerResult SingleHeadComponent where
  type Input SingleHeadComponent  = (V.Vector Float, StepCount, Int)
  type Output SingleHeadComponent = TokenVector
  type State SingleHeadComponent  = HeadCache

  -- create empty key/value caches per head
  initState :: SingleHeadComponent -> TransformerResult HeadCache
  initState _ = do
    -- get model config
    network <- ask
    let cacheSize = seqLen network * headDimension network
    -- allocate mutable vectors
    kc <- liftIO $ MV.replicate cacheSize 0.0
    vc <- liftIO $ MV.replicate cacheSize 0.0
    return HeadCache { headKeyCache = kc, headValueCache = vc }

  tick :: SingleHeadComponent -> State SingleHeadComponent -> Input SingleHeadComponent -> TransformerResult (State SingleHeadComponent, Output SingleHeadComponent)
  tick headComp hc (inputToken, stepCount, headDim) = do
    let q = matrixVectorMult (wqHead headComp) inputToken
        k = matrixVectorMult (wkHead headComp) inputToken
        v = matrixVectorMult (wvHead headComp) inputToken

    q' <- applyRotary (rotary headComp) stepCount q
    k' <- applyRotary (rotary headComp) stepCount k

    updateCache (headKeyCache hc) stepCount headDim k'
    updateCache (headValueCache hc) stepCount headDim v

    tokenOut <- headAttentionPerHead hc q' stepCount headDim

    return (hc, tokenOut)

data TransformerLayerComponent = TransformerLayerComponent
  {
    multiHeadAttention :: MultiHeadAttentionComponent,
    feedforwardNetwork :: FeedForwardNetworkComponent
  }

data TransformerDecoderComponent = TransformerDecoderComponent
  { modelEmbedding :: EmbeddingComponent,
    modelLayers :: [TransformerLayerComponent]
  }

data HeadCache = HeadCache
  { headKeyCache   :: MVectorFloat  -- Size: seqLen * headDim
  , headValueCache :: MVectorFloat  -- Size: seqLen * headDim
  }

newtype LayerAttentionCache = LayerAttentionCache
  { headCaches :: [HeadCache]
  }

newtype DecoderCache = DecoderCaches
  { layerCaches :: [LayerAttentionCache]
  }

runModel :: TransformerDecoderComponent -> Token -> StepCount -> TransformerResult TokenVector
runModel dec tokenCode stepCount = do
  let embeddingLayer = modelEmbedding dec
      layers = modelLayers dec
  tokenVector <- embed embeddingLayer tokenCode
  decoderCaches <- get
  (finalTokenVector, finalCaches) <- runLayers layers (layerCaches decoderCaches) tokenVector stepCount
  put $ DecoderCaches { layerCaches = finalCaches }
  return finalTokenVector

runLayers :: [TransformerLayerComponent] -> [LayerAttentionCache] -> TokenVector -> StepCount -> TransformerResult (TokenVector, [LayerAttentionCache])
runLayers [] [] tv _ = return (tv, [])
runLayers (layer : restLayers) (layerCaches : restCaches) tv step = do
  let mha = multiHeadAttention layer
      ffn = feedforwardNetwork layer
  (tv', updatedLayerCaches) <- runLayer mha ffn layerCaches tv step
  (tv'', restUpdated) <- runLayers restLayers restCaches tv' step
  return (tv'', updatedLayerCaches : restUpdated)
runLayers (_ : _) [] _ _ = liftIO $ fail "Mismatch: Non-empty layers but empty caches"
runLayers [] (_ : _) _ _ = liftIO $ fail "Mismatch: Empty layers but non-empty caches"

runLayer :: MultiHeadAttentionComponent -> FeedForwardNetworkComponent -> LayerAttentionCache -> TokenVector -> StepCount -> TransformerResult (TokenVector, LayerAttentionCache)
runLayer mha ffn layerCaches (TokenVector inputToken) step = do
  network <- ask
  let rmsWeights = rmsAtt mha
      outputProjectionWeights = mWo mha
      headDim = headDimension network

  let
      nrmsComp = nrms mha
  (_, normalizedInput) <- tick nrmsComp () (inputToken, rmsWeights)
  
  qkvList <- forM (heads mha) $ \headComp -> do
    (q, k, v) <- runSingleHeadQKV headComp normalizedInput
    let rec = rotary headComp
    (_, (q', k')) <- tick rec () (rotary headComp, step, q, k)
    return (q', k', v)

  let (qList, _, _) = unzip3 qkvList

  -- Update per-head caches
  let headCachesList = headCaches layerCaches
  forM_ (zip headCachesList qkvList) $ \(hc, (_, k', v')) -> do
    let (HeadCache kc vc) = hc
    updateCache kc step headDim k'
    updateCache vc step headDim v'
    return ()

  -- Compute per-head attention outputs
  headOutputs <- forM (zip qList headCachesList) $ \(qHead, hc) ->
    headAttentionPerHead hc qHead step headDim

  let multiHeadOut = V.concat [vec | TokenVector vec <- headOutputs]
      attentionDelta = matrixVectorMult outputProjectionWeights multiHeadOut
      tokenAfterAttention = V.zipWith (+) inputToken attentionDelta

  -- Apply FFN (stateless)
  (_, ffnOut) <- tick ffn () tokenAfterAttention

  let finalToken = V.zipWith (+) tokenAfterAttention ffnOut

  return (TokenVector finalToken, layerCaches)  -- Caches modified in-place

embed :: EmbeddingComponent -> Token -> TransformerResult TokenVector
embed embedding (Token tokenCode) = do
  let
    vocab = vocabulary embedding
    rowStart = fromIntegral tokenCode * ncols vocab
    tokenVector = TokenVector $ V.slice rowStart (ncols vocab) (items2D vocab)
  return tokenVector

applyRotary :: RotaryEncodingComponent -> StepCount -> V.Vector Float -> TransformerResult (V.Vector Float)
applyRotary (RotaryEncodingComponent freqCos freqSin) (StepCount step) input = do
  network <- ask
  let
      cosFrequencies = getRow step freqCos
      sinFrequencies = getRow step freqSin
      headDim = headDimension network

  -- Apply rotation per head
  return $ applyRotaryPositionEncoding headDim cosFrequencies sinFrequencies input

applyRotaryPositionEncoding :: Int -> V.Vector Float -> V.Vector Float -> V.Vector Float -> V.Vector Float
applyRotaryPositionEncoding headDim cosFrequencies sinFrequencies input = let

  processedPairs = map (\pairIndex ->
    let realComponent = input V.! pairIndex
        imagComponent = input V.! (pairIndex + 1)
        cosValue = cosFrequencies V.! (pairIndex `div` 2)
        sinValue = sinFrequencies V.! (pairIndex `div` 2)
        rotatedReal = realComponent * cosValue - imagComponent * sinValue
        rotatedImag = realComponent * sinValue + imagComponent * cosValue
    in [(pairIndex, rotatedReal), (pairIndex + 1, rotatedImag)]
    ) [0, 2 .. headDim - 2]

  updates = concat processedPairs
  rotated = input V.// updates
  result = input V.// zip [0..headDim-1] (V.toList rotated)
  in result

data NetworkConfig = NetworkConfig
  { modelDim :: Int,
    hiddenDim :: Int,
    numLayers :: Int,
    numAttentionHeads :: Int,
    numKeyValueHeads :: Int,
    vocabSize :: Int,
    seqLen :: Int,
    headDimension :: Int,
    decoder :: TransformerDecoderComponent
  }

-- Per-head cache update
updateCache :: MVectorFloat -> StepCount -> Int -> V.Vector Float -> TransformerResult ()
updateCache cache (StepCount step) headDim slice = do
  let offset = step * headDim
  hsm <- liftIO $ V.thaw slice
  liftIO $ MV.copy (MV.slice offset headDim cache) hsm

-- Per-head attention computation
headAttentionPerHead :: HeadCache -> V.Vector Float -> StepCount -> Int -> TransformerResult TokenVector
headAttentionPerHead (HeadCache kc vc) qHead (StepCount stepInt) headDim = do
  kVec <- liftIO $ V.unsafeFreeze kc
  vVec <- liftIO $ V.unsafeFreeze vc
  let scaling = sqrt (fromIntegral headDim :: Float)
      sequenceLength = stepInt + 1
      rawScores = V.generate sequenceLength $ \pos ->
        let offset = pos * headDim
            kSlice = V.slice offset headDim kVec
            dotProd = V.sum (V.zipWith (*) qHead kSlice)
        in dotProd / scaling
      softValues = softmax rawScores sequenceLength
      headScores = V.toList softValues
      zeroVec = V.replicate headDim 0.0
      addScaled acc pos =
        let offset = pos * headDim
            vSlice = V.slice offset headDim vVec
            score = headScores !! pos
            scaled = V.map (* score) vSlice
        in V.zipWith (+) acc scaled
      result = foldl' addScaled zeroVec [0 .. sequenceLength - 1]
  return $ TokenVector result

-- QKV per head
runSingleHeadQKV :: SingleHeadComponent -> V.Vector Float -> TransformerResult (V.Vector Float, V.Vector Float, V.Vector Float)
runSingleHeadQKV headComp normalizedInput = do
  let q = matrixVectorMult (wqHead headComp) normalizedInput
      k = matrixVectorMult (wkHead headComp) normalizedInput
      v = matrixVectorMult (wvHead headComp) normalizedInput
  return (q, k, v)

-- classifier logits for a given token vector
transformerLogits :: TransformerDecoderComponent -> V.Vector Float -> TransformerResult (V.Vector Float)
transformerLogits dec tokenVec = do
  let
      vocab = vocabulary (modelEmbedding dec)
      rmsWeight = rmsFinalWeight (modelEmbedding dec)
      tokenWithRms = rmsNorm tokenVec rmsWeight
      logits = V.generate (nrows vocab) $ \row ->
        let start = row * ncols vocab
            end = start + ncols vocab
            dot = sum [(items2D vocab V.! i) * (tokenWithRms V.! (i - start)) | i <- [start .. end - 1]]
         in dot
  return logits

parseNetworkConfigFile :: BG.Get NetworkConfig
parseNetworkConfigFile = do
  modelDim' <- fromIntegral <$> BG.getInt32le
  hiddenDim' <- fromIntegral <$> BG.getInt32le
  nLayers' <- fromIntegral <$> BG.getInt32le
  numAttentionHeads' <- fromIntegral <$> BG.getInt32le
  numKeyValueHeads' <- fromIntegral <$> BG.getInt32le
  vocabSize' <- fromIntegral <$> BG.getInt32le
  seqLen' <- fromIntegral <$> BG.getInt32le
  tokenEmbeddingTable' <- readArray2D vocabSize' modelDim'
  rmsAttWeight' <- readArray2D nLayers' modelDim' :: BG.Get Array2D
  wq' <- readArray3D nLayers' modelDim' modelDim' :: BG.Get Array3D
  wk' <- readArray3D nLayers' modelDim' modelDim' :: BG.Get Array3D
  wv' <- readArray3D nLayers' modelDim' modelDim' :: BG.Get Array3D
  wo' <- readArray3D nLayers' modelDim' modelDim' :: BG.Get Array3D
  rmsFfnWeight' <- readArray2D nLayers' modelDim' :: BG.Get Array2D
  w1' <- readArray3D nLayers' hiddenDim' modelDim' :: BG.Get Array3D
  w2' <- readArray3D nLayers' modelDim' hiddenDim' :: BG.Get Array3D
  w3' <- readArray3D nLayers' hiddenDim' modelDim' :: BG.Get Array3D
  rmsFinalWeight' <- readVector modelDim'
  freqCisReal' <- readArray2D seqLen' ((modelDim' `div` numAttentionHeads') `div` 2)
  freqCisImag' <- readArray2D seqLen' ((modelDim' `div` numAttentionHeads') `div` 2)

  let
      headDim = modelDim' `div` numAttentionHeads'
      -- Construct the Embedding
      embedding = EmbeddingComponent
              { vocabulary = tokenEmbeddingTable' -- Embedding weights, shape [vocabSize, modelDim]
              , rmsFinalWeight = rmsFinalWeight' -- Final RMSNorm weights, shape [modelDim]
              }
      -- Construct the RotaryEncoding
      rotary = RotaryEncodingComponent
              { freqCos = freqCisReal' -- Real part of rotary positional encoding frequencies, shape [seqLen, headDim / 2]
              , freqSin = freqCisImag' -- Imaginary part of rotary positional encoding frequencies, shape [seqLen, headDim / 2]
              }
      -- Construct the list of TransformerLayers
      layers = 
        [ TransformerLayerComponent
            { multiHeadAttention = MultiHeadAttentionComponent
                { heads = [ SingleHeadComponent
                              { wqHead = getHeadArray2D layerIdx headIdx headDim wq' -- Query projection weights for a single head, shape [headDim, modelDim]
                              , wkHead = getHeadArray2D layerIdx headIdx headDim wk' -- Key projection weights for a single head, shape [headDim, modelDim]
                              , wvHead = getHeadArray2D layerIdx headIdx headDim wv' -- Value projection weights for a single head, shape [headDim, modelDim]
                              , rotary = rotary -- Rotary encoding component
                              }
                          | headIdx <- [0..numAttentionHeads' - 1]
                          ]
                , mWo     = getArray2D layerIdx wo' -- Output projection matrix, shape [modelDim, modelDim]
                , rmsAtt = getRow layerIdx rmsAttWeight' -- RMSNorm weights before QKV projection, shape [modelDim]
                , nrms   = NormRMS
                }, 
              feedforwardNetwork = FeedForwardNetworkComponent
                      { fW1 = getArray2D layerIdx w1' -- Feed-forward gate-up projection weights, shape [hiddenDim, modelDim]
                      , fW2 = getArray2D layerIdx w2' -- Feed-forward down projection weights, shape [modelDim, hiddenDim]
                      , fW3 = getArray2D layerIdx w3' -- Feed-forward gate-down projection weights, shape [hiddenDim, modelDim]
                      , fRMSFfn = getRow layerIdx rmsFfnWeight' -- RMSNorm weights for feed-forward, shape [modelDim]
                      }
              }
          | layerIdx <- [0..nLayers' - 1]
          ]
      -- Construct the TransformerArchitecture
      decoder = TransformerDecoderComponent
              { modelEmbedding = embedding -- Embedding component
              , modelLayers = layers -- List of transformer layers
              }
  return $
    NetworkConfig
      { modelDim = modelDim',
        hiddenDim = hiddenDim',
        numLayers = nLayers',
        numAttentionHeads = numAttentionHeads',
        numKeyValueHeads = numKeyValueHeads',
        vocabSize = abs vocabSize',
        seqLen = seqLen',
        headDimension = headDim,
        decoder = decoder
      }

-- | Extract the weight matrix for one head of one layer
--   layerIdx ∈ [0..sizeX-1]
--   headIdx  ∈ [0..numHeads-1]
--   headDim  = rowsPerHead
getHeadArray2D :: Int      -- ^ layer index
               -> Int      -- ^ head index
               -> Int      -- ^ headDim (rows per head)
               -> Array3D  -- ^ the big 3D tensor (layers × rows × cols)
               -> Array2D  -- ^ the resulting (headDim × ncols) matrix
getHeadArray2D layerIdx headIdx headDim (Array3D vec _ sizeY sizeZ) =
  let
    -- sizeY is the total number of rows = numHeads * headDim
    ncols    = sizeZ
    startRow = headIdx * headDim
    -- Offset to this layer’s slice:
    layerOffset = layerIdx * sizeY * sizeZ
    -- Grab each row belonging to this head:
    newItems = V.concat
      [ V.slice (layerOffset + (row * ncols)) ncols vec
      | row <- [startRow .. startRow + headDim - 1]
      ]
  in Array2D { items2D = newItems, nrows = headDim, ncols = ncols }
