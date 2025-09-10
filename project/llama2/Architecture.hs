module Architecture
  ( Embedding (..),
    RotaryEncoding (..),
    MultiHeadAttention (..),
    FeedForward (..),
    TransformerLayer (..),
    TransformerArchitecture (..),
    Embeddable (..),
    Positional (..),
    FeedForwardNet (..),
    LayerComp (..),
    Model (..),
    runTransformerModel
  ) where

import Control.Monad (foldM)
import Control.Monad.Reader ( MonadIO(liftIO), MonadReader(ask) )
import Control.Monad.State ( gets )
import qualified Data.Vector.Unboxed as V
import Types
    ( getArray2D,
      getRow,
      Array2D(items2D, ncols),
      NetworkConfig(model),
      TransformerModel(wo, freqCisReal, freqCisImag),
      AttentionKV(AttentionKV, feedforwardNetworkOutput, multiHeadOutput,
                  projectedAttentionOutput, keyCache, valueCache),
      Token,
      TokenVector(..),
      StepCount(..),
      LayerIndex(..) )
import Primitives
    ( TransformerResult,
      applyMatrixVectorMult,
      computeMultiHeadAttention,
      computeQKV,
      applyFeedForwardNetwork,
      transformerLogits )

-- Data definitions mirroring architecture boxes
newtype Embedding = Embedding
  { embMatrix :: Array2D
  }

data RotaryEncoding = RotaryEncoding
  { freqCos :: Array2D,
    freqSin :: Array2D
  }

data MultiHeadAttention = MultiHeadAttention
  { m_wq :: Array2D,
    m_wk :: Array2D,
    m_wv :: Array2D,
    m_wo :: Array2D,
    m_rmsAtt :: Array2D -- kept as placeholder; actual rms is per-row
  }

data FeedForward = FeedForward
  { f_w1 :: Array2D,
    f_w2 :: Array2D,
    f_w3 :: Array2D,
    f_rmsFfn :: Array2D
  }

data TransformerLayer = TransformerLayer
  { layerIndex :: LayerIndex,
    layerMha :: MultiHeadAttention,
    layerFfn :: FeedForward
  }

data TransformerArchitecture = TransformerArchitecture
  { modelEmbedding :: Embedding,
    modelRotary :: RotaryEncoding,
    modelLayers :: [TransformerLayer]
  }

-- Behavioural interfaces (typeclasses)
class Embeddable e where
  embedToken :: e -> Token -> TransformerResult TokenVector

class Positional p where
  applyPosition :: p -> StepCount -> TokenVector -> TransformerResult TokenVector

class FeedForwardNet f where
  runFFNComp :: f -> LayerIndex -> TokenVector -> TransformerResult TokenVector

class LayerComp l where
  runLayerComp :: l -> StepCount -> TokenVector -> TransformerResult TokenVector

class Model m where
  runTransformer :: m -> StepCount -> Token -> TransformerResult (V.Vector Float)

-- Instances wiring to primitives
instance Embeddable Embedding where
  embedToken :: Embedding -> Token -> TransformerResult TokenVector
  embedToken Embedding {embMatrix} token = do
    let vocab = embMatrix
        rowStart = fromIntegral token * ncols vocab
        tok = TokenVector $ V.slice rowStart (ncols vocab) (items2D vocab)
    return tok

instance Positional RotaryEncoding where
  applyPosition :: RotaryEncoding -> StepCount -> TokenVector -> TransformerResult TokenVector
  applyPosition _ (StepCount _) = return

instance FeedForwardNet FeedForward where
  runFFNComp :: FeedForward -> LayerIndex -> TokenVector -> TransformerResult TokenVector
  runFFNComp FeedForward {} layerIdx (TokenVector inputVec) = do
    -- wrap primitive applyFeedForwardNetwork
    network <- ask
    let weights = model network
    applyFeedForwardNetwork weights layerIdx inputVec
    st <- gets id
    outVec <- liftIO $ V.freeze (feedforwardNetworkOutput st)
    return $ TokenVector outVec

instance LayerComp TransformerLayer where
  runLayerComp :: TransformerLayer -> StepCount -> TokenVector -> TransformerResult TokenVector
  runLayerComp TransformerLayer {layerIndex} stepCount tokenVec = do
    network <- ask
    let weights = model network
        li = layerIndex
        LayerIndex li' = li
        freqCosValues = getRow (let StepCount s = stepCount in s) (freqCisReal weights)
        freqSinValues = getRow (let StepCount s = stepCount in s) (freqCisImag weights)

    -- QKV + cache update
    computeQKV weights stepCount li freqCosValues freqSinValues tokenVec

    -- Run multihead attention
    AttentionKV {multiHeadOutput, projectedAttentionOutput, keyCache = kc, valueCache = vc} <- gets id

    computeMultiHeadAttention stepCount li multiHeadOutput kc vc multiHeadOutput

    -- Output projection W_o
    liftIO $ applyMatrixVectorMult (getArray2D li' (wo weights)) multiHeadOutput projectedAttentionOutput

    attentionDelta <- liftIO $ V.freeze projectedAttentionOutput
    let TokenVector token = tokenVec
        tokenAfterAttention = V.zipWith (+) token attentionDelta

    -- FFN
    applyFeedForwardNetwork weights li tokenAfterAttention
    st <- gets id
    ffnOut <- liftIO $ V.freeze (feedforwardNetworkOutput st)
    return $ TokenVector $ V.zipWith (+) tokenAfterAttention ffnOut

-- Top-level model runner
instance Model TransformerArchitecture where
  runTransformer :: TransformerArchitecture -> StepCount -> Token -> TransformerResult (V.Vector Float)
  runTransformer TransformerArchitecture {modelEmbedding, modelLayers} stepCount tokenCode = do
    TokenVector vec <- embedToken modelEmbedding tokenCode
    finalToken <- foldM (\acc layer -> runLayerComp layer stepCount acc) (TokenVector vec) modelLayers
    let TokenVector finalVec = finalToken
    transformerLogits finalVec

runTransformerModel :: TransformerArchitecture -> StepCount -> Token -> TransformerResult (V.Vector Float)
runTransformerModel = runTransformer
