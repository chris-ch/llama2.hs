module Transformer (generateTokens) where

import Control.Monad (foldM)
import Control.Monad.Reader (MonadIO (liftIO), MonadReader (ask))
import Control.Monad.State (gets)
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.Vector.Unboxed as V
import GHC.Unicode (isSpace)
import System.IO (hFlush, stdout)
import Types (NetworkConfig (..), AttentionKV (..), StepCount (..), LayerIndex(..), Token, TransformerModel (..), TokenVector(..), PromptTokens, Vocabulary, getArray2D, getRow)
import Primitives (transformerLogits, getTokenEmbedding, applyFeedForwardNetwork, computeQKV, computeMultiHeadAttention, applyMatrixVectorMult, drawSample, softmax, TransformerResult)

--------------------------------------------------------------------------------
-- Transformer runtime
--------------------------------------------------------------------------------

-- Layer Token
createLayerToken :: StepCount -> LayerIndex -> V.Vector Float -> V.Vector Float -> TokenVector -> TransformerResult TokenVector
createLayerToken currentStep layerIndex freqCosValues freqSinValues inputToken = do
  network <- ask
  AttentionKV {queryOutput, keyCache, valueCache, projectedAttentionOutput, feedforwardNetworkOutput, multiHeadOutput} <- gets id
  let weights = model network
      LayerIndex layerIdx = layerIndex
      outputProjectionWeights = getArray2D layerIdx (wo weights)

  computeQKV weights currentStep layerIndex freqCosValues freqSinValues inputToken

  -- Compute multi-head attention directly into concatenatedHeads buffer
  computeMultiHeadAttention currentStep layerIndex queryOutput keyCache valueCache multiHeadOutput

  -- Apply output projection in-place
  liftIO $ applyMatrixVectorMult outputProjectionWeights multiHeadOutput projectedAttentionOutput

  -- Convert mutable output vector back to immutable
  attentionDelta <- liftIO $ V.freeze projectedAttentionOutput
  let TokenVector tokenVector = inputToken
      tokenAfterAttention = V.zipWith (+) tokenVector attentionDelta

  -- Apply FFN
  applyFeedForwardNetwork weights layerIndex tokenAfterAttention
  ffnOut <- liftIO $ V.freeze feedforwardNetworkOutput
  return $ TokenVector $ V.zipWith (+) tokenAfterAttention ffnOut

-- Transformer step
transformer :: Token -> StepCount -> TransformerResult (V.Vector Float)
transformer tokenCode stepCount = do
  network <- ask

  -- Getting the token embedding
  let weights = model network
  
  tokenVector <- getTokenEmbedding tokenCode

  -- Plucking out the current row of freq_cis_real and freq_cis_imag
  let StepCount step = stepCount
      freqCosValues = getRow step (freqCisReal weights)
      freqSinValues = getRow step (freqCisImag weights)

  -- Forwarding all the layers
  TokenVector finalTokenVector <-
    foldM
      (\accToken layerIndex -> createLayerToken stepCount (LayerIndex layerIndex) freqCosValues freqSinValues accToken)
      tokenVector
      [0 .. numLayers network - 1]

  -- Classifier into logits
  transformerLogits finalTokenVector

generateNextToken :: StepCount -> PromptTokens -> Float -> Vocabulary -> Token -> Int -> TransformerResult (BS.ByteString, Token)
generateNextToken timestep promptTokens temperature vocab tokenCode seedValue = do
  network <- ask
  logits <- transformer tokenCode timestep
  let StepCount step = timestep
  nextToken <-
    if step < length promptTokens
      then return (promptTokens !! step)
      else
        if temperature == 0.0
          then return $ fromIntegral (V.maxIndex logits)
          else do
            liftIO $ drawSample seedValue $ softmax (V.map (/ temperature) logits) (vocabSize network)
  let word = vocab !! fromIntegral nextToken :: BS.ByteString
      firstChar = BSC.head word :: Char
      tokenStr =
        if tokenCode == 1 && isSpace firstChar
          then BSC.tail (vocab !! fromIntegral nextToken)
          else vocab !! fromIntegral nextToken
  return (tokenStr, nextToken)

generateTokens :: StepCount -> PromptTokens -> Float -> Vocabulary -> Int -> TransformerResult ([BS.ByteString], StepCount)
generateTokens maxSteps promptTokens temperature vocab seedValue = do
  network <- ask
  go network (StepCount 0) [] 1
  where
    go network timestep result token
      | timestep >= maxSteps || (timestep /= StepCount 0 && token == 1) = return (result, timestep)
      | otherwise = do
          (tokenStr, nextToken) <- generateNextToken timestep promptTokens temperature vocab token seedValue
          liftIO $ putStr $ BSC.unpack tokenStr
          liftIO $ hFlush stdout
          go network (timestep + StepCount 1) (result ++ [tokenStr]) nextToken
