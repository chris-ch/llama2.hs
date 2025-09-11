module Transformer (generateTokens) where

import Control.Monad (foldM, forM_)
import Control.Monad.Reader (MonadIO (liftIO), MonadReader (ask))
import Control.Monad.State (gets)
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.Vector.Unboxed as V
import GHC.Unicode (isSpace)
import System.IO (hFlush, stdout)
import qualified Data.Vector.Unboxed.Mutable as MV
import Types (AttentionKV (..), StepCount (..), LayerIndex(..), Token, TokenVector(..), PromptTokens, Vocabulary, getArray2D, HeadIndex(..), MVectorFloat)
import Primitives (drawSample, softmax, matrixVectorMult)
import Architecture (transformerLogits, computeQKV, TransformerResult, NetworkConfig (..), TransformerParams(..), TransformerDecoder (..), embed, TransformerLayer (..), runFeedForward, runAttention, cacheIndex)


--------------------------------------------------------------------------------
-- Transformer runtime
--------------------------------------------------------------------------------

-- Layer Token

createLayerToken :: StepCount -> LayerIndex -> TokenVector -> TransformerResult TokenVector
createLayerToken currentStep layerIndex inputToken = do
  network <- ask
  AttentionKV {keyCache, valueCache} <- gets id
  let model = params network
      LayerIndex layerIdx = layerIndex
      outputProjectionWeights = getArray2D layerIdx (wo model)
      dec = decoder network
      LayerIndex li = layerIndex
      layer = modelLayers dec !! li
      mha = multiHeadAttention layer
      ffn = feedforwardNetwork layer
      numHeads = numAttentionHeads network
      headDim = headDimension network
  
  (queryOutFrozen, keyOutputFrozen, valueOutputFrozen) <- computeQKV model currentStep layerIndex inputToken

  forM_ [0 .. numHeads - 1] $ \headIndex -> do
    -- Update cache with slices from keyOutput and valueOutput
    let keyHeadSlice = V.slice (headIndex * headDim) headDim keyOutputFrozen
        valueHeadSlice = V.slice (headIndex * headDim) headDim valueOutputFrozen
    updateCacheWithHead layerIndex (HeadIndex headIndex) currentStep keyHeadSlice keyCache
    updateCacheWithHead layerIndex (HeadIndex headIndex) currentStep valueHeadSlice valueCache

  -- Compute multi-head attention and copy it into buffer
  multiHeadOut <- runAttention mha layerIndex queryOutFrozen currentStep
  let attentionDelta = matrixVectorMult outputProjectionWeights multiHeadOut
  
  let TokenVector tokenVector = inputToken
      tokenAfterAttention = V.zipWith (+) tokenVector attentionDelta

  -- Apply FFN
  ffnOut <- runFeedForward ffn tokenAfterAttention
  return $ TokenVector $ V.zipWith (+) tokenAfterAttention ffnOut
  
-- Multi-head
updateCacheWithHead :: LayerIndex -> HeadIndex -> StepCount -> V.Vector Float -> MVectorFloat -> TransformerResult ()
updateCacheWithHead layerIndex headIndex stepIndex headSlice cache = do
  network <- ask
  let headDim = headDimension network
      cacheOffset = cacheIndex network stepIndex layerIndex headIndex 0
  hsm <- V.thaw headSlice
  MV.copy (MV.slice cacheOffset headDim cache) hsm

-- Transformer step
transformer :: Token -> StepCount -> TransformerResult (V.Vector Float)
transformer tokenCode stepCount = do
  network <- ask

  let
    dec = decoder network
    embeddingLayer = modelEmbedding dec
  tokenVector <- embed embeddingLayer tokenCode

  -- Forwarding all the layers
  TokenVector finalTokenVector <-
    foldM
      (\accToken layerIndex -> createLayerToken stepCount (LayerIndex layerIndex) accToken)
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
