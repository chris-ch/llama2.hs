module Transformer (generateTokens) where

import Control.Monad.Reader (MonadIO (liftIO), MonadReader (ask))
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.Vector.Unboxed as V
import GHC.Unicode (isSpace)
import System.IO (hFlush, stdout)
import Types (StepCount (..), Token, TokenVector(..), PromptTokens, Vocabulary)
import Primitives (drawSample, softmax)
import Architecture (transformerLogits, TransformerResult, NetworkConfig (..), runModel)


--------------------------------------------------------------------------------
-- Transformer runtime
--------------------------------------------------------------------------------

-- Transformer step
transformer :: Token -> StepCount -> TransformerResult (V.Vector Float)
transformer tokenCode stepCount = do
  network <- ask
  let
    dec = decoder network
  TokenVector finalTokenVector <-runModel dec tokenCode stepCount
  -- Classifier into logits
  transformerLogits dec finalTokenVector

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
