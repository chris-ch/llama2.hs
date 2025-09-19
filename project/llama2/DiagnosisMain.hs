module Main where

import Prelude

import qualified Data.Binary.Get as BG
import qualified Data.ByteString.Lazy as BSL
import System.Environment (getArgs)
import Text.Printf (printf)

-- Import type-level constants from your Helpers.hs (adjust path if needed)
import Foreign (Int32)
import Control.Monad (replicateM)

-- Helper to convert Nat to Int (simplified; assumes you have natToNum from Clash.Prelude)
natToNum :: Int
natToNum = 0  -- Placeholder; replace with actual values for calculation
-- For calculation purposes, hardcode the 15M values here to avoid Clash dependency issues
modelDim :: Int
modelDim = 64
hiddenDim :: Int
hiddenDim = 172
numLayers :: Int
numLayers = 5
numQueryHeads :: Int
numQueryHeads = 8
numKeyValueHeads :: Int
numKeyValueHeads = 4
headDimension :: Int
headDimension = 8
freqDim :: Int
freqDim = 4
vocabSize :: Int
vocabSize = 512
seqLen :: Int
seqLen = 512


headerSize :: Int
headerSize = 7 * 4  -- 7 Int32, little-endian

data TensorInfo = TensorInfo
  { name :: String
  , expectedFloats :: Int
  , byteSize :: Int
  , cumulativeBytes :: Int
  } deriving (Show)

calculateTensorSizes :: [TensorInfo]
calculateTensorSizes =
  let
    embeddingFloats = vocabSize * modelDim
    rmsAttFloats = numLayers * modelDim
    qkvFloats = numLayers * numKeyValueHeads * headDimension * modelDim  -- wk, wv; wq uses numQueryHeads but same size
    woFloats = numLayers * numQueryHeads * modelDim * headDimension
    rmsFfnFloats = numLayers * modelDim
    w1w2w3Floats = numLayers * hiddenDim * modelDim  -- Each of w1, w2, w3
    rmsFinalFloats = modelDim
    freqFloats = seqLen * freqDim  -- For both real and imag
    tensors =
      [ TensorInfo "tokenEmbeddingTable" embeddingFloats (embeddingFloats * 4) 0
      , TensorInfo "rmsAttWeight" rmsAttFloats (rmsAttFloats * 4) 0
      , TensorInfo "wq" (numLayers * numQueryHeads * headDimension * modelDim) (numLayers * numQueryHeads * headDimension * modelDim * 4) 0
      , TensorInfo "wk" qkvFloats (qkvFloats * 4) 0
      , TensorInfo "wv" qkvFloats (qkvFloats * 4) 0
      , TensorInfo "wo" woFloats (woFloats * 4) 0
      , TensorInfo "rmsFfnWeight" rmsFfnFloats (rmsFfnFloats * 4) 0
      , TensorInfo "w1" w1w2w3Floats (w1w2w3Floats * 4) 0
      , TensorInfo "w2" w1w2w3Floats (w1w2w3Floats * 4) 0
      , TensorInfo "w3" w1w2w3Floats (w1w2w3Floats * 4) 0
      , TensorInfo "rmsFinalWeight" rmsFinalFloats (rmsFinalFloats * 4) 0
      , TensorInfo "freqCisReal" freqFloats (freqFloats * 4) 0
      , TensorInfo "freqCisImag" freqFloats (freqFloats * 4) 0
      ]
    cumulative = scanl (\cum t -> cum + byteSize t) headerSize (init tensors)
  in zipWith (\t cum -> t { cumulativeBytes = cum }) tensors cumulative

getHeader :: BSL.ByteString -> [Int32]
getHeader = BG.runGet g
  where
    g = replicateM 7 BG.getInt32le

diagnoseFile :: FilePath -> IO ()
diagnoseFile filePath = do
  content <- BSL.readFile filePath
  let totalBytes = BSL.length content
      header = getHeader content
      remainingBytes = totalBytes - fromIntegral headerSize
      totalFloats = remainingBytes `div` 4
      tensors = calculateTensorSizes
  putStrLn $ "File: " ++ filePath
  putStrLn $ "Total bytes: " ++ show totalBytes
  putStrLn $ "Header (7 Int32 values): " ++ show header
  putStrLn $ "Expected total floats: " ++ show (sum (map expectedFloats tensors))
  putStrLn $ "Actual floats after header: " ++ show totalFloats
  putStrLn "\nTensor breakdown:"
  mapM_ (printTensorInfo (fromIntegral totalBytes)) tensors
    where
      printTensorInfo tb t = do
        let endPos = cumulativeBytes t + byteSize t
            sufficient = endPos <= tb
        printf "%s: Expected %d floats (%d bytes), starts at %d, ends at %d -- Sufficient: %s\n"
          (name t) (expectedFloats t) (byteSize t) (cumulativeBytes t) endPos (show sufficient)

main :: IO ()
main = do
  args <- getArgs
  case args of
    [file] -> diagnoseFile file
    _ -> putStrLn "Usage: stack exec diagnose-model ./data/stories15M.bin"
