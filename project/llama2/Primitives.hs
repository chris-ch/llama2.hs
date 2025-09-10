module Primitives
  ( 
    applyMatrixVectorMult,
    rmsNorm,
    silu,
    dotProductMV,
    softmax,
    drawSample,
    applyRotaryPositionEncoding
  ) where

import Control.Monad (foldM, forM_)
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Unboxed.Mutable as MV
import qualified System.Random as R
import Types
  ( 
    Token,
    MVectorFloat,
    Array2D (..),
    HeadIndex (..)
  )

-- Linear algebra helpers
applyMatrixVectorMult :: Array2D -> MVectorFloat -> MVectorFloat -> IO ()
applyMatrixVectorMult array2D vecM result = do
  let nrows' = nrows array2D
      ncols' = ncols array2D
      items = items2D array2D
  forM_ [0 .. nrows' - 1] $ \row -> do
    let rowStart = row * ncols'
        loop !i !acc
          | i >= ncols' = return acc
          | otherwise = do
            v <- MV.unsafeRead vecM i
            let m = V.unsafeIndex items (rowStart + i)
            loop (i + 1) (acc + v * m)
    s <- loop 0 0.0
    MV.unsafeWrite result row s

-- dot product on mutable vectors
dotProductMV :: MVectorFloat -> MVectorFloat -> IO Float
dotProductMV vec1 vec2 = do
  let len = min (MV.length vec1) (MV.length vec2)
  foldM
    ( \acc i -> do
        a <- MV.read vec1 i
        b <- MV.read vec2 i
        return $ acc + a * b
    )
    0.0
    [0 .. len - 1]

-- RMS Norm
rmsNorm :: V.Vector Float -> V.Vector Float -> V.Vector Float
rmsNorm vector weights =
  let squareNorm = V.foldl' (\acc v -> acc + v * v) 0.0 vector
      ss = (squareNorm / fromIntegral (V.length vector)) + 1e-5
      normalized = V.map (* (1.0 / sqrt ss)) vector
   in V.zipWith (*) weights normalized

-- Activation
silu :: Float -> Float
silu x = x / (1.0 + exp (-x))

-- Softmax (applies to prefix)
softmax :: V.Vector Float -> Int -> V.Vector Float
softmax values size = V.concat [softmaxValues, V.slice size (V.length values - size) values]
  where
    maxVal = V.maximum (V.take size values)
    expValues = V.map (\x -> exp (x - maxVal)) (V.take size values)
    sumExpValues = V.sum expValues
    softmaxValues = V.map (/ sumExpValues) expValues

-- Sampling
drawSample :: Int -> V.Vector Float -> IO Token
drawSample randomSeed probabilities = do
  let gen = R.mkStdGen randomSeed
      (randomValue, _) = R.random gen :: (Float, R.StdGen)
      cumulativeDistribution = V.scanl1 (+) probabilities
      selectedIndex = V.length (V.takeWhile (< randomValue) cumulativeDistribution)
  return $ fromIntegral (min selectedIndex (V.length probabilities - 1))

applyRotaryPositionEncoding :: Int -> HeadIndex -> V.Vector Float -> V.Vector Float -> V.Vector Float -> V.Vector Float
applyRotaryPositionEncoding headDim (HeadIndex headIndex) cosFrequencies sinFrequencies input = let
  baseIndex = headIndex * headDim
  slice = V.slice baseIndex headDim input

  -- Process pairs just like the primed version
  processedPairs = map (\pairIndex ->
    let realComponent = slice V.! pairIndex
        imagComponent = slice V.! (pairIndex + 1)
        cosValue = cosFrequencies V.! (pairIndex `div` 2)
        sinValue = sinFrequencies V.! (pairIndex `div` 2)
        rotatedReal = realComponent * cosValue - imagComponent * sinValue
        rotatedImag = realComponent * sinValue + imagComponent * cosValue
    in [(pairIndex, rotatedReal), (pairIndex + 1, rotatedImag)]
    ) [0, 2 .. headDim - 2]

  updates = concat processedPairs
  rotated = slice V.// updates
  result = input V.// zip [baseIndex..baseIndex+headDim-1] (V.toList rotated)
  in result
