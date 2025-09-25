module Helpers (
  dotProduct
  , liftA4
  , liftA5
  , rmsNorm
  , matrixVectorMult
) where

import Clash.Prelude

import Model.Core.Types

-- Dot product of two Vecs
dotProduct :: KnownNat n => Vec n Float -> Vec n Float -> Float
dotProduct v1 v2 = sum $ zipWith (*) v1 v2

-- Vector multiplication by a Matrix
matrixVectorMult :: forall n m. KnownNat m => CArray2D n m -> Vec m Float -> Vec n Float
matrixVectorMult (CArray2D mat) vec = map (dotProduct vec) mat

-- RMS Norm
rmsNorm :: Vec ModelDim Float -> Vec ModelDim Float -> Vec ModelDim Float
rmsNorm vec weights =
  let squareNorm = sum (map (\v -> v * v) vec)
      ss = (squareNorm / fromIntegral (length vec)) + 1e-5
      normalized = map (* (1.0 / sqrt ss)) vec
  in zipWith (*) weights normalized

liftA4 :: Applicative f => (a -> b -> c -> d -> e) -> f a -> f b -> f c -> f d -> f e
liftA4 f a b c d = liftA3 f a b c <*> d

liftA5 :: Applicative g => (a -> b -> c -> d -> e -> f) -> g a -> g b -> g c -> g d -> g e -> g f
liftA5 f fa fb fc fd fe = f <$> fa <*> fb <*> fc <*> fd <*> fe
