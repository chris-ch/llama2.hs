module Model.Embedding.PRNG (
    sampledTokenSignal
) where

import Clash.Prelude
import Data.Maybe (fromMaybe)
import Model.Core.Types (Temperature, IntermediateData (..), Token, VocabSize, CArray2D (..), ModelDim, EmbeddingComponent (..))
import qualified Model.Layers.TransformerLayer as TransformerLayer (TransformerDecoderComponent(..))
import qualified Clash.Sized.Vector as CV
import Helpers (rmsNorm, dotProduct)

-- xorshift32 core (synthesizable, 1-cycle combinational)
xorshift32 :: Unsigned 32 -> Unsigned 32
xorshift32 s0 =
  let s1 = s0 `xor` shiftL s0 13
      s2 = s1 `xor` shiftR s1 17
      s3 = s2 `xor` shiftL s2 5
  in s3

-- classifier logits for a given token vector
transformerLogits :: TransformerLayer.TransformerDecoderComponent -> Vec ModelDim Float -> Vec VocabSize Float
transformerLogits decoder tokenVector = logits where
    vocab = vocabulary (TransformerLayer.modelEmbedding decoder)
    rmsWeight = rmsFinalWeight (TransformerLayer.modelEmbedding decoder)
    tokenWithRms = rmsNorm tokenVector rmsWeight
    CArray2D vocabRows = vocab
    logits = map (dotProduct tokenWithRms) vocabRows

-- logits from the current data
logitsSignal :: TransformerLayer.TransformerDecoderComponent -> Signal dom IntermediateData -> Signal dom (Vec VocabSize Float)
logitsSignal decoder nextIntermediateDataSignal = transformerLogits decoder . feedForwardOutput <$> nextIntermediateDataSignal

-- Sampling on pulse
sampledTokenSignal :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool 
  -> Signal dom Temperature 
  -> Signal dom Token
  -> TransformerLayer.TransformerDecoderComponent 
  -> Signal dom IntermediateData 
  -> Signal dom Token
sampledTokenSignal readyPulseSignal temperatureSignal seedSignal decoder nextIntermediateDataSignal =
  liftA3
    (\temperature logits randomVal ->
        if temperature <= 0.0 then argMax logits
        else let probabilities = softmax temperature logits
             in sampleFromProbs randomVal probabilities)
    temperatureSignal (logitsSignal decoder nextIntermediateDataSignal) (uniformRandom01Signal readyPulseSignal seedSignal)

-- PRNG state
firstPulseSignal :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool -> Signal dom Bool
firstPulseSignal readyPulseSignal = regEn True readyPulseSignal (pure False)

mixedSeedSignal :: Signal dom (Unsigned 32) -> Signal dom (Unsigned 32)
mixedSeedSignal seedSignal = (`xor` 0x9E3779B9) <$> seedSignal

prngStateSignal :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool -> Signal dom (Unsigned 32) ->Signal dom (Unsigned 32)
prngStateSignal readyPulseSignal seedSignal =
  let nextVal = mux (firstPulseSignal readyPulseSignal) (xorshift32 <$> mixedSeedSignal seedSignal)
                                      (xorshift32 <$> prngStateSignal readyPulseSignal seedSignal)
  in regEn 2463534242 readyPulseSignal nextVal

uniformRandom01Signal :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool -> Signal dom (Unsigned 32) -> Signal dom Float
uniformRandom01Signal readyPulseSignal seedSignal = (/ 16777216.0) . fromIntegral . (`shiftR` 8) <$> prngStateSignal readyPulseSignal seedSignal

-- Pure, synthesizable categorical sampling from probabilities summing to ~1.0
sampleFromProbs :: forall n. (KnownNat (n + 1), KnownNat n) => Float -> Vec (n + 1) Float -> Unsigned 32
sampleFromProbs u probs =
  let cdf = CV.scanl1 (+) probs
      idx = fromMaybe maxBound (findIndex (>= u) cdf)
  in fromIntegral (fromEnum idx)

softmax :: forall n. KnownNat (n + 1) => Float -> Vec (n + 1) Float -> Vec (n + 1) Float
softmax t xs =
  let m    = maximum xs
      exps = map (\x -> exp ((x - m) / t)) xs
      s    = sum exps
  in map (/ s) exps

-- | Find the index of the maximum element in a non-empty vector
argMax :: forall n. ( KnownNat (n + 1)) =>Vec (n+1) Float -> Unsigned 32
argMax vec = fst $ foldl compareMax (0, head vec) (imap (\i x -> (fromIntegral i, x)) vec)
  where
    compareMax :: (Unsigned 32, Float) -> (Unsigned 32, Float) -> (Unsigned 32, Float)
    compareMax (maxIdx, maxVal) (i, x)
      | x > maxVal = (i, x)
      | otherwise  = (maxIdx, maxVal)
