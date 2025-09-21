{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}

module DebugDump (dumpLayerSums, traceFullStackPos01, tracePos01AllLayers
) where

import Prelude

import qualified Clash.Prelude          as C
import           Numeric                (showFFloat)
import qualified Data.Foldable          as F

import Helpers
  ( TransformerDecoderComponent(..)
  , TransformerLayerComponent(..)
  , MultiHeadAttentionComponent(..)
  , FeedForwardNetworkComponent(..)
  , CArray2D(..), NumLayers, Token, EmbeddingComponent (..), SeqLen, NumQueryHeads, HeadDimension, NumKeyValueHeads, VocabSize, ModelDim, dotProduct, rmsNorm, runSingleHeadQKV, applyRotaryToHead, StepCount (..), computeAttentionScores, computeAttentionWeights, computeAttentionOutput, matrixVectorMult, argMax, transformerLogits
  )
import Text.Printf (printf)
import qualified Data.List as DL
import Data.Ord (Down(..))
import qualified Data.Foldable as DF

-- Sum of absolute values, in Double for stability/printing
sumAbs :: [Float] -> Double
sumAbs = sum . fmap (realToFrac . abs)

-- Flatten a 2D Clash matrix into a Haskell list
toList2 :: CArray2D n m -> [Float]
toList2 (CArray2D mat) = concatMap C.toList (C.toList mat)

-- Dump per-layer absolute-sum checksums for all major tensors.
-- Output format mirrors the C helper so you can diff them directly.
dumpLayerSums :: TransformerDecoderComponent -> IO ()
dumpLayerSums TransformerDecoderComponent{..} = do
  putStrLn "=== Haskell per-layer checksum sums (abs) ==="
  -- Build one IO action per layer using C.imap, then run them with sequenceA_
  let actions :: C.Vec NumLayers (IO ())
      actions = C.imap dumpOne modelLayers
  F.sequenceA_ actions
  putStrLn "=== End checksums ==="
 where
  pr :: String -> Double -> IO ()
  pr tag v = putStrLn (tag ++ showFFloat (Just 6) v "")

  dumpOne :: C.Index n -> TransformerLayerComponent -> IO ()
  dumpOne lidx TransformerLayerComponent{..} = do
    let 
        MultiHeadAttentionComponent{..} = multiHeadAttention
        FeedForwardNetworkComponent{..} = feedforwardNetwork

        chk :: String -> [Float] -> IO ()
        chk name xs =
          pr ("[CHK] L=" ++ show lidx ++ " " ++ name ++ "  sum|x|=") (sumAbs xs)

    -- Attention-side
    chk "rms_att" (C.toList rmsAtt)
    -- Feed-forward-side
    chk "rms_ffn" (C.toList fRMSFfn)
    chk "W1"      (toList2 fW1)
    chk "W2"      (toList2 fW2)
    chk "W3"      (toList2 fW3)
    -- W_O is stored per head as (ModelDim x HeadDim); flatten across heads
    chk "WO"      (concatMap toList2 (C.toList mWo))

topKPairs :: Int -> C.Vec n Float -> [(Int, Float)]
topKPairs k v =
  take k $ DL.sortBy (\(_,a) (_,b) -> compare (Down a) (Down b)) $
      zip [0..] (C.toList v)

ppTopK :: String -> [(Int,Float)] -> IO ()
ppTopK tag xs = do
  putStr tag
  let one (i,v) = "(" ++ show i ++ ", " ++ printf "%.7g" v ++ ")"
  putStrLn $ " " ++ DL.intercalate ", " (map one xs)

-- Full-stack tracer: threads x through all layers, builds 2-step KV per layer,
-- then applies final RMS + tied classifier (embedding table).
traceFullStackPos01
  :: TransformerDecoderComponent
  -> Token  -- t0 (BOS=1)
  -> Token  -- t1 (next prompt token)
  -> IO ()
traceFullStackPos01 dec t0 t1 = do
  let EmbeddingComponent{vocabulary=CArray2D vocab, rmsFinalWeight=rfw} = modelEmbedding dec
      layersV = modelLayers dec
      -- logits from a pre-RMS'd vector (because we will apply final RMS explicitly here)
      logitsFromRms :: C.Vec ModelDim Float -> C.Vec VocabSize Float
      logitsFromRms xr = C.map (`dotProduct` xr) vocab

      -- Feed-forward one layer for pos=0 and pos=1, threading x and using 2-step KV
      stepLayer (x0, x1) layer =
        let mha = multiHeadAttention layer
            ffn = feedforwardNetwork layer

            -- pre-norms
            xHat0 = rmsNorm x0 (rmsAtt mha)
            xHat1 = rmsNorm x1 (rmsAtt mha)

            -- per-head QKV for both positions, with RoPE applied (q,k)
            qkv h xh =
              let (q0,k0,v0) = runSingleHeadQKV (heads mha C.!! h) xh
                  (qr0,kr0)  = applyRotaryToHead (heads mha C.!! h) (StepCount 0) (q0,k0)
              in (qr0,kr0,v0)

            qkv1 h =
              let (q1,k1,v1) = runSingleHeadQKV (heads mha C.!! h) xHat1
                  (qr1,kr1)  = applyRotaryToHead (heads mha C.!! h) (StepCount 1) (q1,k1)
              in (qr1,kr1,v1)

            -- Build length-2 sequences (we lift them into SeqLen by filling zeros elsewhere)
            zerosSeq :: C.Vec SeqLen (C.Vec HeadDimension Float)
            zerosSeq = C.repeat (C.repeat 0)

            headOuts0 :: C.Vec NumQueryHeads (C.Vec HeadDimension Float)
            headOuts1 :: C.Vec NumQueryHeads (C.Vec HeadDimension Float)
            (headOuts0, headOuts1) =
              C.unzip $
                C.imap
                  (\h _ ->
                    let (q0r,k0r,v0) = qkv h xHat0
                        (q1r,k1r,v1) = qkv1 h
                        -- map this Q head to its KV bank
                        nQ = C.snatToNum (C.SNat @NumQueryHeads) :: Int
                        nK = C.snatToNum (C.SNat @NumKeyValueHeads) :: Int
                        kvMul = max 1 (nQ `div` nK)
                        kvIdx :: C.Index NumKeyValueHeads
                        kvIdx = fromInteger . toInteger $ (fromIntegral h `div` kvMul)
                        -- two-step K/V for this KV bank
                        kSeq = C.replace d1' k1r (C.replace d0' k0r zerosSeq)
                        vSeq = C.replace d1' v1  (C.replace d0' v0  zerosSeq)
                        -- attention at pos0 (uses only t<=0)
                        scores0  = computeAttentionScores q0r kSeq
                        weights0 = computeAttentionWeights
                                     (C.imap (\t s -> if fromEnum t <= 0 then s else s - 1.0e9) scores0)
                        out0     = computeAttentionOutput weights0 vSeq
                        -- attention at pos1 (uses t<=1)
                        scores1  = computeAttentionScores q1r kSeq
                        weights1 = computeAttentionWeights
                                     (C.imap (\t s -> if fromEnum t <= 1 then s else s - 1.0e9) scores1)
                        out1     = computeAttentionOutput weights1 vSeq
                    in (out0, out1))
                  (C.indicesI @NumQueryHeads)

            -- Project each head with W_O slice and sum
            yAttn0 = foldl1 (C.zipWith (+))
                   $ C.zipWith matrixVectorMult (mWo mha) headOuts0
            yAttn1 = foldl1 (C.zipWith (+))
                   $ C.zipWith matrixVectorMult (mWo mha) headOuts1

            xAtt0 = C.zipWith (+) x0 yAttn0
            xAtt1 = C.zipWith (+) x1 yAttn1

            -- FFN + residual
            xHatF0  = rmsNorm xAtt0 (fRMSFfn ffn)
            gate0   = matrixVectorMult (fW1 ffn) xHatF0
            up0     = matrixVectorMult (fW3 ffn) xHatF0
            core0   = C.zipWith (*) (C.map (\z -> z / (1 + exp (-z))) gate0) up0
            x0'     = C.zipWith (+) xAtt0 (matrixVectorMult (fW2 ffn) core0)

            xHatF1  = rmsNorm xAtt1 (fRMSFfn ffn)
            gate1   = matrixVectorMult (fW1 ffn) xHatF1
            up1     = matrixVectorMult (fW3 ffn) xHatF1
            core1   = C.zipWith (*) (C.map (\z -> z / (1 + exp (-z))) gate1) up1
            x1'     = C.zipWith (+) xAtt1 (matrixVectorMult (fW2 ffn) core1)
        in (x0', x1')

      x0init = vocab C.!! t0
      x1init = vocab C.!! t1

      (x0final, x1final) = DL.foldl' stepLayer (x0init, x1init) (C.toList layersV)

      -- final RMS, then logits via tied embedding
      x0r = rmsNorm x0final rfw
      x1r = rmsNorm x1final rfw
      logits0 = logitsFromRms x0r
      logits1 = logitsFromRms x1r

      am0 = fromIntegral (argMax logits0) :: Int
      am1 = fromIntegral (argMax logits1) :: Int
      am0v = logits0 C.!! (toEnum am0 :: C.Index VocabSize)
      am1v = logits1 C.!! (toEnum am1 :: C.Index VocabSize)

  putStrLn "[FULL] P0 x_final_rms[0:8]:"
  print (take 8 (C.toList x0r))
  ppTopK "[FULL] P0 top5:" (topKPairs 5 logits0)
  putStrLn $ "[FULL] P0 argmax: (" ++ show am0 ++ ", " ++ printf "%.7g" am0v ++ ")"

  putStrLn "[FULL] P1 x_final_rms[0:8]:"
  print (take 8 (C.toList x1r))
  ppTopK "[FULL] P1 top5:" (topKPairs 5 logits1)
  putStrLn $ "[FULL] P1 argmax: (" ++ show am1 ++ ", " ++ printf "%.7g" am1v ++ ")"

  where
    d0' = 0 :: C.Index SeqLen
    d1' = 1 :: C.Index SeqLen

-- Full-model trace at pos 0, and pos 1 if a second token is provided
-- Pure L0 and pos=1 tracer for any layer; no transpose, no lists inside
-- Minimal dumper: only converts to list at the very end
dump8 :: String -> C.Vec n Float -> IO ()
dump8 tag v = putStrLn $ tag ++ " " ++ show (take 8 (C.toList v))

tracePos01AllLayers
  :: TransformerDecoderComponent
  -> Token  -- t0 (BOS=1)
  -> Token  -- t1 (next prompt token)
  -> IO ()
tracePos01AllLayers dec t0 t1 = do
  let EmbeddingComponent{vocabulary=CArray2D vocab} = modelEmbedding dec
      layersV = modelLayers dec
      x0  = vocab C.!! t0
      x1  = vocab C.!! t1
      silu z = z / (1 + exp (-z))

      -- explicit zeros with full type to help inference
      zerosSeq :: C.Vec SeqLen (C.Vec HeadDimension Float)
      zerosSeq = C.repeat (C.repeat 0)

  -- Iterate layers using Traversable; zip indices with layers
  DF.for_ (C.zip (C.indicesI @NumLayers) layersV) $ \(lIdx, layer) -> do
    let l = fromEnum lIdx
        mha = multiHeadAttention layer
        ffn = feedforwardNetwork layer

    putStrLn $ "[L" ++ show l ++ " P0] ----"
    -- P0 attention pre-norm
    let xHat0  = rmsNorm x0 (rmsAtt mha)
        -- QKV per head @ P0
        qkv0 h = runSingleHeadQKV (heads mha C.!! h) xHat0
        v0  h = let (_,_,v) = qkv0 h in v
        vHeads0 = C.map v0 C.indicesI
        -- sum_h (W_O(h) @ v_h)
        yAttn0  = C.foldl1 (C.zipWith (+))
                $ C.zipWith matrixVectorMult (mWo mha) vHeads0
        xAfterAttn0 = C.zipWith (+) x0 yAttn0

        xHatFfn0  = rmsNorm xAfterAttn0 (fRMSFfn ffn)
        gate0     = matrixVectorMult (fW1 ffn) xHatFfn0
        up0       = matrixVectorMult (fW3 ffn) xHatFfn0
        core0     = C.zipWith (*) (C.map silu gate0) up0
        ffnOut0   = matrixVectorMult (fW2 ffn) core0
        xAfterFfn0 = C.zipWith (+) xAfterAttn0 ffnOut0

    dump8 "[L P0] (a) xHat:" xHat0
    dump8 "[L P0] (c) WO@heads:" yAttn0
    dump8 "[L P0] (d) x_after_attn:" xAfterAttn0
    dump8 "[L P0] (e) xHat_ffn:" xHatFfn0
    dump8 "[L P0] (h) ffn_core:" core0
    dump8 "[L P0] (i) x_after_ffn:" xAfterFfn0

    putStrLn $ "[L" ++ show l ++ " P1] ----"
    -- P1 with two time steps in KV
    let xHat1 = rmsNorm x1 (rmsAtt mha)

        kSeq h =
          let (_,k0,_) = qkv h xHat0
              (_,k1,_) = qkv h xHat1
              (_,k0r)  = applyRotaryToHead (heads mha C.!! h) (StepCount 0) (C.repeat 0, k0)
              (_,k1r)  = applyRotaryToHead (heads mha C.!! h) (StepCount 1) (C.repeat 0, k1)
          in C.replace d1 k1r (C.replace d0 k0r zerosSeq)

        vSeq h =
          let (_,_,v0') = qkv h xHat0
              (_,_,v1) = qkv h xHat1
          in C.replace d1 v1 (C.replace d0 v0' zerosSeq)

        q1 h =
          let (q,_,_) = qkv h xHat1
              (qr,_)  = applyRotaryToHead (heads mha C.!! h) (StepCount 1) (q, C.repeat 0)
          in qr

        scores h  = computeAttentionScores (q1 h) (kSeq h)   -- Vec SeqLen Float
        weights h = computeAttentionWeights                   -- causal mask t>1
                   (C.imap (\t s -> if fromEnum t <= 1 then s else s - 1.0e9) (scores h))
        headOut h = computeAttentionOutput (weights h) (vSeq h)

        yAttn1 = C.foldl1 (C.zipWith (+))
              $ C.zipWith matrixVectorMult (mWo mha) (C.map headOut C.indicesI)
        xAfterAttn1 = C.zipWith (+) x1 yAttn1

        xHatFfn1  = rmsNorm xAfterAttn1 (fRMSFfn ffn)
        gate1     = matrixVectorMult (fW1 ffn) xHatFfn1
        up1       = matrixVectorMult (fW3 ffn) xHatFfn1
        core1     = C.zipWith (*) (C.map silu gate1) up1
        ffnOut1   = matrixVectorMult (fW2 ffn) core1
        xAfterFfn1 = C.zipWith (+) xAfterAttn1 ffnOut1
        qkv h = runSingleHeadQKV (heads mha C.!! h)

    dump8 "[L P1] (a) xHat:" xHat1
    dump8 "[L P1] (c) WO@heads:" yAttn1
    dump8 "[L P1] (d) x_after_attn:" xAfterAttn1
    dump8 "[L P1] (e) xHat_ffn:" xHatFfn1
    dump8 "[L P1] (h) ffn_core:" core1
    dump8 "[L P1] (i) x_after_ffn:" xAfterFfn1
    -- at end of P0 block
    let logits0 = transformerLogits dec xAfterFfn0
        top50   = topKPairs 5 logits0
        am0     = argMax logits0
        am0i    = fromIntegral am0 :: Int
        am0v    = logits0 C.!! (toEnum am0i :: C.Index VocabSize)
    ppTopK "[P0] top5:" top50
    putStrLn $ "[P0] argmax: (" ++ show am0i ++ ", " ++ printf "%.7g" am0v ++ ")"

    -- at end of P1 block
    let logits1 = transformerLogits dec xAfterFfn1
        top51   = topKPairs 5 logits1
        am1     = argMax logits1
        am1i    = fromIntegral am1 :: Int
        am1v    = logits1 C.!! (toEnum am1i :: C.Index VocabSize)
    ppTopK "[P1] top5:" top51
    putStrLn $ "[P1] argmax: (" ++ show am1i ++ ", " ++ printf "%.7g" am1v ++ ")"
  where
    d0 = 0 :: C.Index SeqLen
    d1 = 1 :: C.Index SeqLen
