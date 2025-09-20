{-# LANGUAGE OverloadedStrings #-}
module Tokenizer
  ( Tokenizer
  , buildTokenizer
  , encodeTokens   -- encodeTokens tok text bos eos
  , decodePiece    -- decodePiece tok prev token
  , vocabVector    -- to expose vocab if you need it
  ) where

import Prelude
import qualified Data.Binary.Get          as BG
import qualified Data.ByteString.Lazy     as BSL
import qualified Data.Map.Strict          as M
import qualified Data.Vector              as V
import           Data.Bits                ((.&.))
import           Data.Maybe               (fromMaybe)
import           Data.Word                (Word8)
import Control.Monad (replicateM)

-- Tokenizer structure equivalent to the C fields we need
data Tokenizer = Tokenizer
  { vocabVector    :: V.Vector BSL.ByteString  -- size = vocab_size
  , scoresVector   :: V.Vector Float           -- size = vocab_size
  , sortedMap      :: M.Map BSL.ByteString Int -- str -> id (like sorted_vocab + bsearch)
  , maxTokenLength :: !Int                     -- header field (not strictly needed)
  , spaceId        :: !Int                     -- id of " " (dummy-prefix)
  }

-- Parse tokenizer.bin exactly like C (little-endian)
buildTokenizer :: BSL.ByteString -> Int -> Tokenizer
buildTokenizer blob vocabSize =
  let (maxTokLen, vocabScores, vocab) = BG.runGet (parser vocabSize) blob
      vVec   = V.fromList vocab
      sVec   = V.fromList vocabScores
      m      = M.fromList (zip vocab [0..])
      spId   = fromMaybe (-1) (M.lookup " " m)  -- C looks up exactly one ASCII space
  in Tokenizer { vocabVector    = vVec
               , scoresVector   = sVec
               , sortedMap      = m
               , maxTokenLength = maxTokLen
               , spaceId        = spId
               }

  where
    parser :: Int -> BG.Get (Int, [Float], [BSL.ByteString])
    parser n = do
      maxTokLen <- fromIntegral <$> BG.getInt32le
      xs <- replicateM n readOne
      let (scs, toks) = unzip xs
      pure (maxTokLen, scs, toks)

    readOne :: BG.Get (Float, BSL.ByteString)
    readOne = do
      sc  <- BG.getFloatle
      len <- fromIntegral <$> BG.getInt32le
      str <- BG.getLazyByteString len
      pure (sc, str)

-- Public encode: exactly like C encode(tokenizer, text, bos, eos)
-- bos/eos flags mirror the C API (non-zero => include)
encodeTokens :: Tokenizer -> BSL.ByteString -> Bool -> Bool -> [Int]
encodeTokens tok text bos eos =
  let
    -- start with optional BOS (=1)
    start = [1 | bos]

    -- add dummy-prefix " " token if text != ""
    start' = if BSL.null text then start else start ++ [spaceId tok]

    -- initial tokens from UTF-8 codepoints with byte-fallback (+3)
    baseTokens = start' ++ utf8ToTokens tok text

    -- iterative BPE merges (same ordering/criterion as C)
    merged = bpeMergeLoop tok baseTokens

    -- optional EOS (=2)
    final = if eos then merged ++ [2] else merged
  in final

-- Decode piece exactly like C decode(): BOS rule + <0xXX> byte tokens
decodePiece :: Tokenizer -> Int -> Int -> BSL.ByteString
decodePiece tok prev tokenId
  | tokenId < 0 || tokenId >= V.length (vocabVector tok) = BSL.empty
  | otherwise =
      let piece0 = vocabVector tok V.! tokenId
          piece1 = if prev == 1 then dropLeadingSpace piece0 else piece0
      in case parseBytePiece piece1 of
           Just b  -> BSL.pack [b]
           Nothing -> piece1

-- Helpers ---------------------------------------------------------------------

-- BOS rule: if piece starts with single ASCII space (0x20), drop it
dropLeadingSpace :: BSL.ByteString -> BSL.ByteString
dropLeadingSpace bs =
  case BSL.uncons bs of
    Just (c, rest) | c == 0x20 -> rest
    _                          -> bs

-- Detect "<0xXX>" and parse to a single byte (accepts 0-9A-F; C used uppercase via %02hhX)
parseBytePiece :: BSL.ByteString -> Maybe Word8
parseBytePiece s =
  -- exact length 6: '<' '0' 'x' h h '>'
  -- Allow 'x' or 'X' and upper/lower hex digits
  case BSL.unpack s of
    [60,48,x,h1,h2,62] | x == 120 || x == 88
                       , isHex h1, isHex h2 ->
        Just (hexVal h1 * 16 + hexVal h2)
    _  -> Nothing
  where
    isHex w = (w >= 48 && w <= 57) || (w >= 65 && w <= 70) || (w >= 97 && w <= 102)
    hexVal w
      | w >= 48 && w <= 57  = fromIntegral (w - 48)       -- '0'..'9'
      | w >= 65 && w <= 70  = fromIntegral (w - 65 + 10)  -- 'A'..'F'
      | otherwise           = fromIntegral (w - 97 + 10)  -- 'a'..'f'

-- Convert UTF-8 text to token ids:
-- - accumulate bytes of a single UTF-8 codepoint (1..4 bytes)
-- - lookup the full codepoint in vocab; if found, emit its id
-- - else emit each byte as (byte + 3)  [exactly like C]
utf8ToTokens :: Tokenizer -> BSL.ByteString -> [Int]
utf8ToTokens tok bs = go (BSL.unpack bs) [] []
  where
    m = sortedMap tok

    go :: [Word8] -> [Word8] -> [Int] -> [Int]
    go [] cur acc =
      case finishCodepoint cur of
        []   -> reverse acc
        toks -> reverse acc ++ toks

    go (b:rest) cur acc =
      let cur' = if isContinuation b then cur ++ [b] else [b]
          nextIsCont = case rest of
                         (n:_) -> isContinuation n
                         []    -> False
          -- stop extending if next is not continuation or we already have 4 bytes
          completeNow = not nextIsCont || (length cur' >= 4)
      in if completeNow
           then let toks = finishCodepoint cur'
                 in go rest [] (reverse toks ++ acc) -- prepend toks to acc (we reverse at end)
           else go rest cur' acc

    finishCodepoint :: [Word8] -> [Int]
    finishCodepoint bytes =
      let buffer = BSL.pack bytes
      in case M.lookup buffer m of
           Just i -> [i]
           Nothing -> map (\w -> fromIntegral w + 3) bytes

    isContinuation :: Word8 -> Bool
    isContinuation b = (b .&. 0xC0) == 0x80

-- BPE merge loop: in each iteration, find the single best adjacent pair
-- whose concatenation exists in vocab with the highest score; merge it; repeat.
bpeMergeLoop :: Tokenizer -> [Int] -> [Int]
bpeMergeLoop tok = go
  where
    v    = vocabVector tok
    sc   = scoresVector tok
    m    = sortedMap tok
    negInf = -1.0e10 :: Float

    go toks =
      case findBest toks of
        Nothing -> toks
        Just (idx, newId) ->
          -- replace toks[idx], drop toks[idx+1]
          let toks' = take idx toks ++ [newId] ++ drop (idx + 2) toks
          in go toks'

    findBest :: [Int] -> Maybe (Int, Int)  -- (position, mergedTokenId)
    findBest toks =
      let pairs = zip [0..] (zip toks (drop 1 toks))
          step best (i, (a,b)) =
            let abStr = (v V.! a) `BSL.append` (v V.! b)
            in case M.lookup abStr m of
                 Just mid ->
                   let s = sc V.! mid
                   in case best of
                        Nothing           -> if s > negInf then Just (i, mid) else Nothing
                        Just (_, bestMid) ->
                          if s > (sc V.! bestMid) then Just (i, mid) else best
                 Nothing -> best
      in foldl step Nothing pairs
