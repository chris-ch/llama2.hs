module Model.Cache
  ( -- KV banking
    toRamOperation
  , computeBankAddress
  , KvBank(..)
  , KVRamOwner(..)
  , makeRamOwnerKV
  , writeSequencer
    -- Global caches
  , AttentionCache(..)
  , initAttentionCache
  ) where

import Clash.Prelude
import qualified Prelude as P

import Model.Types
  ( BankDepth, BankAddress, CacheDepth, CacheAddress, TrueDualPortRunner )

import Helpers ( NumKeyValueHeads, HeadDimension, SeqLen )

import Data.Maybe (isJust)

-- Convert (address + optional write) into a RAM operation stream
toRamOperation
  :: NFDataX a
  => Signal dom (Index n)              -- read address signal
  -> Signal dom (Maybe (Index n, a))   -- optional write address + data signal
  -> Signal dom (RamOp n a)
toRamOperation addressSignal writeMaybeSignal =
  mux (isJust <$> writeMaybeSignal)
      (uncurry RamWrite . fromJustX <$> writeMaybeSignal)
      (RamRead <$> addressSignal)

-- Compute bank address from sequence index and head-dimension index
computeBankAddress :: Index SeqLen -> Index HeadDimension -> BankAddress
computeBankAddress sequenceIndex headDimIndex =
  toEnum (fromIntegral headDimIndex + fromIntegral sequenceIndex * natToNum @HeadDimension)

-- Per-bank K/V dual-port (instantiate RAM once, feed both ports)
mkTrueDualPortRunner :: (HiddenClockResetEnable dom, KnownNat n, NFDataX a) => TrueDualPortRunner dom n a
mkTrueDualPortRunner (addressA, writeA) (addressB, writeB) =
  trueDualPortBlockRam (toRamOperation addressA writeA) (toRamOperation addressB writeB)

data KvBank dom = KvBank
  { runKeyBank   :: TrueDualPortRunner dom BankDepth Float
  , runValueBank :: TrueDualPortRunner dom BankDepth Float
  }

makeBankKV :: HiddenClockResetEnable dom => KvBank dom
makeBankKV = KvBank { runKeyBank = mkTrueDualPortRunner, runValueBank = mkTrueDualPortRunner }

newtype KVRamOwner dom = KVRamOwner
  { kvBanks :: Vec NumKeyValueHeads (KvBank dom)
  }

makeRamOwnerKV :: HiddenClockResetEnable dom => KVRamOwner dom
makeRamOwnerKV = KVRamOwner { kvBanks = map (const makeBankKV) indicesI }

-- Writer sequencer for one bank: generates (addr, write) streams and a done pulse.
writeSequencer
  :: HiddenClockResetEnable dom
  => Signal dom Bool                                    -- enable signal
  -> Signal dom (Index SeqLen)                          -- sequence position signal
  -> Signal dom (Vec HeadDimension Float, Vec HeadDimension Float) -- K and V vectors signal
  -> ( Signal dom BankAddress
     , Signal dom (Maybe (BankAddress, Float))  -- K write signal
     , Signal dom (Maybe (BankAddress, Float))  -- V write signal
     , Signal dom Bool)                      -- done (1-cycle on last d)
writeSequencer enableSignal sequencePositionSignal keyValueVectorsSignal =
  (bankAddressSignal, keyWriteSignal, valueWriteSignal, doneSignal)
 where
  dimensionCounterSignal   = register 0 nextDimensionCounterSignal
  nextDimensionCounterSignal =
    mux enableSignal
        (P.fmap (\d -> if d == maxBound then 0 else succ d) dimensionCounterSignal)
        (pure 0)

  isLastDimensionSignal    = (== maxBound) <$> dimensionCounterSignal
  doneSignal               = (&&) <$> enableSignal <*> isLastDimensionSignal

  keyElementSignal   = (\(kVec,_) d -> kVec !! d) <$> keyValueVectorsSignal <*> dimensionCounterSignal
  valueElementSignal = (\(_,vVec) d -> vVec !! d) <$> keyValueVectorsSignal <*> dimensionCounterSignal

  bankAddressSignal  = computeBankAddress <$> sequencePositionSignal <*> dimensionCounterSignal

  keyWriteSignal =
    mux enableSignal (Just <$> bundle (bankAddressSignal, keyElementSignal)) (pure Nothing)
  valueWriteSignal =
    mux enableSignal (Just <$> bundle (bankAddressSignal, valueElementSignal)) (pure Nothing)

-- ----------------------------------------------------------------------------
-- Global attention caches (all layers); simple 1-port RAMs

data AttentionCache dom = AttentionCache
  { keyCacheRam   :: Signal dom CacheAddress -> Signal dom (Maybe (CacheAddress, Float)) -> Signal dom Float
  , valueCacheRam :: Signal dom CacheAddress -> Signal dom (Maybe (CacheAddress, Float)) -> Signal dom Float
  }

initAttentionCache :: forall dom. HiddenClockResetEnable dom => AttentionCache dom
initAttentionCache = AttentionCache
  { keyCacheRam   = blockRam (replicate (SNat @CacheDepth) 0)
  , valueCacheRam = blockRam (replicate (SNat @CacheDepth) 0)
  }
