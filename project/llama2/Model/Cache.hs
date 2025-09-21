module Model.Cache
  ( -- KV banking
    toOp
  , bankAddr
  , KvBank(..)
  , KVRamOwner(..)
  , mkKVRamOwner
  , writeSequencer
    -- Global caches
  , AttentionCache(..)
  , initAttentionCache
  ) where

import Clash.Prelude
import qualified Prelude as P

import Model.Types
  ( BankDepth, BankAddr, CacheDepth, CacheAddr, TdprRunner )

import Helpers ( NumKeyValueHeads, HeadDimension, SeqLen )

import Data.Maybe (isJust)

-- Turn (addr, maybe write (addr,data)) into a RamOp stream
toOp
  :: NFDataX a => Signal dom (Index n)
  -> Signal dom (Maybe (Index n, a))
  -> Signal dom (RamOp n a)
toOp addrSig wrSig =
  mux (isJust <$> wrSig) (uncurry RamWrite . fromJustX <$> wrSig) (RamRead <$> addrSig)

bankAddr :: Index SeqLen -> Index HeadDimension -> BankAddr
bankAddr s d =
  toEnum (fromIntegral d + fromIntegral s * natToNum @HeadDimension)

-- Per-bank K/V dual-port (instantiate RAM once, feed both ports)
mkTdprRunner :: (HiddenClockResetEnable dom, KnownNat n, NFDataX a) => TdprRunner dom n a
mkTdprRunner (addrA, wrA) (addrB, wrB) = trueDualPortBlockRam (toOp addrA wrA) (toOp addrB wrB)

data KvBank dom = KvBank
  { runK :: TdprRunner dom BankDepth Float
  , runV :: TdprRunner dom BankDepth Float
  }

mkKvBank :: HiddenClockResetEnable dom => KvBank dom
mkKvBank = KvBank { runK = mkTdprRunner, runV = mkTdprRunner }

newtype KVRamOwner dom = KVRamOwner
  { kvBanks :: Vec NumKeyValueHeads (KvBank dom)
  }

mkKVRamOwner :: HiddenClockResetEnable dom => KVRamOwner dom
mkKVRamOwner = KVRamOwner { kvBanks = map (const mkKvBank) indicesI }

-- Writer sequencer for one bank: generates (addr, write) streams and a done pulse.
writeSequencer
  :: HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom (Index SeqLen)
  -> Signal dom (Vec HeadDimension Float, Vec HeadDimension Float)
  -> ( Signal dom BankAddr
     , Signal dom (Maybe (BankAddr, Float))  -- K write
     , Signal dom (Maybe (BankAddr, Float))  -- V write
     , Signal dom Bool)                      -- done (1-cycle on last d)
writeSequencer enSig seqPosSig kvSig = (addr, kWr, vWr, doneSig)
 where
  dCnt   = register 0 nextD
  nextD  = mux enSig (P.fmap (\d -> if d == maxBound then 0 else succ d) dCnt) (pure 0)
  lastD  = (== maxBound) <$> dCnt
  doneSig = (&&) <$> enSig <*> lastD

  kElem  = (\(k,_) d -> k !! d) <$> kvSig <*> dCnt
  vElem  = (\(_,v) d -> v !! d) <$> kvSig <*> dCnt
  addr   = bankAddr <$> seqPosSig <*> dCnt

  kWr    = mux enSig (Just <$> bundle (addr, kElem)) (pure Nothing)
  vWr    = mux enSig (Just <$> bundle (addr, vElem)) (pure Nothing)

-- ----------------------------------------------------------------------------
-- Global attention caches (all layers); simple 1-port RAMs

data AttentionCache dom = AttentionCache
  { keyCache   :: Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float
  , valueCache :: Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float
  }

initAttentionCache :: forall dom. HiddenClockResetEnable dom => AttentionCache dom
initAttentionCache = AttentionCache
  { keyCache   = blockRam (replicate (SNat @CacheDepth) 0)
  , valueCache = blockRam (replicate (SNat @CacheDepth) 0)
  }
