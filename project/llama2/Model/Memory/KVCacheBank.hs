module Model.Memory.KVCacheBank (
    KvBank(..)
  , KVRamOwner(..)
  , makeRamOwnerKV
  , writeSequencer
) where

import Clash.Prelude
import qualified Prelude as P

import Model.Core.Types (TrueDualPortRunner, BankDepth, BankAddress, NumKeyValueHeads, HeadDimension, SequenceLength)
import qualified Model.Memory.RamOps as RamOps (toRamOperation)
import qualified Model.Memory.Addressing as Addressing

data KvBank dom = KvBank
  { runKeyBank   :: TrueDualPortRunner dom BankDepth Float
  , runValueBank :: TrueDualPortRunner dom BankDepth Float
  }

-- Per-bank K/V dual-port (instantiate RAM once, feed both ports)
mkTrueDualPortRunner :: (HiddenClockResetEnable dom, KnownNat n, NFDataX a) => TrueDualPortRunner dom n a
mkTrueDualPortRunner (addressA, writeA) (addressB, writeB) =
  trueDualPortBlockRam (RamOps.toRamOperation addressA writeA) (RamOps.toRamOperation addressB writeB)

makeBankKV :: HiddenClockResetEnable dom => KvBank dom
makeBankKV = KvBank { runKeyBank = mkTrueDualPortRunner, runValueBank = mkTrueDualPortRunner }

newtype KVRamOwner dom = KVRamOwner
  { kvBanks :: Vec NumKeyValueHeads (KvBank dom)
  }

makeRamOwnerKV :: HiddenClockResetEnable dom => KVRamOwner dom
makeRamOwnerKV = KVRamOwner { kvBanks = map (const makeBankKV) indicesI }

-- Writer sequencer for one bank: generates (addr, write) streams and a done pulse.
-- Walks through HeadDimension over successive cycles, writing K and V vectors
-- to the bank. Emits a one-cycle done pulse after writing the last dimension.
writeSequencer
  :: HiddenClockResetEnable dom
  => Signal dom Bool                                    -- enable signal
  -> Signal dom (Index SequenceLength)                          -- sequence position signal
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

  bankAddressSignal  = Addressing.computeBankAddress <$> sequencePositionSignal <*> dimensionCounterSignal

  keyWriteSignal =
    mux enableSignal (Just <$> bundle (bankAddressSignal, keyElementSignal)) (pure Nothing)
  valueWriteSignal =
    mux enableSignal (Just <$> bundle (bankAddressSignal, valueElementSignal)) (pure Nothing)
