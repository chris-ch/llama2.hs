module Model.Memory.Addressing (
    computeBankAddress
) where

import Clash.Prelude

import Model.Core.Types (BankAddress, SeqLen, HeadDimension)

-- Compute bank address from sequence index and head-dimension index
-- Assumes headDimIndex in [0, HeadDimension-1], sequenceIndex in [0, SeqLen-1].
-- Bank address = time * HeadDimension + headDimIndex.
computeBankAddress :: Index SeqLen -> Index HeadDimension -> BankAddress
computeBankAddress sequenceIndex headDimIndex =
  toEnum (fromIntegral headDimIndex + fromIntegral sequenceIndex * natToNum @HeadDimension)
