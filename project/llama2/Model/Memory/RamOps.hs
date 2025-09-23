module Model.Memory.RamOps (
    toRamOperation
) where

import Clash.Prelude
import qualified Prelude as P
import Data.Maybe (isJust)

-- Convert separate read and optional-write signals into a unified RAM operation stream.
--   - If write is present, emit a RamWrite
--   - Otherwise emit a RamRead
toRamOperation
  :: NFDataX a
  => Signal dom (Index n)              -- read address signal
  -> Signal dom (Maybe (Index n, a))   -- optional write address + data signal
  -> Signal dom (RamOp n a)
toRamOperation addressSignal writeMaybeSignal =
  mux (isJust <$> writeMaybeSignal)
      (uncurry RamWrite . fromJustX <$> writeMaybeSignal)
      (RamRead <$> addressSignal)
