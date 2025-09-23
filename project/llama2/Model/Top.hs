module Model.Top
  ( topEntity, topEntityWithTap
  ) where

import Clash.Prelude

import Helpers
  (
  Temperature, Seed, ModelDim, NumLayers, SeqLen
  )

import Model.Top.Transformer (multiCycleTransformer)
import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.TransformerLayer as TransformerLayer (TransformerDecoderComponent)
import qualified Model.Top.Transformer as Transformer


-- ============================================================================
-- Top Entity
-- ============================================================================
topEntity
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Signal dom (Unsigned 32)  -- Input token
  -> Signal dom Temperature
  -> Signal dom Seed
  -> (Signal dom (Unsigned 32), Signal dom Bool)
topEntity decoder = multiCycleTransformer decoder (repeat Cache.makeRamOwnerKV)

-- ====== NEW: top with attention tap out ======
topEntityWithTap
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Signal dom (Unsigned 32)  -- Input token
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom (Unsigned 32)         -- sampled token
     , Signal dom Bool                  -- ready pulse (end of last FFN)
     , Signal dom Bool                  -- tap pulse (end of Cycle3 for active layer)
     , Signal dom (Index NumLayers)     -- tap layer index
     , Signal dom (Index SeqLen)        -- tap sequence position
     , Signal dom (Vec ModelDim Float)  -- dbgXHat
     , Signal dom (Vec ModelDim Float)  -- dbgWOHeads
     , Signal dom (Vec ModelDim Float)  -- dbgXAfterAttn
     )
topEntityWithTap decoder =
  Transformer.multiCycleTransformerWithTap decoder (repeat Cache.makeRamOwnerKV)
