module Model.Top
  ( topEntity
  ) where

import Clash.Prelude

import Helpers
  ( 
  Temperature, Seed
  )

import Model.Top.Transformer (multiCycleTransformer)
import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.TransformerLayer as TransformerLayer (TransformerDecoderComponent)


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
