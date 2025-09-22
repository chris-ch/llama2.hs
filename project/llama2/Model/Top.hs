module Model.Top
  ( topEntity
  ) where

import Clash.Prelude

import Helpers
  ( TransformerDecoderComponent(..)
  
  , Temperature, Seed
  )


import Model.Cache
  ( makeRamOwnerKV)

import Model.Top.Transformer (multiCycleTransformer)


-- ============================================================================
-- Top Entity
-- ============================================================================
topEntity
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> Signal dom (Unsigned 32)  -- Input token
  -> Signal dom Temperature
  -> Signal dom Seed
  -> (Signal dom (Unsigned 32), Signal dom Bool)
topEntity decoder = multiCycleTransformer decoder (repeat makeRamOwnerKV)
