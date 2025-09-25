module Model.Top
  ( topEntity
  ) where

import Clash.Prelude

import Model.Core.Types
  (
  Temperature, Seed, ModelDim, NumLayers, SeqLen
  )

import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.TransformerLayer as TransformerLayer (TransformerDecoderComponent)
import qualified Model.Core.Transformer as Transformer

-- ====== NEW: top with attention tap out ======
topEntity
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
topEntity decoder =
  Transformer.multiCycleTransformer decoder (repeat Cache.makeRamOwnerKV)
