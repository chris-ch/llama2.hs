module Model.Top
  ( topEntity
  ) where

import Clash.Prelude

import Model.Core.Types
  ( Temperature, Seed, ModelDim, NumLayers, SeqLen, Token
  )

import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.TransformerLayer as TransformerLayer (TransformerDecoderComponent)
import qualified Model.Core.Transformer as Transformer

-- ====== NEW: top with attention tap out (includes pre-WO concat) ======
topEntity
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Signal dom Token  -- Input token
  -> Signal dom Bool           -- ^ inputTokenValid: high when inputTokenSignal carries the prompt token (pos 0)
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token                -- sampled token
     , Signal dom Bool                 -- ready pulse (end of last FFN)
     , Signal dom Bool                 -- tap pulse (end of Cycle3 for active layer)
     , Signal dom (Index NumLayers)    -- tap layer index
     , Signal dom (Index SeqLen)       -- tap sequence position
     , Signal dom (Vec ModelDim Float) -- dbgXHat
     )
topEntity decoder = Transformer.multiCycleTransformer decoder (repeat Cache.makeRamOwnerKV)
