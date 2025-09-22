module Model.Layers.Attention.AttentionHead (
    AttentionCache(..)
    , initAttentionCache
) where

import Model.Core.Types ( CacheDepth, CacheAddress )
import Clash.Prelude

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
