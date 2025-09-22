module Model.Types
  ( -- State machine
    CycleStage(..)
  , ProcessingState(..)
  , initialProcessingState
  , nextState
    -- Intermediate data
  , IntermediateData(..)
  , initialIntermediateData
    -- Geometry and helpers
  , BankDepth
  , BankAddr
  , CacheDepth
  , CacheAddr
  , TdprRunner
  ) where

import Clash.Prelude
import qualified GHC.TypeNats
import GHC.Stack (HasCallStack)

import Helpers
  ( NumQueryHeads, NumKeyValueHeads, NumLayers, SeqLen, HeadDimension, ModelDim )

-- Bank geometry
type BankDepth = SeqLen GHC.TypeNats.* HeadDimension
type BankAddr  = Index BankDepth

-- Global KV-cache geometry (all layers × KV heads × seq × headDim)
type CacheDepth = NumLayers GHC.TypeNats.* NumKeyValueHeads GHC.TypeNats.* SeqLen GHC.TypeNats.* HeadDimension
type CacheAddr  = Index CacheDepth

-- Dual-port RAM runner type (true dual port)
type TdprRunner dom n a =
       ( Signal dom (Index n)               -- Port A address
       , Signal dom (Maybe (Index n, a)) )  -- Port A write (optional)
    -> ( Signal dom (Index n)               -- Port B address
       , Signal dom (Maybe (Index n, a)) )  -- Port B write (optional)
    -> ( Signal dom a                       -- qA
       , Signal dom a )                     -- qB

-- ============================================================================
-- Multi-Cycle State Machine
-- ============================================================================

data CycleStage =
    Cycle1_ReadCache
  | Cycle2_ComputeQKV
  | Cycle3_ComputeAttn
  | Cycle4_WriteCache
  | Cycle5_ComputeFFN
  deriving (Show, Eq, Enum, Bounded, Generic)

instance NFDataX CycleStage where
  rnfX :: CycleStage -> ()
  rnfX x = seq x ()
  hasUndefined :: CycleStage -> Bool
  hasUndefined _ = False
  ensureSpine :: CycleStage -> CycleStage
  ensureSpine x = x
  deepErrorX :: HasCallStack => String -> CycleStage
  deepErrorX = errorX

data ProcessingState = ProcessingState
  { psStage  :: CycleStage
  , psLayer  :: Index NumLayers
  , psSeqPos :: Index SeqLen
  } deriving (Show, Generic, NFDataX)

initialProcessingState :: ProcessingState
initialProcessingState = ProcessingState
  { psStage  = Cycle1_ReadCache
  , psLayer  = 0
  , psSeqPos = 0
  }

-- Single state transition function (one step)
nextState :: ProcessingState -> ProcessingState
nextState st = case psStage st of
  Cycle1_ReadCache   -> st { psStage = Cycle2_ComputeQKV }
  Cycle2_ComputeQKV  -> st { psStage = Cycle3_ComputeAttn }
  Cycle3_ComputeAttn -> st { psStage = Cycle4_WriteCache }
  Cycle4_WriteCache  -> st { psStage = Cycle5_ComputeFFN }
  Cycle5_ComputeFFN ->
    if psLayer st == maxBound
      then st { psStage  = Cycle1_ReadCache
              , psLayer  = 0
              , psSeqPos = if psSeqPos st == maxBound then 0 else succ (psSeqPos st)
              }
      else st { psStage  = Cycle1_ReadCache
              , psLayer  = succ (psLayer st)
              }

-- ============================================================================
-- Intermediate data storage
-- ============================================================================

data IntermediateData = IntermediateData
  { idInputVec   :: Vec ModelDim Float
  , idQueries    :: Vec NumQueryHeads (Vec HeadDimension Float)
  , idKeys       :: Vec NumKeyValueHeads (Vec HeadDimension Float)
  , idValues     :: Vec NumKeyValueHeads (Vec HeadDimension Float)
  , idAttnOutput :: Vec ModelDim Float
  , idFFNOutput  :: Vec ModelDim Float
  } deriving (Show, Generic, NFDataX)

initialIntermediateData :: IntermediateData
initialIntermediateData = IntermediateData
  { idInputVec   = repeat 0
  , idQueries    = repeat (repeat 0)
  , idKeys       = repeat (repeat 0)
  , idValues     = repeat (repeat 0)
  , idAttnOutput = repeat 0
  , idFFNOutput  = repeat 0
  }
  