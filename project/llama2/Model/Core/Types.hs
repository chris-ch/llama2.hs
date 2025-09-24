module Model.Core.Types
  ( -- State machine
    CycleStage(..)
  , ProcessingState(..)
  , initialProcessingState
  , nextProcessingState
    -- Intermediate data
  , IntermediateData(..)
  , initialIntermediateData
    -- Geometry and helpers
  , BankDepth
  , BankAddress
  , CacheDepth
  , CacheAddress
  , TrueDualPortRunner
  ) where

import Clash.Prelude
import qualified GHC.TypeNats
import GHC.Stack (HasCallStack)

import Helpers
  ( NumQueryHeads, NumKeyValueHeads, NumLayers, SeqLen, HeadDimension, ModelDim )

-- ============================================================================
-- Bank and Cache Geometry
-- ============================================================================

type BankDepth   = SeqLen GHC.TypeNats.* HeadDimension
type BankAddress = Index BankDepth

-- Global KV-cache geometry (all layers × KV heads × seq × headDim)
type CacheDepth   = NumLayers GHC.TypeNats.* NumKeyValueHeads GHC.TypeNats.* SeqLen GHC.TypeNats.* HeadDimension
type CacheAddress = Index CacheDepth

-- Dual-port RAM runner type (true dual port)
type TrueDualPortRunner dom n a =
       ( Signal dom (Index n)               -- Port A address
       , Signal dom (Maybe (Index n, a)) )  -- Port A write (optional)
    -> ( Signal dom (Index n)               -- Port B address
       , Signal dom (Maybe (Index n, a)) )  -- Port B write (optional)
    -> ( Signal dom a                       -- Port A read output
       , Signal dom a )                     -- Port B read output

-- ============================================================================
-- Multi-Cycle State Machine
-- ============================================================================

-- FSM stages for each layer:
--   1. ReadCache      (load K/V from RAM)
--   2. ComputeQKV     (compute Q/K/V from input)
--   3. ComputeAttention (stream attention)
--   4. WriteCache     (store new K/V to RAM)
--   5. ComputeFeedForward (FFN)
data CycleStage =
    Stage1_LoadKV
  | Stage2_ProjectQKV
  | Stage3_Attend
  | Stage4_WriteKV
  | Stage5_FeedForward
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

-- Tracks which stage, which layer, and which sequence position
-- the pipeline is currently processing.
data ProcessingState = ProcessingState
  { processingStage  :: CycleStage
  , processingLayer  :: Index NumLayers
  , sequencePosition :: Index SeqLen
  } deriving (Show, Generic, NFDataX)

initialProcessingState :: ProcessingState
initialProcessingState = ProcessingState
  { processingStage  = Stage1_LoadKV
  , processingLayer  = 0
  , sequencePosition = 0
  }

-- Single state transition function (one step)
nextProcessingState :: ProcessingState -> ProcessingState
nextProcessingState state = case processingStage state of
  Stage1_LoadKV          -> state { processingStage = Stage2_ProjectQKV }
  Stage2_ProjectQKV         -> state { processingStage = Stage3_Attend }
  Stage3_Attend   -> state { processingStage = Stage4_WriteKV }
  Stage4_WriteKV         -> state { processingStage = Stage5_FeedForward }
  Stage5_FeedForward ->
    if processingLayer state == maxBound
      then state { processingStage  = Stage1_LoadKV
                 , processingLayer  = 0
                 , sequencePosition = if sequencePosition state == maxBound
                                        then 0 else succ (sequencePosition state)
                 }
      else state { processingStage  = Stage1_LoadKV
                 , processingLayer  = succ (processingLayer state)
                 }

-- ============================================================================
-- Intermediate Data Storage
-- ============================================================================

-- Per-layer intermediate data vectors carried through the pipeline.
-- Updated selectively depending on cycle stage.
data IntermediateData = IntermediateData
  { inputVector       :: Vec ModelDim Float
  , queryVectors      :: Vec NumQueryHeads (Vec HeadDimension Float)
  , keyVectors        :: Vec NumKeyValueHeads (Vec HeadDimension Float)
  , valueVectors      :: Vec NumKeyValueHeads (Vec HeadDimension Float)
  , attentionOutput   :: Vec ModelDim Float
  , feedForwardOutput :: Vec ModelDim Float
  } deriving (Show, Generic, NFDataX)

initialIntermediateData :: IntermediateData
initialIntermediateData = IntermediateData
  { inputVector       = repeat 0
  , queryVectors      = repeat (repeat 0)
  , keyVectors        = repeat (repeat 0)
  , valueVectors      = repeat (repeat 0)
  , attentionOutput   = repeat 0
  , feedForwardOutput = repeat 0
  }
