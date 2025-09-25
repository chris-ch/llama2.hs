module Model.Core.Types
  ( -- State machine
    CycleStage(..)
  , ProcessingState(..)
  , IntermediateData(..)
    -- Geometry and helpers
  , BankDepth
  , BankAddress
  , CacheDepth
  , CacheAddress
  , TrueDualPortRunner
  , Token
  , Temperature
  , Seed
  , HiddenDim
  , ModelDim
  ,  NumQueryHeads
  , NumLayers
  , NumKeyValueHeads
  , SeqLen
  , HeadDimension
  , FreqDim
  , VocabSize
  , SingleHeadComponent(..)
  , RotaryEncodingComponent(..)
  , EmbeddingComponent(..)
  , CArray2D(..)
  ) where

import Clash.Prelude
import qualified GHC.TypeNats
import GHC.Stack (HasCallStack)

{- 
-- model config 260K
type ModelDim = 64
type HiddenDim = 172
type NumLayers = 5
type NumQueryHeads = 8
type NumKeyValueHeads = 4
type HeadDimension  = 8
type FreqDim = 4
type VocabSize = 512 :: Nat
vocabSize :: Int
vocabSize = natToNum @VocabSize
type SeqLen         = 512
seqLen :: Int
seqLen = natToNum @SeqLen
 -}

-- model config 15M
type ModelDim = 288
type HiddenDim = 768
type NumLayers = 6
type NumQueryHeads = 6
type NumKeyValueHeads = 6
type HeadDimension  = 48
type FreqDim = 24
type VocabSize = 32000 :: Nat
type SeqLen         = 256

{- 
-- model config 42M
type ModelDim = 512
type HiddenDim = 1376
type NumLayers = 8
type NumQueryHeads = 8
type NumKeyValueHeads = 8
type HeadDimension  = 64
type FreqDim = 32
type VocabSize = 32000 :: Nat
vocabSize :: Int
vocabSize = natToNum @VocabSize
type SeqLen         = 1024
seqLen :: Int
seqLen = natToNum @SeqLen
 -}
{-
-- model config 110M
type ModelDim = 768
type HiddenDim = 2048
type NumLayers = 12
type NumQueryHeads = 12
type NumKeyValueHeads = 12
type HeadDimension  = 64
type FreqDim = 32
type VocabSize = 32000 :: Nat
vocabSize :: Int
vocabSize = natToNum @VocabSize
type SeqLen         = 1024
seqLen :: Int
seqLen = natToNum @SeqLen
-}

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

newtype CArray2D (n :: Nat) (m :: Nat) = CArray2D (Vec n (Vec m Float)) deriving (Show)

type Token = Unsigned 32
type Temperature = Float
type Seed = Unsigned 32

-- Data definitions for LLM architecture

data EmbeddingComponent = EmbeddingComponent
  { vocabulary :: CArray2D VocabSize ModelDim,
    rmsFinalWeight :: Vec ModelDim Float
  } deriving (Show)

data RotaryEncodingComponent = RotaryEncodingComponent
  { freqCos :: CArray2D SeqLen FreqDim,
    freqSin :: CArray2D SeqLen FreqDim
  } deriving (Show)

data SingleHeadComponent = SingleHeadComponent
  { wqHead :: CArray2D HeadDimension ModelDim
  , wkHead :: CArray2D HeadDimension ModelDim
  , wvHead :: CArray2D HeadDimension ModelDim
  , rotary :: RotaryEncodingComponent
  } deriving (Show)
