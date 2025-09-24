module Model.Core.PipelineController
  ( PipelineOutputs(..)
  , runPipelineController
  ) where

import Clash.Prelude
import Helpers (NumLayers, SeqLen)
import Model.Core.Types
  ( ProcessingState(..), CycleStage(..)
  , initialProcessingState, nextProcessingState
  )

data PipelineOutputs dom = PipelineOutputs
  { processingState   :: Signal dom ProcessingState
  , stageSignal       :: Signal dom CycleStage
  , layerIndex        :: Signal dom (Index NumLayers)
  , seqPos            :: Signal dom (Index SeqLen)
  , readyPulse        :: Signal dom Bool
  , stageFinished     :: Signal dom Bool
  }

-- Drive the global FSM and readyPulse. The two inputs are already
-- selected for “this layer” (i.e. post (!!) mux).
runPipelineController
  :: HiddenClockResetEnable dom
  => Signal dom Bool  -- attnDoneThisLayer (Cycle3)
  -> Signal dom Bool  -- writeDoneThisLayer (Cycle4)
  -> PipelineOutputs dom
runPipelineController attnDoneThisLayer writeDoneThisLayer = outs
 where
  -- State register
  advance s done = if done then nextProcessingState s else s
  procState = register initialProcessingState (advance <$> procState <*> stageFinishedSig)

  -- Convenience fields
  stageSig = processingStage <$> procState
  layerIx  = processingLayer <$> procState
  posIx    = sequencePosition <$> procState

  -- Ready pulse: last layer finishing Cycle5
  isLastLayerFFN =
    liftA2 (\ps _ -> processingStage ps == Cycle5_ComputeFeedForward
                  && processingLayer ps == maxBound)
           procState (pure ())
  readyPulseRaw =
    let rising now prev = now && not prev
    in liftA2 rising isLastLayerFFN (register False isLastLayerFFN)

  -- “when to advance” policy
  isStage st = (== st) <$> stageSig
  stageFinishedSig =
    mux (isStage Cycle1_ReadCache)           (pure True)               $
    mux (isStage Cycle2_ComputeQKV)          (pure True)               $
    mux (isStage Cycle3_ComputeAttention)    attnDoneThisLayer         $
    mux (isStage Cycle4_WriteCache)          writeDoneThisLayer        $
    mux (isStage Cycle5_ComputeFeedForward)  (not <$> readyPulseRaw)   $
    pure False

  outs = PipelineOutputs
    { processingState = procState
    , stageSignal     = stageSig
    , layerIndex      = layerIx
    , seqPos          = posIx
    , readyPulse      = readyPulseRaw
    , stageFinished   = stageFinishedSig
    }
