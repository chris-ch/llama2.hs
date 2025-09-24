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

runPipelineController
  :: HiddenClockResetEnable dom
  => Signal dom Bool  -- attnDoneThisLayer (Cycle3)
  -> Signal dom Bool  -- writeDoneThisLayer (Cycle4)
  -> PipelineOutputs dom
runPipelineController attnDoneThisLayer writeDoneThisLayer = outs
 where
  advance s done = if done then nextProcessingState s else s
  procState = register initialProcessingState (advance <$> procState <*> stageFinishedSig)

  stageSig = processingStage <$> procState
  layerIx  = processingLayer <$> procState
  posIx    = sequencePosition <$> procState

  isLastLayerFFN =
    liftA2 (\ps _ -> processingStage ps == Stage5_FeedForward
                  && processingLayer ps == maxBound)
           procState (pure ())
  readyPulseRaw =
    let rising now prev = now && not prev
    in liftA2 rising isLastLayerFFN (register False isLastLayerFFN)

  isStage st = (== st) <$> stageSig
  stageFinishedSig =
    mux (isStage Stage1_LoadKV)           (pure True)               $
    mux (isStage Stage2_ProjectQKV)          (pure True)               $
    mux (isStage Stage3_Attend)    attnDoneThisLayer         $
    mux (isStage Stage4_WriteKV)          writeDoneThisLayer        $
    mux (isStage Stage5_FeedForward)  (not <$> readyPulseRaw)   $
    pure False

  outs = PipelineOutputs
    { processingState = procState
    , stageSignal     = stageSig
    , layerIndex      = layerIx
    , seqPos          = posIx
    , readyPulse      = readyPulseRaw
    , stageFinished   = stageFinishedSig
    }
