module Model.Core.PipelineController
  ( PipelineOutputs(..)
  , runPipelineController
  ) where

import Clash.Prelude
import Model.Core.Types (NumLayers, SeqLen, ProcessingState(..), CycleStage(..))

initialProcessingState :: ProcessingState
initialProcessingState = ProcessingState
  { processingStage  = Stage1_LoadKV
  , processingLayer  = 0
  , sequencePosition = 0
  }

-- Single state transition function (one step)
nextProcessingState :: ProcessingState -> ProcessingState
nextProcessingState state = case processingStage state of
  Stage1_LoadKV      -> state { processingStage = Stage2_ProjectQKV }
  Stage2_ProjectQKV  -> state { processingStage = Stage3_Attend }
  Stage3_Attend      -> state { processingStage = Stage4_WriteKV }
  Stage4_WriteKV     -> state { processingStage = Stage5_FeedForward }
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
  => Signal dom Bool
  -> Signal dom Bool
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
    mux (isStage Stage1_LoadKV)      (pure True)               $
    mux (isStage Stage2_ProjectQKV)  (pure True)               $
    mux (isStage Stage3_Attend)      attnDoneThisLayer         $
    mux (isStage Stage4_WriteKV)     writeDoneThisLayer        $
    mux (isStage Stage5_FeedForward) (not <$> readyPulseRaw)   $
    pure False

  outs = PipelineOutputs
    { processingState = procState
    , stageSignal     = stageSig
    , layerIndex      = layerIx
    , seqPos          = posIx
    , readyPulse      = readyPulseRaw
    , stageFinished   = stageFinishedSig
    }
