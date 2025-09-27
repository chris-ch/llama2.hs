module Model.Layers.Attention.AttentionHead (
  streamHeadAttentionAddrIO
) where

import Model.Core.Types (HeadDimension, SeqLen, BankAddress)
import Clash.Prelude
import qualified Model.Memory.Addressing as Addressing

-- FSM phases of the attention head
data AttentionPhase = PhaseDotProduct | PhaseAccumulate | PhaseFinalize
  deriving (Generic, NFDataX, Eq, Show)

-- Full FSM / working state
data AttentionState = AttentionState
  { currentPhase      :: AttentionPhase
  , isRunning         :: Bool
  , prevStartSignal   :: Bool
  , currentSeqIndex   :: Index SeqLen
  , queryVector       :: Vec HeadDimension Float
  , keyBypassVector   :: Vec HeadDimension Float
  , valueBypassVector :: Vec HeadDimension Float
  , seqIndexCounter   :: Index SeqLen
  , headDimCounter    :: Index HeadDimension
  , prevSeqIndex      :: Index SeqLen
  , prevHeadDimIndex  :: Index HeadDimension
  , prevDataValid     :: Bool
  , prevKeyValue      :: Float
  , prevValueValue    :: Float
  , attentionScores   :: Vec SeqLen Float
  , maxAttentionScore :: Float
  , expDenominator    :: Float
  , isSecondPass      :: Bool
  , outputVector      :: Vec HeadDimension Float
  , dotAccumulator    :: Float
  , currentAddress    :: BankAddress
  , donePulse         :: Bool
  }
  deriving (Generic, NFDataX)

type AttentionInput =
  ( Bool                           -- start
  , Index SeqLen                   -- current sequence position
  , Vec HeadDimension Float        -- query vector (Q)
  , Vec HeadDimension Float        -- current key vector bypass (K[pos])
  , Vec HeadDimension Float        -- current value vector bypass (V[pos])
  , Float                          -- key RAM read data
  , Float                          -- value RAM read data
  )

type AttentionOutput =
  ( BankAddress
  , Vec HeadDimension Float        -- attended head output vector
  , Bool                           -- busy
  , Bool                           -- done (1-cycle pulse)
  )

streamHeadAttentionAddrIO
  :: (HiddenClockResetEnable dom)
  => Signal dom Bool                               -- start (true during Stage3_Attend)
  -> Signal dom (Index SeqLen)                     -- current sequence position
  -> Signal dom (Vec HeadDimension Float)          -- query vector Q
  -> Signal dom (Vec HeadDimension Float)          -- current key bypass K(pos)
  -> Signal dom (Vec HeadDimension Float)          -- current value bypass V(pos)
  -> Signal dom Float                              -- key RAM read data
  -> Signal dom Float                              -- value RAM read data
  -> ( Signal dom BankAddress
     , Signal dom (Vec HeadDimension Float)
     , Signal dom Bool
     , Signal dom Bool )
streamHeadAttentionAddrIO startSignal seqIndexSignal qSignal kBypassSignal vBypassSignal kRamSignal vRamSignal =
  (addressOut, outputVecOut, busyOut, doneOut)
 where
  -- Constants
  scale :: Float
  scale = 1.0 / sqrt ((natToNum @HeadDimension) :: Float)

  negBig :: Float
  negBig = -1.0e30

  initialState :: AttentionState
  initialState = AttentionState
    { currentPhase      = PhaseDotProduct
    , isRunning         = False
    , prevStartSignal   = False
    , currentSeqIndex   = 0
    , queryVector       = repeat 0
    , keyBypassVector   = repeat 0
    , valueBypassVector = repeat 0
    , seqIndexCounter   = 0
    , headDimCounter    = 0
    , prevSeqIndex      = 0
    , prevHeadDimIndex  = 0
    , prevDataValid     = False
    , prevKeyValue      = 0
    , prevValueValue    = 0
    , attentionScores   = repeat 0
    , maxAttentionScore = negBig
    , expDenominator    = 0
    , isSecondPass      = False
    , outputVector      = repeat 0
    , dotAccumulator    = 0
    , currentAddress    = 0
    , donePulse         = False
    }

  step :: AttentionState -> AttentionInput -> (AttentionState, AttentionOutput)
  step st (startNow, seqIndexIn, qIn, kBypassIn, vBypassIn, kRamIn, vRamIn) =
    let
      -- detect rising edge of start signal
      startRise = startNow && not (prevStartSignal st)

      -- reset on new start
      st0 = if startRise
        then (initialState
                { isRunning         = True
                , currentPhase      = PhaseDotProduct
                , currentSeqIndex   = seqIndexIn
                , queryVector       = qIn
                , keyBypassVector   = kBypassIn
                , valueBypassVector = vBypassIn })
        else st { donePulse = False }

      seqIdx     = currentSeqIndex st0
      qVec       = queryVector st0
      kByp       = keyBypassVector st0
      vByp       = valueBypassVector st0

      incHeadDim d' = if d' == maxBound then 0 else succ d'
      incSeqIdx t'  = if t' == maxBound then maxBound else succ t'
      lastHeadDim d' = d' == maxBound
      lastSeqIdx t' = t' == seqIdx

      addressThisCycle =
        case currentPhase st0 of
          PhaseDotProduct ->
            Addressing.computeBankAddress (seqIndexCounter st0) (headDimCounter st0)
          PhaseAccumulate ->
            if isSecondPass st0
              then Addressing.computeBankAddress (seqIndexCounter st0) (headDimCounter st0)
              else 0
          PhaseFinalize -> 0

      -- RAM read values align with previous issued address
      kAligned = if prevSeqIndex st0 == seqIdx
                   then kByp !! prevHeadDimIndex st0 else prevKeyValue st0
      vAligned = if prevSeqIndex st0 == seqIdx
                   then vByp !! prevHeadDimIndex st0 else prevValueValue st0
      qAligned = qVec !! prevHeadDimIndex st0

      dotAccNow =
        if isRunning st0 && currentPhase st0 == PhaseDotProduct && prevDataValid st0
          then dotAccumulator st0 + qAligned * kAligned
          else dotAccumulator st0

      endOfPrevVec = prevDataValid st0 && lastHeadDim (prevHeadDimIndex st0)

      scoreThisStep = dotAccNow * scale
      scores'       = if endOfPrevVec
                        then replace (prevSeqIndex st0) scoreThisStep (attentionScores st0)
                        else attentionScores st0
      maxScore'     = if endOfPrevVec
                        then max (maxAttentionScore st0) scoreThisStep
                        else maxAttentionScore st0
      dotAcc'       = if endOfPrevVec then 0 else dotAccNow

      dotPhaseDone  = endOfPrevVec && lastSeqIdx (prevSeqIndex st0)

      dNext = incHeadDim (headDimCounter st0)
      tNext = if lastHeadDim (headDimCounter st0)
                then (if lastSeqIdx (seqIndexCounter st0) then seqIndexCounter st0 else incSeqIdx (seqIndexCounter st0))
                else seqIndexCounter st0

      -- Next state after DOT phase
      st1 = case currentPhase st0 of
              PhaseDotProduct ->
                if isRunning st0
                  then if dotPhaseDone
                         then st0 { currentPhase      = PhaseAccumulate
                                  , seqIndexCounter   = 0
                                  , headDimCounter    = 0
                                  , prevDataValid     = False
                                  , dotAccumulator    = 0
                                  , attentionScores   = scores'
                                  , maxAttentionScore = maxScore'
                                  , currentAddress    = 0
                                  }
                         else st0 { seqIndexCounter   = tNext
                                  , headDimCounter    = dNext
                                  , dotAccumulator    = dotAcc'
                                  , attentionScores   = scores'
                                  , maxAttentionScore = maxScore'
                                  , currentAddress    = addressThisCycle
                                  }
                  else st0 { currentAddress = 0 }
              _ -> st0

      -- ACC phase passes
      (denom', isSecondPass', outVec', addrAcc, accDoneNow) =
        case currentPhase st1 of
          PhaseAccumulate | isRunning st1 && not (isSecondPass st1) ->
            let t'      = seqIndexCounter st1
                sVal    = attentionScores st1 !! t'
                addVal  = exp (sVal - maxAttentionScore st1)
                denomN  = expDenominator st1 + addVal
                lastT   = t' == seqIdx
            in if lastT
                 then (denomN, True,  outputVector st1, 0, False)
                 else (denomN, False, outputVector st1, 0, False)
          PhaseAccumulate | isRunning st1 && isSecondPass st1 ->
            let outV   = outputVector st1
                wUn    = if prevDataValid st1 then exp ( (attentionScores st1 !! prevSeqIndex st1) - maxAttentionScore st1) else 0
                w      = if expDenominator st1 /= 0 then wUn / expDenominator st1 else 0
                outAt  = (outV !! prevHeadDimIndex st1) + (if prevDataValid st1 then w * vAligned else 0)
                outV'  = if prevDataValid st1 then replace (prevHeadDimIndex st1) outAt outV else outV
                accDone = prevDataValid st1 && lastHeadDim (prevHeadDimIndex st1) && lastSeqIdx (prevSeqIndex st1)
            in (expDenominator st1, isSecondPass st1, outV', Addressing.computeBankAddress (seqIndexCounter st1) (headDimCounter st1), accDone)
          _ -> (expDenominator st1, isSecondPass st1, outputVector st1, 0, False)

      st2 = case currentPhase st1 of
              PhaseAccumulate | isRunning st1 && not (isSecondPass st1) ->
                let t' = if seqIndexCounter st1 == seqIdx then seqIndexCounter st1 else incSeqIdx (seqIndexCounter st1)
                in if isSecondPass'
                     then st1 { isSecondPass   = True
                              , expDenominator = denom'
                              , seqIndexCounter = 0
                              , headDimCounter  = 0
                              , prevDataValid   = False
                              , currentAddress  = 0
                              }
                     else st1 { expDenominator = denom', seqIndexCounter = t', currentAddress = 0 }
              PhaseAccumulate | isRunning st1 && isSecondPass st1 ->
                if accDoneNow
                  then st1 { currentPhase = PhaseFinalize
                           , outputVector = outVec'
                           , currentAddress = 0
                           }
                  else st1 { outputVector = outVec'
                           , seqIndexCounter = if lastHeadDim (headDimCounter st1) then (if lastSeqIdx (seqIndexCounter st1) then seqIndexCounter st1 else incSeqIdx (seqIndexCounter st1)) else seqIndexCounter st1
                           , headDimCounter  = incHeadDim (headDimCounter st1)
                           , currentAddress  = addrAcc
                           }
              _ -> st1

      -- Final phase
      st3 = case currentPhase st2 of
              PhaseFinalize ->
                st2 { currentPhase = PhaseFinalize
                    , isRunning    = False
                    , donePulse    = True
                    , currentAddress = 0
                    }
              _ -> st2

      -- Update pipeline registers
      st4 = st3
        { prevStartSignal = startNow
        , prevSeqIndex    = seqIndexCounter st3
        , prevHeadDimIndex= headDimCounter st3
        , prevDataValid   = case currentPhase st3 of
                              PhaseDotProduct -> True
                              PhaseAccumulate | isSecondPass st3 -> True
                              _ -> False
        , prevKeyValue    = kRamIn
        , prevValueValue  = vRamIn
        }

      out :: AttentionOutput
      out = ( currentAddress st4
            , outputVector st4
            , isRunning st4
            , donePulse st4
            )
    in (st4, out)

  oSig = mealy step initialState (bundle (startSignal, seqIndexSignal, qSignal, kBypassSignal, vBypassSignal, kRamSignal, vRamSignal))

  (addressOut, outputVecOut, busyOut, doneOut) = unbundle oSig
