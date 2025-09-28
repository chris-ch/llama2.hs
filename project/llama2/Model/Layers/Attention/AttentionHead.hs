module Model.Layers.Attention.AttentionHead (
  streamHeadAttentionAddrIO
) where

import Clash.Prelude
import qualified Model.Memory.Addressing as Addressing
import Model.Core.Types (HeadDimension, SeqLen, BankAddress)

-- Internal state for one-pass streaming attention over a single head
data OnePassState = OnePassState
  { running      :: Bool
  , prevStart    :: Bool
  , pos          :: Index SeqLen
  , qVec         :: Vec HeadDimension Float
  , kByp         :: Vec HeadDimension Float
  , vByp         :: Vec HeadDimension Float
  , tCtr         :: Index SeqLen
  , iCtr         :: Index HeadDimension
  , prevT        :: Index SeqLen
  , prevI        :: Index HeadDimension
  , prevValid    :: Bool
  , prevK        :: Float
  , prevV        :: Float
  , dotAcc       :: Float
  , maxScore     :: Float
  , denom        :: Float
  , outVec       :: Vec HeadDimension Float
  , vBuf         :: Vec HeadDimension Float
  , curAddr      :: BankAddress
  , donePulse    :: Bool
  }
  deriving (Generic, NFDataX)

type AttentionInput =
  ( Bool                           -- start (true during Stage3_Attend)
  , Index SeqLen                   -- pos (current sequence index)
  , Vec HeadDimension Float        -- query vector Q
  , Vec HeadDimension Float        -- K(pos) bypass vector
  , Vec HeadDimension Float        -- V(pos) bypass vector
  , Float                          -- key RAM read data (aligned)
  , Float                          -- value RAM read data (aligned)
  )

type AttentionOutput =
  ( BankAddress
  , Vec HeadDimension Float        -- attended head output vector
  , Bool                           -- busy
  , Bool                           -- done (1-cycle pulse)
  )

{-# INLINE selectBypass #-}
selectBypass
  :: Index SeqLen -> Index SeqLen -> Index HeadDimension
  -> Vec HeadDimension Float -> Float -> Float
selectBypass pos t i bypassVec ramOut =
  if t == pos then bypassVec !! i else ramOut

streamHeadAttentionAddrIO
  :: HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom (Index SeqLen)
  -> Signal dom (Vec HeadDimension Float)
  -> Signal dom (Vec HeadDimension Float)
  -> Signal dom (Vec HeadDimension Float)
  -> Signal dom Float
  -> Signal dom Float
  -> ( Signal dom BankAddress
     , Signal dom (Vec HeadDimension Float)
     , Signal dom Bool
     , Signal dom Bool )
streamHeadAttentionAddrIO startSig posSig qSig kBypSig vBypSig kRamSig vRamSig =
  unbundle oSig
 where
  scale :: Float
  scale = 1.0 / sqrt (natToNum @HeadDimension :: Float)

  negBig :: Float
  negBig = -1.0e30

  initialState :: OnePassState
  initialState = OnePassState
    { running   = False, prevStart = False
    , pos       = 0, qVec = repeat 0, kByp = repeat 0, vByp = repeat 0
    , tCtr      = 0, iCtr = 0, prevT = 0, prevI = 0, prevValid = False
    , prevK     = 0, prevV = 0
    , dotAcc    = 0, maxScore = negBig, denom = 0
    , outVec    = repeat 0, vBuf = repeat 0
    , curAddr   = 0, donePulse = False
    }

  step :: OnePassState -> AttentionInput -> (OnePassState, AttentionOutput)
  step st (startNow, posIn, qIn, kBypIn, vBypIn, kRamIn, vRamIn) =
    let
      startRise = startNow && not (prevStart st)

      -- Start: latch Q/K(pos)/V(pos), reset accumulators and counters
      st0 = if startRise
              then initialState
                    { running = True, pos = posIn
                    , qVec = qIn, kByp = kBypIn, vByp = vBypIn
                    }
              else st { donePulse = False }

      run = running st0

      -- Issue address for current (tCtr,iCtr) while running
      addrThis = if run
                   then Addressing.computeBankAddress (tCtr st0) (iCtr st0)
                   else 0

      -- Align RAM outputs (previous address), apply bypass for t==pos
      kAligned = if prevValid st0
                   then selectBypass (pos st0) (prevT st0) (prevI st0) (kByp st0) (prevK st0)
                   else 0
      vAligned = if prevValid st0
                   then selectBypass (pos st0) (prevT st0) (prevI st0) (vByp st0) (prevV st0)
                   else 0
      qAligned = if prevValid st0 then (qVec st0 !! prevI st0) else 0

      -- Accumulate dot(q,k_t) and capture V(t,i) into a row buffer
      dotAcc' = if run && prevValid st0 then dotAcc st0 + qAligned * kAligned else dotAcc st0
      vBuf'   = if run && prevValid st0 then replace (prevI st0) vAligned (vBuf st0) else vBuf st0

      endOfPrevRow = prevValid st0 && prevI st0 == maxBound
      lastTPrev    = prevT st0 == pos st0

      -- When a full row t finishes, update online softmax:
      -- Keep track of running max m, denom d, and output outVec.
      -- If new score s <= m:   d' = d + exp(s-m);   out' = out + exp(s-m) * V(t)
      -- If new score s  > m:   d' = 1 + d*exp(m-s); out' = V(t) + out*exp(m-s); m' = s
      sNow       = dotAcc' * scale
      m0         = maxScore st0
      d0         = denom st0
      out0       = outVec st0
      (m1, d1, out1) =
        if run && endOfPrevRow then
          if sNow <= m0
            then
              let a = exp (sNow - m0)
              in (m0, d0 + a, zipWith (+) out0 (map (* a) vBuf'))
            else
              let a = exp (m0 - sNow)
              in (sNow, 1 + d0 * a, zipWith (+) (map (* a) out0) vBuf')
        else (m0, d0, out0)

      -- Advance (t,i) counters
      iNext = if run then (if iCtr st0 == maxBound then 0 else succ (iCtr st0)) else iCtr st0
      tNext = if run
                then if iCtr st0 == maxBound
                       then (if tCtr st0 == pos st0 then tCtr st0 else succ (tCtr st0))
                       else tCtr st0
                else tCtr st0

      -- Clear accumulators at row end; clear vBuf after we folded it in
      dotNext = if endOfPrevRow then 0 else dotAcc'
      vBufNext = if endOfPrevRow then repeat 0 else vBuf'

      -- Done when we just finished row t == pos
      doneNow = run && endOfPrevRow && lastTPrev

      st1 = st0 { curAddr   = addrThis
                , dotAcc    = dotNext
                , maxScore  = m1
                , denom     = d1
                , outVec    = out1
                , vBuf      = vBufNext
                , tCtr      = tNext
                , iCtr      = iNext
                , donePulse = doneNow
                , running   = if doneNow then False else run
                }

      -- Register previous address and RAM outputs for next cycle
      st2 = st1 { prevStart = startNow
                , prevT     = tCtr st1
                , prevI     = iCtr st1
                , prevValid = running st1
                , prevK     = kRamIn
                , prevV     = vRamIn
                , curAddr   = if running st1 then addrThis else 0
                }

      out :: AttentionOutput
      out = ( curAddr st2, outVec st2, running st2, donePulse st2 )
    in (st2, out)

  oSig = mealy step initialState (bundle (startSig, posSig, qSig, kBypSig, vBypSig, kRamSig, vRamSig))