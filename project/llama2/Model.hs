module Model (topEntity, multiCycleTransformer, initAttentionCache, ProcessingState(..)) where

import Clash.Prelude
import qualified Prelude as P
import qualified GHC.TypeNats
import Data.Functor (($>))
import Helpers
  ( NumQueryHeads, NumKeyValueHeads, NumLayers, SeqLen, HeadDimension, ModelDim, VocabSize
  , Token, TransformerLayerComponent(..), TransformerDecoderComponent (..)
  , MultiHeadAttentionComponent(..), EmbeddingComponent (..)
  , runSingleHeadQKV, applyRotaryToHead, StepCount (..)
  , computeFeedForward, transformerLogits, argMax, embed
  , sampleFromProbs, computeMultiHeadAttention, seqLen, liftA4, liftA5, rmsNorm, softmax
  )
import GHC.Stack (HasCallStack)

-- ============================================================================
-- Multi-Cycle State Machine
-- ============================================================================

data CycleStage =
    Cycle1_ReadCache
  | Cycle2_ComputeQKV
  | Cycle3_ComputeAttn
  | Cycle4_WriteCache
  | Cycle5_ComputeFFN
  deriving (Show, Eq, Enum, Bounded)

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
  { psStage      :: CycleStage
  , psLayer      :: Index NumLayers
  , psSeqPos     :: Index SeqLen
  } deriving (Show, Generic, NFDataX)

initialProcessingState :: ProcessingState
initialProcessingState = ProcessingState
  { psStage = Cycle1_ReadCache
  , psLayer = 0
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
      then st { psStage = Cycle1_ReadCache
              , psLayer = 0
              , psSeqPos = succ (psSeqPos st)  -- wraps as Index
              }
      else st { psStage = Cycle1_ReadCache
              , psLayer = succ (psLayer st)
              }

-- ============================================================================
-- Intermediate data storage
-- ============================================================================

data IntermediateData = IntermediateData
  { idInputVec    :: Vec ModelDim Float
  , idQueries     :: Vec NumQueryHeads (Vec HeadDimension Float)
  , idKeys        :: Vec NumKeyValueHeads (Vec HeadDimension Float)
  , idValues      :: Vec NumKeyValueHeads (Vec HeadDimension Float)
  , idCachedKeys  :: Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float))
  , idCachedVals  :: Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float))
  , idAttnOutput  :: Vec ModelDim Float
  , idFFNOutput   :: Vec ModelDim Float
  } deriving (Show, Generic, NFDataX)

initialIntermediateData :: IntermediateData
initialIntermediateData = IntermediateData
  { idInputVec   = repeat 0
  , idQueries    = repeat (repeat 0)
  , idKeys       = repeat (repeat 0)
  , idValues     = repeat (repeat 0)
  , idCachedKeys = repeat (repeat (repeat 0))
  , idCachedVals = repeat (repeat (repeat 0))
  , idAttnOutput = repeat 0
  , idFFNOutput  = repeat 0
  }

-- ============================================================================
-- Memory Interface
-- ============================================================================

type CacheDepth = NumLayers GHC.TypeNats.* NumKeyValueHeads GHC.TypeNats.* SeqLen GHC.TypeNats.* HeadDimension
type CacheAddr  = Index CacheDepth

cacheAddr :: Index NumLayers -> Index NumKeyValueHeads -> Index SeqLen -> Index HeadDimension -> CacheAddr
cacheAddr l h s d =
  let headSize  = natToNum @HeadDimension
      seqSize   = natToNum @SeqLen * headSize
      layerSize = natToNum @NumKeyValueHeads * seqSize
      addr = fromIntegral d + fromIntegral s * headSize + fromIntegral h * seqSize + fromIntegral l * layerSize
  in addr

data AttentionCache dom = AttentionCache
  { keyCache   :: Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float
  , valueCache :: Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float
  }

initAttentionCache :: forall dom. HiddenClockResetEnable dom => AttentionCache dom
initAttentionCache = AttentionCache
  { keyCache = blockRam (replicate (SNat @CacheDepth) 0)
  , valueCache = blockRam (replicate (SNat @CacheDepth) 0)
  }

-- ============================================================================
-- Cache Access (sequential, single-element per cycle)
-- ============================================================================


-- Read one element/cycle with explicit start, pos, busy, done.
-- Keeps your local buffer for now (item 4 will remove it).
readCachedSequenceWithGated
  :: forall dom
   . HiddenClockResetEnable dom
  => (Signal dom CacheAddr -> Signal dom (Maybe (CacheAddr, Float)) -> Signal dom Float) -- BRAM port
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom Bool                 -- start (asserted when stage enters)
  -> Signal dom (Index SeqLen)       -- pos (causal horizon, inclusive)
  -> ( Signal dom (Vec SeqLen (Vec HeadDimension Float))  -- buffer
     , Signal dom Bool               -- busy
     , Signal dom Bool               -- done (1-cycle pulse)
     )
readCachedSequenceWithGated ram l h startSig posSig = (bufSig, activeSig, doneSig)
 where
  -- rising edge detect
  startPrev  = register False startSig
  startPulse = liftA2 (\now prev -> now && not prev) startSig startPrev

  -- active bit: set on startPulse, clear on done
  activeSig :: Signal dom Bool
  activeSig = s where
    s = register False next
    next = liftA3 (\a sp dn -> (a && not dn) || sp) s startPulse doneSig

  -- 2D counters (t,d), enabled only when active
  tCnt :: Signal dom (Index SeqLen)
  dCnt :: Signal dom (Index HeadDimension)

  -- increment when active, reset on startPulse
  tCnt = regEn 0 (startPulse .||. en) tNext
  dCnt = regEn 0 (startPulse .||. en) dNext

  en = activeSig

  -- stop at (t==pos, d==max) but still issue that last address;
  -- tNext/dNext compute the next indices while active
  atLastElem = liftA3 (\t d p -> t == p && d == maxBound) tCnt dCnt posSig

  dNext = mux en
            (liftA2 (\d lastD -> if lastD then 0 else succ d)
                    dCnt
                    ((== maxBound) <$> dCnt))
            dCnt
  tNext = mux en
            (liftA3 (\t d p ->
               if d == maxBound
                  then if t == p then 0 else succ t
                  else t)
             tCnt dCnt posSig)
            tCnt

  -- addresses and read
  addrSig = cacheAddr l h <$> tCnt <*> dCnt
  qSig    = ram addrSig (pure Nothing)

  -- capture previous (t,d) for writeback due to 1-cycle RAM read latency
  tPrev = register 0 tCnt
  dPrev = register 0 dCnt

  -- done pulses one cycle after we issued the last address while active
  lastIssued = (&&) <$> activeSig <*> atLastElem
  doneSig    = register False lastIssued

  -- local buffer reset on startPulse
  bufInit = repeat (repeat 0) :: Vec SeqLen (Vec HeadDimension Float)

  bufSig :: Signal dom (Vec SeqLen (Vec HeadDimension Float))
  bufSig = b where
    b0 = mux startPulse (pure bufInit) b -- clear on start
    b  = register bufInit (writeBack <$> b0 <*> tPrev <*> dPrev <*> qSig <*> activeSig)
    writeBack buf t d q a =
      if a then replace t (replace d q (buf !! t)) buf else buf

-- Wrappers for keys/vals
readCachedKeysGated
  :: HiddenClockResetEnable dom
  => AttentionCache dom
  -> Index NumLayers -> Index NumKeyValueHeads
  -> Signal dom Bool
  -> Signal dom (Index SeqLen)
  -> ( Signal dom (Vec SeqLen (Vec HeadDimension Float))
     , Signal dom Bool, Signal dom Bool)
readCachedKeysGated cache = readCachedSequenceWithGated (keyCache cache)

readCachedValsGated
  :: HiddenClockResetEnable dom
  => AttentionCache dom
  -> Index NumLayers -> Index NumKeyValueHeads
  -> Signal dom Bool
  -> Signal dom (Index SeqLen)
  -> ( Signal dom (Vec SeqLen (Vec HeadDimension Float))
     , Signal dom Bool, Signal dom Bool)
readCachedValsGated cache = readCachedSequenceWithGated (valueCache cache)

-- Sequential writer: one element of K and one element of V per cycle.
-- Enable is asserted for the whole stage; done pulses on last element.
writeToCacheSequence
  :: forall dom
   . HiddenClockResetEnable dom
  => AttentionCache dom
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom (Index SeqLen)                                   -- ^ sequence position
  -> Signal dom (Vec HeadDimension Float, Vec HeadDimension Float) -- ^ current (K,V) vectors
  -> Signal dom Bool                                             -- ^ enable (high across stage)
  -> Signal dom Bool                                             -- ^ done (1-cycle pulse)
writeToCacheSequence cache l h seqPosSig kvSig enSig =
  doneSig
 where
  keyRam   = keyCache cache
  valueRam = valueCache cache

  -- dimension counter; increments only when enabled, resets to 0 otherwise
  dCnt :: Signal dom (Index HeadDimension)
  dCnt = register 0 nextD
  nextD = mux enSig
              (P.fmap (\d -> if d == maxBound then 0 else succ d) dCnt)
              (pure 0)

  -- last element this cycle?
  lastElemSig :: Signal dom Bool
  lastElemSig = (== maxBound) <$> dCnt

  -- done when enabled and writing the last element
  doneSig :: Signal dom Bool
  doneSig = (&&) <$> enSig <*> lastElemSig

  -- select the K,V element for current dimension
  kElemSig :: Signal dom Float
  kElemSig = (\(k,_ ) d -> k !! d) <$> kvSig <*> dCnt

  vElemSig :: Signal dom Float
  vElemSig = (\(_ ,v) d -> v !! d) <$> kvSig <*> dCnt

  -- compute linear addresses
  kAddrSig = cacheAddr l h <$> seqPosSig <*> dCnt
  vAddrSig = kAddrSig  -- same addressing expression for value cache

  -- build write enables
  kWriteSig :: Signal dom (Maybe (CacheAddr, Float))
  kWriteSig = mux enSig (Just <$> bundle (kAddrSig, kElemSig)) (pure Nothing)

  vWriteSig :: Signal dom (Maybe (CacheAddr, Float))
  vWriteSig = mux enSig (Just <$> bundle (vAddrSig, vElemSig)) (pure Nothing)

  -- perform writes; tie off reads
  _kQ = keyRam   kAddrSig kWriteSig
  _vQ = valueRam vAddrSig vWriteSig

-- Micro-FSM for Cycle3: iterate t = 0..pos (one per cycle).
-- done pulses one cycle after the last t was issued.
cycle3AttnMicro
  :: HiddenClockResetEnable dom
  => Signal dom Bool                 -- start
  -> Signal dom (Index SeqLen)       -- pos
  -> ( Signal dom (Index SeqLen)     -- t counter (valid while busy)
     , Signal dom Bool               -- busy
     , Signal dom Bool               -- done pulse (1-cycle)
     )
cycle3AttnMicro startSig posSig = (tCnt, busySig, doneSig)
 where
  startPrev  = register False startSig
  startPulse = liftA2 (\now prev -> now && not prev) startSig startPrev
  
  busySig = sig where
    sig = register False next
    next = liftA3 (\b sp dn -> (b && not dn) || sp) sig startPulse doneSig

  tCnt = regEn 0 (startPulse .||. busySig) tNext

  atLast = (==) <$> tCnt <*> posSig

  tNext = mux busySig
             (liftA2 (\t lastT -> if lastT then 0 else succ t) tCnt atLast)
             tCnt

  issuedLast = (&&) <$> busySig <*> atLast
  doneSig    = register False issuedLast

-- ============================================================================
-- Multi-Cycle Transformer Layer
-- ============================================================================

multiCycleTransformerLayer
  :: forall dom
   . (HiddenClockResetEnable dom)
  => TransformerLayerComponent
  -> AttentionCache dom
  -> Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom IntermediateData
  -> (Signal dom IntermediateData
     , Signal dom Bool     -- writeDone (Cycle4)
     , Signal dom Bool     -- readDone  (Cycle1)
     , Signal dom Bool)    -- attnDone  (Cycle3)
multiCycleTransformerLayer layer cache layerIdx stateSig dataSig =
  (nextDataSig, writeDoneThisLayerSig, readDoneThisLayerSig, attnDoneThisLayerSig)
 where
  mha = multiHeadAttention layer
  ffn = feedforwardNetwork layer

  -- Stage enables (only when this layer is active)
  stgEq s = liftA2 (\st _ -> psStage st == s && psLayer st == layerIdx) stateSig (pure ())
  rdStart  = stgEq Cycle1_ReadCache
  qkvStart = stgEq Cycle2_ComputeQKV -- unchanged comb for now
  attStart = stgEq Cycle3_ComputeAttn
  wrStart  = stgEq Cycle4_WriteCache
  ffnStart = stgEq Cycle5_ComputeFFN -- unchanged comb for now

  posSig = psSeqPos <$> stateSig

  -- Gated reads for all KV heads (still maps over heads; item 3 will refactor to a single RAM owner)
  keysBufPerHead :: Vec NumKeyValueHeads (Signal dom (Vec SeqLen (Vec HeadDimension Float)))
  keysBusyPerHead, keysDonePerHead :: Vec NumKeyValueHeads (Signal dom Bool)

  (keysBufPerHead, keysBusyPerHead, keysDonePerHead) =
    unzip3 $ map (\h -> readCachedKeysGated cache layerIdx h rdStart posSig) indicesI

  valsBufPerHead :: Vec NumKeyValueHeads (Signal dom (Vec SeqLen (Vec HeadDimension Float)))
  valsBusyPerHead, valsDonePerHead :: Vec NumKeyValueHeads (Signal dom Bool)

  (valsBufPerHead, valsBusyPerHead, valsDonePerHead) =
    unzip3 $ map (\h -> readCachedValsGated cache layerIdx h rdStart posSig) indicesI

  -- per-head done (both K and V finished for that head)
  perHeadDone :: Vec NumKeyValueHeads (Signal dom Bool)
  perHeadDone = zipWith (liftA2 (&&)) valsDonePerHead keysDonePerHead

  -- layer read-done when all heads are done
  readDoneThisLayerSig :: Signal dom Bool
  readDoneThisLayerSig = fmap and (sequenceA perHeadDone)

  -- Bundle buffers back into Signals for processCycle (temporary, until streaming attention)
  cachedKeysSig :: Signal dom (Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float)))
  cachedKeysSig = sequenceA keysBufPerHead

  cachedValsSig :: Signal dom (Vec NumKeyValueHeads (Vec SeqLen (Vec HeadDimension Float)))
  cachedValsSig = sequenceA valsBufPerHead

  -- Attention micro-FSM (control only for now)
  (_tCnt, attBusy, attDone) = cycle3AttnMicro attStart posSig
  attnDoneThisLayerSig = attDone

  -- Writes (unchanged)
  writeEnableSig = wrStart
  kvAllHeadsSig  = liftA2 (zipWith (,)) (idKeys <$> dataSig) (idValues <$> dataSig)

  writeDonePerHead =
    map (\hIdx ->
           writeToCacheSequence cache layerIdx hIdx
             posSig
             (((!!) <$> kvAllHeadsSig) <*> pure hIdx)
             writeEnableSig)
        indicesI

  writeDoneAllHeads = fmap and (sequenceA writeDonePerHead)
  writeDoneThisLayerSig =
    (\st doneH -> (psStage st == Cycle4_WriteCache && psLayer st == layerIdx) && doneH)
      <$> stateSig <*> writeDoneAllHeads

  -- Compute body (unchanged semantics; just moved under stage guards)
  baseNextDataSig = liftA4 processCycle stateSig dataSig cachedKeysSig cachedValsSig
  nextDataSig     = baseNextDataSig

  processCycle st idata keysNow valsNow
    | psLayer st /= layerIdx = idata
    | otherwise =
        case psStage st of
          Cycle1_ReadCache -> idata
                                { idCachedKeys = keysNow
                                , idCachedVals = valsNow
                                }
          Cycle2_ComputeQKV ->
            let
              -- pre-norm input for attention
              xHat = rmsNorm (idInputVec idata) (rmsAtt mha)

              -- queries for all NumQueryHeads from x̂, with RoPE
              queries = imap (\hIdx _ ->
                let headComp = heads mha !! hIdx
                    (q, _, _) = runSingleHeadQKV headComp xHat
                    (q_rot, _) = applyRotaryToHead headComp
                                    (StepCount $ fromIntegral $ psSeqPos st)
                                    (q, repeat 0)
                in q_rot
                ) indicesI

              -- keys/values for NumKeyValueHeads from x̂, with RoPE on K
              headsPerGroupI :: Int
              headsPerGroupI = natToNum @NumQueryHeads `P.div` natToNum @NumKeyValueHeads

              toIndexClamped :: forall n. KnownNat n => Int -> Index n
              toIndexClamped i =
                let hi = natToNum @n - 1
                    j  = max 0 (min hi i)
                in toEnum j

              keysAndValues =
                imap (\hIdx _ ->
                  let qIdx :: Index NumQueryHeads
                      qIdx = toIndexClamped @NumQueryHeads (fromEnum hIdx * headsPerGroupI)
                      headComp = heads mha !! qIdx
                      (_, k, v) = runSingleHeadQKV headComp xHat
                      (_, k_rot) = applyRotaryToHead headComp
                                    (StepCount $ fromIntegral $ psSeqPos st)
                                    (repeat 0, k)
                  in (k_rot, v)
                ) indicesI

              (keys, values) = unzip keysAndValues
            in idata { idQueries = queries, idKeys = keys, idValues = values }

          Cycle3_ComputeAttn ->
            let posIdx = psSeqPos st
                curK   = idKeys idata
                curV   = idValues idata
                keysForAttn = imap (\h kCache -> replace posIdx (curK !! h) kCache) (idCachedKeys idata)
                valsForAttn = imap (\h vCache -> replace posIdx (curV !! h) vCache) (idCachedVals idata)
                attnOut = computeMultiHeadAttention
                            mha
                            (idInputVec idata)
                            (idQueries idata)
                            keysForAttn
                            valsForAttn
                            posIdx
            in idata { idAttnOutput = attnOut }
          Cycle4_WriteCache -> idata
          Cycle5_ComputeFFN ->
            let ffnOut = computeFeedForward ffn (idAttnOutput idata)
            in idata { idFFNOutput = ffnOut }

-- ============================================================================
-- Multi-Cycle Full Transformer (advance on stageDone)
-- ============================================================================

-- xorshift32 core (synthesizable, 1-cycle combinational)
xorshift32 :: Unsigned 32 -> Unsigned 32
xorshift32 s0 =
  let s1 = s0 `xor` shiftL s0 13
      s2 = s1 `xor` shiftR s1 17
      s3 = s2 `xor` shiftL s2 5
  in s3

multiCycleTransformer
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> Vec NumLayers (AttentionCache dom)
  -> Signal dom Token
  -> Signal dom Float
  -> Signal dom Int
  -> (Signal dom Token, Signal dom Bool)
multiCycleTransformer decoder caches tokenSig temperatureSig seedSig =
  (outputTokenSig, readyPulseSig)
 where
  embedding    = modelEmbedding decoder
  layers       = modelLayers decoder
  inputVecSig  = embed (vocabulary embedding) <$> tokenSig

  -- Global controller: advance only when current stageDone is asserted
  advance st done = if done then nextState st else st
  procStateSig = register initialProcessingState (advance <$> procStateSig <*> stageDoneSig)

  -- Intermediate data register
  currentDataSig = register initialIntermediateData nextDataSig

  -- Load input at token start
  inputLoadedSig = liftA3 (\st cur inp ->
                      if psStage st == Cycle1_ReadCache && psLayer st == 0
                        then cur { idInputVec = inp } else cur)
                    procStateSig currentDataSig inputVecSig

  -- Run layers; gather per-stage done pulses
  foldStep
    :: ( Signal dom IntermediateData
      , Vec NumLayers (Signal dom Bool)  -- write dones
      , Vec NumLayers (Signal dom Bool)  -- read  dones
      , Vec NumLayers (Signal dom Bool)) -- attn  dones
    -> (TransformerLayerComponent, AttentionCache dom, Index NumLayers)
    -> ( Signal dom IntermediateData
      , Vec NumLayers (Signal dom Bool)
      , Vec NumLayers (Signal dom Bool)
      , Vec NumLayers (Signal dom Bool))
  foldStep (dSig, wD, rD, aD) (layer, cache, lidx) =
    let (newD, wDoneL, rDoneL, aDoneL) =
          multiCycleTransformerLayer layer cache lidx procStateSig dSig
        dSel = liftA3 (\st old new -> if psLayer st == lidx then new else old) procStateSig dSig newD
    in ( dSel
      , replace lidx wDoneL wD
      , replace lidx rDoneL rD
      , replace lidx aDoneL aD)

  (nextDataSig, writeDoneVec, readDoneVec, attnDoneVec) =
    foldl foldStep (inputLoadedSig, repeat (pure False), repeat (pure False), repeat (pure False))
                  (zip3 layers caches indicesI)

  writeDoneAny = fmap or (sequenceA writeDoneVec)
  readDoneAny  = fmap or (sequenceA readDoneVec)
  attnDoneAny  = fmap or (sequenceA attnDoneVec)

  -- Stage done selection
  stgSig = psStage <$> procStateSig
  is c   = (== c) <$> stgSig

  stageDoneSig =
    mux (is Cycle1_ReadCache)   readDoneAny  $
    mux (is Cycle2_ComputeQKV)  (pure True)  $
    mux (is Cycle3_ComputeAttn) attnDoneAny  $
    mux (is Cycle4_WriteCache)  writeDoneAny $
    mux (is Cycle5_ComputeFFN)  (pure True)  $
    pure False

  -- Ready pulse at last layer FFN completion (rising edge)
  isLastFFN  = liftA2 (\st _ -> psStage st == Cycle5_ComputeFFN && psLayer st == maxBound)
                      procStateSig (pure ())
  readyPulseSig = liftA2 (\now prev -> now && not prev) isLastFFN (register False isLastFFN)

  -- existing logits
  logitsNow      = transformerLogits decoder . idFFNOutput <$> nextDataSig
  latchedLogits  = regEn (repeat 0) readyPulseSig logitsNow

  -- PRNG state: seed is mixed in on the first pulse; otherwise advance each pulse
  firstPulseSig :: Signal dom Bool
  firstPulseSig = regEn True readyPulseSig (pure False)

  mixedSeedSig :: Signal dom (Unsigned 32)
  mixedSeedSig = fromIntegral . (`xor` 0x9E3779B9) <$> seedSig

  prngStateSig :: Signal dom (Unsigned 32)
  prngStateSig =
    let nextVal = mux firstPulseSig (xorshift32 <$> mixedSeedSig)
                                    (xorshift32 <$> prngStateSig)
    in regEn 2463534242 readyPulseSig nextVal  -- non-zero default

  -- Convert to Float in [0,1). Use top 24 bits as mantissa fraction.
  uniform01Sig :: Signal dom Float
  uniform01Sig = (/ 16777216.0) . fromIntegral . (`shiftR` 8) <$> prngStateSig

  -- Sampling on pulse:
  sampledTokenOnPulse :: Signal dom (Unsigned 32)
  sampledTokenOnPulse =
    liftA3
      (\temp logs u ->
         if temp <= 0.0
           then argMax logs
           else
             let probs = softmax temp logs
             in sampleFromProbs u probs
      )
      temperatureSig latchedLogits uniform01Sig

  outputTokenSig = regEn 0 readyPulseSig sampledTokenOnPulse

-- ============================================================================
-- Top Entity
-- ============================================================================

topEntity
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerDecoderComponent
  -> Signal dom Int
  -> Signal dom Token
  -> Signal dom Float
  -> Signal dom Int
  -> (Signal dom Token, Signal dom Bool)
topEntity decoder _ = multiCycleTransformer decoder (repeat initAttentionCache)
