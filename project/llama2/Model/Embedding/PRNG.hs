module Model.Embedding.PRNG (
    xorshift32
) where

import Clash.Prelude ( Bits(xor, shiftR, shiftL), Unsigned )

-- xorshift32 core (synthesizable, 1-cycle combinational)
xorshift32 :: Unsigned 32 -> Unsigned 32
xorshift32 s0 =
  let s1 = s0 `xor` shiftL s0 13
      s2 = s1 `xor` shiftR s1 17
      s3 = s2 `xor` shiftL s2 5
  in s3
