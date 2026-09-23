"""dmp.div and friends: arithmetic the DMP has no instruction for, written in the DMP language.

Like dmp.quat these are compiled from their source, so every engine (the reference, the IR
interpreter, the emulator running the bytes, silicon) computes exactly the same bits.
"""
import opendmp as dmp

LSB = 2.0 ** -30                 # one unit in the last place of a Q30 word
MAX = 2.0 - 2.0 ** -30           # the largest Q30 value
ABOVE_MIN = -2.0 + 2.0 ** -30    # the smallest Q30 value above -2.0


@dmp.sub                         # ICM: one routine when called twice or more; MPU: inlined
def div(x, y):
    """x / y, truncated toward zero and exact to the last bit of Q30, for every x and y.

    Restoring long division on the magnitudes: the integer bit, then 30 fraction bits.  It works
    on -|x| and -|y|, which always fit (|-2.0| would not), and tests each bit as r <= |y|-less-r
    in that form, so no intermediate leaves [-2, 0].  When the quotient's magnitude would reach
    2, and when y is 0, the result saturates at +-(2 - 2^-30)."""
    a = x
    if x > 0.0:
        a = -x
    a = dmp.assume(a, -2.0, 0.0)                      # -|x|
    b = y
    if y > 0.0:
        b = -y
    b = dmp.assume(b, -2.0, 0.0)                      # -|y|
    q = 0.0
    sat = 0
    if b >= -1.0:                                     # |y| <= 1: 2|y| fits, so test |x| >= 2|y|
        if a <= b + b:
            sat = 1                                   # (y == 0 lands here too)
    if sat == 1:
        q = MAX
    else:
        r = a                                         # -remainder, in (b, 0] after the first bit
        if r <= b:                                    # |x| >= |y|: the integer bit
            r = dmp.assume(r - b, ABOVE_MIN, 0.0)
            q = LSB
        r = dmp.assume(r, ABOVE_MIN, 0.0)             # -2.0 only when x == -2.0 and |y| < 2, which
        for _ in dmp.loop(30):                        # the integer bit took; the 30 fraction bits
            u = dmp.assume(b - r, -2.0, 0.0)          # -(|y| - |r|)
            if r <= u:                                # 2|r| >= |y|: the bit is 1
                r = dmp.assume(r - u, ABOVE_MIN, 0.0)     # -(2|r| - |y|)
                q = dmp.assume(q + q + LSB, 0.0, 2.0)
            else:
                r = dmp.assume(r + r, ABOVE_MIN, 0.0)
                q = dmp.assume(q + q, 0.0, 2.0)
    if x < 0.0:
        q = -q
    if y < 0.0:
        q = -q
    return q
