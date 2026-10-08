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


INT_MAX = 2147483647


@dmp.sub                         # ICM: one routine when called twice or more; MPU: inlined
def idiv(n, m):
    """n // m on Int values: floor division, as Python's, exact for every 32-bit n and m.

    Restoring long division, one dividend bit per step from the top (32 steps).  The remainder
    is kept negated, rn = -r in (-|m|, 0], and each bit is tested as rn - bit <= -|m| - rn, so no
    intermediate overflows even for |m| = 2^31.  n // 0 and the one quotient that does not fit,
    -2^31 // -1, saturate at the Int limits, with the sign of the true quotient."""
    d = n
    if n < 0:
        d = -n                       # |n| as a raw 32-bit pattern (-2^31 stays 0x80000000)
    b = m
    if m > 0:
        b = -m                       # -|m|, always an Int
    q = 0
    rn = 0
    if m == 0:
        q = INT_MAX
        if n < 0:
            q = -INT_MAX - 1
    else:
        for _ in dmp.loop(32):
            bit = 0
            if d < 0:                # the dividend's next bit (its top bit)
                bit = 1
            d = d + d
            u = b - rn               # -(|m| - r)
            if rn - bit <= u:        # 2r + bit >= |m|: this quotient bit is 1
                rn = rn - bit - u    # -(2r + bit - |m|)
                q = q + q + 1
            else:
                rn = rn + rn - bit   # -(2r + bit)
                q = q + q
        # q holds |n| // |m| as a raw pattern (2^31 when n = -2^31 and |m| = 1)
        if n < 0:
            if m > 0:
                q = -q               # the quotient is negative: floor it
                if rn < 0:
                    q = q - 1
            else:
                if q < 0:            # -2^31 // -1 = 2^31 does not fit
                    q = INT_MAX
        else:
            if m < 0:
                q = -q
                if rn < 0:
                    q = q - 1
    return q


@dmp.sub                         # ICM: one routine when called twice or more; MPU: inlined
def imul(n, m):
    """n * m on Int values: the exact product modulo 2^32, as Int arithmetic wraps.

    The DMP's multiply keeps the top of a 64-bit product (Q30: (a*b) >> 30), so an integer
    product is shift-and-add: m's 32 bits from the top, doubling the partial product each step
    and adding n where the bit is 1 (two's complement makes this right for negative m too)."""
    r = 0
    d = m
    for _ in dmp.loop(32):
        r = r + r
        if d < 0:                    # m's next bit (the top bit of what is left)
            r = r + n
        d = d + d
    return r
