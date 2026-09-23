"""Tone detector (Goertzel): the power in one frequency bin, recomputed every N samples.

    s0 = x + C*s1 - s2          C = 2 cos(2 pi F / RATE): one resonator step per pass
    every N passes:  P = s1^2 + s2^2 - C*s1*s2, then the resonator restarts

Goertzel is one DFT bin for the price of one multiply-accumulate per sample.  It is the
standard way to watch for a known frequency (a motor's imbalance, a structural resonance)
without an FFT.  Here: 10 Hz, in blocks of 40 samples at 200 Hz (bin k = 2 exactly).

In Q30, scaling is the whole game.  On-bin, the resonator grows to about N / (2 sin w) = 65
times the input amplitude.  So the input is scaled by 1/256: the worst resonant input the
sensor can deliver (|x - mean| up to 2 g) keeps the states below 0.64.  At 1/128 that input
would reach 1.28 and break the states' declared range.  Interval arithmetic cannot bound a
resonator, so the step says so with dmp.assume.  The power is computed from halved states.

Packet: the last block's power / 4 (q30), 4 B.
Exercises: a resonator (an IIR coefficient close to 2), a block counter with a periodic reset,
DC removal, a three-factor product.
"""
import math

import opendmp as dmp

RATE = 200
N = 40                                  # block length
F = 10.0                                # Hz
C = 2 * math.cos(2 * math.pi * F / RATE)   # 1.902
SCALE = 1 / 256
DC = 1 / 64                             # DC-blocker weight


@dmp.program(rate=RATE)
def goertzel(s):
    mean: dmp.state = 1.0
    s1: dmp.state(-1.0, 1.0) = 0.0
    s2: dmp.state(-1.0, 1.0) = 0.0
    n: dmp.state = 0
    power: dmp.state = 0.0

    x = s.accel.z
    mean = (1 - DC) * mean + DC * x
    s0 = dmp.assume(SCALE * x - SCALE * mean + C * s1 - s2, -1.0, 1.0)
    s2 = s1
    s1 = s0
    n += 1
    if n == N:
        h1 = s1 * 0.5
        h2 = s2 * 0.5
        power = h1 * h1 + h2 * h2 - (C * h1) * h2
        s1 = 0.0
        s2 = 0.0
        n = 0
    dmp.emit(power=power)
