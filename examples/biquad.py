"""Second-order low-pass (biquad) on all three accelerometer axes, designed at compile time.

The coefficients are computed in ordinary Python from CUTOFF and RATE (RBJ cookbook,
bilinear transform), so a different cutoff is a one-line change.  Direct Form I:

    y = B0*x + B1*x1 + B2*x2 - A1*y1 - A2*y2

is a five-term dot product per axis: one multiply-accumulate run on the MPU.  The input is
halved first because a Butterworth low-pass overshoots a full-scale step, and the output
would wrap at 2 g.

Packet: the filtered x, y, z in units of g/2 (q30), 12 B.
Exercises: compile-time filter design, a helper that updates a state list in place (it is
inlined, so every access resolves to a fixed address), a nested state list, `for` over axes.
"""
import math

import opendmp as dmp

RATE = 200
CUTOFF = 5.0                    # Hz
Q = 1 / math.sqrt(2)            # Butterworth

_w0 = 2 * math.pi * CUTOFF / RATE
_alpha = math.sin(_w0) / (2 * Q)
_a0 = 1 + _alpha
B0 = (1 - math.cos(_w0)) / 2 / _a0         #  0.0055
B1 = 2 * B0                                #  0.0111
B2 = B0
A1 = -2 * math.cos(_w0) / _a0              # -1.7786: inside [-2, 2), as a stable A1 always is
A2 = (1 - _alpha) / _a0                    #  0.8008


def biquad(x, z):
    """One step of the filter; z = [x1, x2, y1, y2] is this axis's history."""
    y = B0 * x + B1 * z[0] + B2 * z[1] - A1 * z[2] - A2 * z[3]
    z[1] = z[0]
    z[0] = x
    z[3] = z[2]
    z[2] = y
    return y


@dmp.program(rate=RATE)
def lowpass(s):
    z: dmp.state = [[0.0, 0.0, 0.0, 0.0] for _ in range(3)]   # per axis: x1 x2 y1 y2
    y = [0.0, 0.0, 0.0]
    for i in range(3):
        y[i] = biquad(s.accel[i] * 0.5, z[i])
    dmp.emit(x=y[0], y=y[1], z=y[2])
