"""Orientation from the gyroscope alone: quaternion strapdown integration, renormalized every
pass without a square root.

    q <- q + q (x) (0, h)            h = w * dt / 2: first-order integration
    q <- q * (1.5 - |q|^2 / 2)       one Newton step towards |q| = 1

The Newton step is the vendor firmware's own (manual section 4.9).  It needs no divide and no
square root, and one step a pass is enough because |q| never strays far from 1.  Each
component of the product is a four-term dot product: one multiply-accumulate run on the MPU.
The result drifts with the gyro bias; mahony.py adds the accelerometer correction.

Packet: w, x, y, z (q30), 16 B.
Exercises: a quaternion in a state list, helpers that return lists, tuple unpacking,
dmp.assume (a range the compiler cannot derive for itself).
"""
import math

import opendmp as dmp

RATE = 200
H = 0.5 * math.radians(1000) / RATE     # gyro unit (1000 dps) -> half-angle per pass: 0.0436


def rate_step(q, hx, hy, hz):
    """q + q (x) (0, hx, hy, hz)"""
    w, x, y, z = q
    return [w - x * hx - y * hy - z * hz,
            x + w * hx + y * hz - z * hy,
            y + w * hy - x * hz + z * hx,
            z + w * hz + x * hy - y * hx]


def normalize(q):
    # |q|^2 stays within a hair of 1; interval arithmetic alone allows 0..4 and would warn
    n = dmp.assume(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], 0.9, 1.1)
    k = 1.5 - 0.5 * n
    # |q[i]| <= sqrt(n) and sqrt(n) * (1.5 - n/2) peaks at exactly 1 (n = 1): unit components
    return [dmp.assume(q[0] * k, -1.0, 1.0), dmp.assume(q[1] * k, -1.0, 1.0),
            dmp.assume(q[2] * k, -1.0, 1.0), dmp.assume(q[3] * k, -1.0, 1.0)]


@dmp.program(rate=RATE)
def gyro_quat(s):
    q: dmp.state(-1.0, 1.0) = [1.0, 0.0, 0.0, 0.0]      # w x y z, starting level
    q = normalize(rate_step(q, s.gyro.x * H, s.gyro.y * H, s.gyro.z * H))
    dmp.emit(w=q[0], x=q[1], y=q[2], z=q[3])
