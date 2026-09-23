"""One-axis tilt by complementary filter: integrate the gyro, and let the accelerometer
pull the result back so the integral cannot drift.

    angle <- K * (angle + gyro_y * dt) + (1 - K) * accel_x

For small angles accel_x is sin(pitch), close to the pitch in radians, so the filter takes
the short-term angle from the gyro and the long-term angle from gravity.  This is PLAN.md
section 2's program with its range warning fixed: a declared range plus the clamp that makes
it true.

Packet: angle (q30, 4 B), accel_z (int16, 2 B).  The interrupt fires while the tilt is past a
limit the host can change live.
Exercises: a ranged state and the clamp that keeps it, a param, a two-term sum of products (one MAC run on the MPU), a
compare that the MPU lowers to the branch-skip idiom.
"""
import math

import opendmp as dmp

RATE = 200                              # Hz
K = 0.98                                # gyro weight
GYRO_RAD = math.radians(1000)           # 1.0 gyro unit = 1000 dps at +/-2000 dps full scale
GYRO_DT = GYRO_RAD / RATE               # rad per gyro unit per pass: 0.0873


@dmp.program(rate=RATE, gyro_fs=2000, accel_fs=2)
def tilt(s):
    angle: dmp.state(-1.5, 1.5) = 0.0   # radians, kept across passes
    limit: dmp.param = 0.25             # radians, host-adjustable

    # a declared range is an assumption, not code: a long fast spin would carry the integral
    # past it, so saturate explicitly
    angle = dmp.clamp(K * (angle + s.gyro.y * GYRO_DT) + (1 - K) * s.accel.x, -1.5, 1.5)
    dmp.emit(angle, s.accel.z)
    if angle > limit:
        dmp.irq()
