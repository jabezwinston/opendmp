"""Exact unit vectors without divide or square root: the gravity direction and the spin axis,
and the cosine of the angle between them (is the device spinning about the vertical?).

u = v / |v| is computed as v * rsqrt(|v|^2):
  1. range reduction: double v until |v|^2 lies in [1/4, 1)           a loop, <= 14 trips
  2. Newton for h = 0.5/sqrt(m):  h <- h * (1.5 - 2*m*h^2)            a loop, <= 6 trips,
                                                                    left once converged
  3. u = 2 * v * h
Unlike gyro_quat.py's single Newton step, this works for any |v|, including free fall and
shocks.  The spin axis is only meaningful while the rotation is well above the gyro noise.

The two targets compile this very differently, which the listings show.  On the ICM-20948 the
loops are real backward jumps and unit() is one subroutine called twice.  On the MPU-6050 each
loop is unrolled into a chain of forward exits and unit() is inlined twice: bigger, same result.

Packet: gravity x, y, z, spin axis x, y, z and their cosine (q30), 28 B.
Exercises: @dmp.sub, dmp.loop with break, an early return, multiple return values,
scaling by powers of two.
"""
import opendmp as dmp

EPS = 2e-6


@dmp.sub                                 # ICM: one subroutine called twice; MPU: inlined
def unit(x, y, z):
    """(x, y, z) / |(x, y, z)| for components in [-0.5, 0.5); a zero vector gives zeros."""
    m = x * x + y * y + z * z            # below 0.75
    if m == 0.0:
        return 0.0, 0.0, 0.0
    for _ in dmp.loop(14):               # 14 doublings reach the smallest nonzero int16
        if m >= 0.25:
            break
        x = x * 2
        y = y * 2
        z = z * 2
        m = m * 4
    h = 0.75                             # h = 0.5 / sqrt(m), in (0.5, 1]
    for _ in dmp.loop(6):
        t = m * h * h
        if abs(0.25 - t) < EPS:
            break
        h = h * (1.5 - t - t)
    return 2 * (x * h), 2 * (y * h), 2 * (z * h)


@dmp.program(rate=100)
def axes(s):
    gx, gy, gz = unit(s.accel.x * 0.25, s.accel.y * 0.25, s.accel.z * 0.25)
    wx, wy, wz = unit(s.gyro.x * 0.25, s.gyro.y * 0.25, s.gyro.z * 0.25)
    c = gx * wx + gy * wy + gz * wz
    dmp.emit(gx=gx, gy=gy, gz=gz, wx=wx, wy=wy, wz=wz, cos=c)
