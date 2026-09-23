"""Exact unit vectors without divide or square root: the gravity direction and the spin axis,
and the cosine of the angle between them (is the device spinning about the vertical?).

u = v / |v| is computed as v * rsqrt(|v|^2):
  1. range reduction: double v until |v|^2 lies in [1/4, 1)           a loop, <= 15 trips,
                                                                    recomputing |v|^2 each trip
  2. Newton for h = 0.5/sqrt(m):  h <- h * (1.5 - 2*m*h^2)            a loop, <= 6 trips,
                                                                    left once converged
  3. u = 2 * v * h
Unlike gyro_quat.py's single Newton step, this works for any |v|, including free fall and
shocks.  The spin axis is only meaningful while the rotation is well above the gyro noise.

The two targets compile this differently, which the listings show.  On the ICM-20948 unit() is
one subroutine called twice; the MPU-6050 has no call, so there it is inlined twice.  On both, the
loops are real backward branches (the MPU's reach 32 bytes, so its longer ones go through hops).

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
    for _ in dmp.loop(15):               # 15 doublings take a 1-LSB sample (x 1/4) to 1/2
        if m >= 0.25:
            break
        # m = x^2 + y^2 + z^2 < 1/4 here, so each component is below 1/2 (and doubles safely)
        dmp.assume(x, -0.5, 0.5)
        dmp.assume(y, -0.5, 0.5)
        dmp.assume(z, -0.5, 0.5)
        x = x * 2
        y = y * 2
        z = z * 2
        # recompute m rather than m * 4: for a vector of a few LSB, x * x floors away most of
        # |v|^2, and scaling that error by 4 per trip would leave |u| up to 12 % off
        m = dmp.assume(x * x + y * y + z * z, 0.0, 1.0)      # 4 x the last m, which was < 1/4
    h = 0.75                             # h = 0.5 / sqrt(m): m in [1/4, 1), h in (1/2, 1]
    for _ in dmp.loop(6):
        t = m * h * h
        if abs(0.25 - t) < EPS:
            break
        h = dmp.assume(h * (1.5 - t - t), 0.25, 1.0)    # from 0.75, Newton dips to 0.28 for m near 1
    # the components of a unit vector, to Newton's precision
    return (dmp.assume(2 * (x * h), -1.01, 1.01), dmp.assume(2 * (y * h), -1.01, 1.01),
            dmp.assume(2 * (z * h), -1.01, 1.01))


@dmp.program(rate=100)
def axes(s):
    gx, gy, gz = unit(s.accel.x * 0.25, s.accel.y * 0.25, s.accel.z * 0.25)
    wx, wy, wz = unit(s.gyro.x * 0.25, s.gyro.y * 0.25, s.gyro.z * 0.25)
    c = dmp.assume(gx * wx + gy * wy + gz * wz, -1.02, 1.02)   # two unit vectors
    dmp.emit(gx=gx, gy=gy, gz=gz, wx=wx, wy=wy, wz=wz, cos=c)
