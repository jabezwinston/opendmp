"""Six-axis attitude with the Mahony filter: gyro integration, pulled towards gravity by a
PI controller on the tilt error.

    v = gravity as q predicts it          (third row of q's rotation matrix)
    e = a x v                             tilt error in body axes (a halved: no wrap at 2 g)
    b <- clamp(b + KI*e)                  integral term: learns the gyro bias; anti-windup
    h = w*dt/2 + KP*e + b                 corrected half-angle step
    q <- normalize(q + q (x) (0, h))      as in gyro_quat.py, from the dmp.quat library

The accelerometer is not normalized.  At rest |a| = 1 g, and the correction is small enough
that the error scaling with |a| during motion does no harm.  (unit_vectors.py shows the
exact normalization, which costs two loops.)  Heading still drifts, because gravity says
nothing about yaw.

Packet: w, x, y, z (q30), 16 B.
Exercises: the dmp.quat library, cross products (MAC runs with a negative term), clamp,
gains derived at compile time (KP = 2.0 is fine: only KP*DT reaches the DMP).
"""
import math

import opendmp as dmp

RATE = 200
DT = 1 / RATE
H = 0.5 * math.radians(1000) * DT       # gyro unit -> half-angle per pass
KP = 2.0                                # rad/s per unit of tilt error
KI = 0.1                                # rad/s^2 per unit of tilt error
KPH = KP * DT                           # e comes from a/2, so KP * dt/2 * 2
KIH = KI * DT * DT
BMAX = 0.5 * math.radians(10) * DT      # the bias estimate may absorb at most 10 dps


@dmp.program(rate=RATE)
def mahony(s):
    q: dmp.state(-1.0, 1.0) = [1.0, 0.0, 0.0, 0.0]
    b: dmp.state = [0.0, 0.0, 0.0]

    vx, vy, vz = dmp.quat.gravity(q)
    ax = s.accel.x * 0.5
    ay = s.accel.y * 0.5
    az = s.accel.z * 0.5
    e = [ay * vz - az * vy,
         az * vx - ax * vz,
         ax * vy - ay * vx]
    h = [0.0, 0.0, 0.0]
    for i in range(3):
        b[i] = dmp.clamp(b[i] + KIH * e[i], -BMAX, BMAX)
        h[i] = s.gyro[i] * H + KPH * e[i] + b[i]
    q = dmp.quat.normalize(dmp.quat.rate_step(q, h))
    dmp.emit(w=q[0], x=q[1], y=q[2], z=q[3])
