"""PID levelling: hold a platform level with one servo, with the control loop running on the
DMP.

    pitch   from a complementary filter, as in tilt.py
    error = setpoint - pitch
    I    <- clamp(I + KI*error, -ILIM, ILIM)       anti-windup by clamping
    u     = clamp(KP*error + I - KD*rate, -1, 1)    derivative on the measurement: the gyro
                                                    supplies it, and a setpoint step cannot kick
The host reads u from the FIFO and forwards it to the servo.  `setpoint` is a live param.

The declared ranges are what keep the unclamped sum below 2 ahead of the final clamp: with
|error| <= 1.1, KP*error + I - KD*rate stays within +/-1.92.  Widen a range or raise a gain
and the compiler warns that the sum can wrap, which would flip the command's sign.

Packet: u and pitch (q30), 8 B.
Exercises: a param with a range (the host runtime refuses writes outside it), explicit
saturation, a three-term sum, the derivative taken straight from the gyro.
"""
import math

import opendmp as dmp

RATE = 200
K = 0.98
GYRO_DT = math.radians(1000) / RATE
KP = 1.2
KI = 0.5 / RATE
KD = 0.05                               # per gyro unit (1000 dps)
ILIM = 0.5


@dmp.program(rate=RATE)
def level(s):
    pitch: dmp.state(-0.8, 0.8) = 0.0
    integ: dmp.state(-ILIM, ILIM) = 0.0
    setpoint: dmp.param(-0.3, 0.3) = 0.0

    rate = s.gyro.y
    pitch = dmp.clamp(K * (pitch + rate * GYRO_DT) + (1 - K) * s.accel.x, -0.8, 0.8)
    error = setpoint - pitch
    integ = dmp.clamp(integ + KI * error, -ILIM, ILIM)
    u = dmp.clamp(KP * error + integ - KD * rate, -1.0, 1.0)
    dmp.emit(u=u, pitch=pitch)
