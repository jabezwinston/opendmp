"""Stream the raw accelerometer: the smallest useful DMP program (milestone M1).

Packet: accel x, y, z as int16, 6 B per sample.
Exercises: sensor extraction, FIFO routing, 16-bit emits.
"""
import opendmp as dmp


@dmp.program(rate=100)
def hello(s):
    dmp.emit(s.accel.x, s.accel.y, s.accel.z)
