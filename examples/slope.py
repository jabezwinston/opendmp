"""The board's slope, x over z of gravity, by exact division; smoothed; an interrupt when steep.

    k   = accel_x / accel_z            tan(pitch) while the board is near level
    avg = 0.9 * avg + 0.1 * k          clamped to +-1.9, so abs(avg) cannot meet -2.0

The DMP has no divide instruction.  dmp.div is restoring long division written in the DMP
language: the integer bit and 30 fraction bits, exact to the last bit of Q30 for every input,
saturating at +-2 (and at z = 0, on edge).  On the ICM a program calling it more than once gets
it as one routine; this one calls it once, so it is inlined on both chips.  `opendmp build`
says which, line by line.

Packet: k and avg (q30), 8 B.
Exercises: dmp.div, a clamp, a ranged param, the compile-time method report.
"""
import opendmp as dmp


@dmp.program(rate=50)
def slope(s):
    avg: dmp.state(-1.9, 1.9) = 0.0
    steep: dmp.param(0.0, 1.9) = 1.0                 # |tan(pitch)| above this interrupts: 45 degrees
    k = dmp.div(s.accel.x, s.accel.z)
    avg = dmp.clamp(0.9 * avg + 0.1 * k, -1.9, 1.9)
    dmp.emit(k=k, avg=avg)
    if abs(avg) > steep:
        dmp.irq()
