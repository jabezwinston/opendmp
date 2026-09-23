"""Pedometer: count the peaks of the acceleration magnitude, with hysteresis and a minimum
interval between steps.

    m = |a|^2 / 16      each axis is quartered first, so the sum of squares stays below 1
    step:   m rises above HI while armed, at least MIN_GAP passes after the previous step
    re-arm: m falls below LO

The thresholds are magnitudes in g, squared and scaled at compile time, so no square root is
needed.

Packet: the step count (Int) and m (q30), 8 B.  The interrupt fires on every step.
Exercises: a three-term sum of squares (one MAC run), Int counters and timers, compound
conditions with `and`, if/elif.
"""
import opendmp as dmp

RATE = 100
HI = (1.20 / 4) ** 2            # a step peaks above 1.2 g
LO = (1.05 / 4) ** 2            # the detector re-arms below 1.05 g
MIN_GAP = int(0.25 * RATE)      # at most 4 steps a second


@dmp.program(rate=RATE)
def pedometer(s):
    steps: dmp.state = 0
    armed: dmp.state = 1
    gap: dmp.state = 0

    ax = s.accel.x * 0.25
    ay = s.accel.y * 0.25
    az = s.accel.z * 0.25
    m = ax * ax + ay * ay + az * az

    if gap > 0:
        gap -= 1
    if armed == 1 and gap == 0 and m > HI:
        steps += 1
        armed = 0
        gap = MIN_GAP
        dmp.irq()
    elif m < LO:
        armed = 1
    dmp.emit(steps=steps, m=m)
