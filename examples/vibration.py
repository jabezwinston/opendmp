"""Vibration monitor: a running mean, the running mean square of the deviation from it, and
an alarm with hysteresis.

    mean <- (1 - B) * mean + B * x             exponential moving average
    d     = (x - mean) / 4                     quartered, so d*d stays far below 2
    ms   <- (1 - G) * ms + G * d*d             mean square of the deviation, /16
    alarm on above ON, off below OFF           RMS limits, squared at compile time

An RMS needs a square root, which the DMP does not have.  Comparing the mean square with
squared limits gives the same decision without one.  The interrupt fires once, on the way up.

Packet: ms (q30) and the alarm state (Int), 8 B.
Exercises: EMA statistics, thresholds derived at compile time, Int state, hysteresis, an
edge-triggered interrupt.
"""
import opendmp as dmp

B = 1 / 64                  # the mean remembers ~64 samples
G = 1 / 16                  # the mean square remembers ~16
ON = (0.30 / 4) ** 2        # alarm above 0.30 g RMS (quartered, then squared)
OFF = (0.20 / 4) ** 2       # clear below 0.20 g RMS


@dmp.program(rate=400)
def vibration(s):
    mean: dmp.state = 1.0           # 1 g: the part starts out lying flat
    ms: dmp.state = 0.0
    alarm: dmp.state = 0

    x = s.accel.z
    mean = (1 - B) * mean + B * x
    d = 0.25 * x - 0.25 * mean      # quarter each side first: x - mean alone can reach 4
    ms = (1 - G) * ms + G * (d * d)
    if alarm == 0 and ms > ON:
        alarm = 1
        dmp.irq()
    elif alarm == 1 and ms < OFF:
        alarm = 0
    dmp.emit(ms=ms, alarm=alarm)
