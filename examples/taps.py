"""Single and double tap detection from the accelerometer's jerk.

    jerk = |z[n] - z[n-1]|      a tap is a spike in the derivative, not in the level
    tap:    jerk > THRESH, outside the QUIET window that follows every tap (the ringing)
    double: a second tap within WINDOW passes of the first
    single: the window runs out after one tap

Packet: the event (Int: 0 none, 1 single, 2 double) and the jerk (q30), 8 B.  The interrupt
fires on every event.
Exercises: a state machine with two countdown timers, abs, nested if/else.
"""
import opendmp as dmp

RATE = 400
THRESH = 0.5 / 2                # a change of 0.5 g within one pass (z is halved below)
QUIET = int(0.04 * RATE)        # ignore 40 ms of ringing after each tap
WINDOW = int(0.30 * RATE)       # a second tap within 300 ms makes a double


@dmp.program(rate=RATE)
def taps(s):
    prev: dmp.state = 0.5           # z/2 at rest, lying flat
    quiet: dmp.state = 0
    window: dmp.state = 0

    z = s.accel.z * 0.5             # halved, so z - prev cannot wrap
    jerk = abs(z - prev)
    prev = z
    event = 0

    if window > 0:
        window -= 1
        if window == 0:
            event = 1               # the window closed on a lone tap
    if quiet > 0:
        quiet -= 1
    elif jerk > THRESH:
        quiet = QUIET
        if window > 0:
            window = 0
            event = 2
        else:
            window = WINDOW
    if event != 0:
        dmp.irq()
    dmp.emit(event=event, jerk=jerk)
