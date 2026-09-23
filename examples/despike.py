"""Spike removal: a median of three on each gyro axis, then a light low-pass.

A single-sample spike (a knock on the enclosure, a glitch on the bus) goes straight through a
linear filter and smears over its whole impulse response.  The median of the last three
samples deletes it outright and keeps real edges sharp:

    med3(a, b, c) = max(min(a, b), min(max(a, b), c))

MIN is one instruction (the f5 family); MAX is MIN of the negations.  The history is a state
list shifted by plain assignments: the unrolled delay line the vendor firmware uses for its
accelerometer queue (ACCQD0..7).

Packet: the filtered gyro x, y, z (q30), 12 B.
Exercises: min/max, a helper (inlined), per-axis state lists, `for` over axes (unrolled).
"""
import opendmp as dmp

ALPHA = 0.25                        # low-pass weight after the median


def med3(a, b, c):
    return max(min(a, b), min(max(a, b), c))


@dmp.program(rate=200)
def despike(s):
    hist: dmp.state = [[0.0, 0.0] for _ in range(3)]      # per axis: x[n-1], x[n-2]
    out: dmp.state = [0.0, 0.0, 0.0]

    for i in range(3):
        x = s.gyro[i]
        m = med3(x, hist[i][0], hist[i][1])
        hist[i][1] = hist[i][0]
        hist[i][0] = x
        # a weighted sum, not out + ALPHA*(m - out): that difference can reach 4 and wrap
        out[i] = (1 - ALPHA) * out[i] + ALPHA * m
    dmp.emit(gyro_x=out[0], gyro_y=out[1], gyro_z=out[2])
