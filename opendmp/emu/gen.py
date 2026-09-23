"""Synthetic sensor data as CSV, for exercising programs without a rig.

    python -m opendmp.emu.gen SCENARIO[,SCENARIO...] --rate 200 --seconds 2 -o data.csv

Scenarios (each for --seconds, concatenated in order): rest, tilt, rotate, walk, taps,
vibration, freefall, shake.  Output is raw int16 counts at the given full-scale ranges.
"""
import argparse
import math
import random

from .files import write_sensor_csv
from .machine import Sample

SCENARIOS = ("rest", "tilt", "rotate", "walk", "taps", "vibration", "freefall", "shake")


def physical(name: str, t: float, i: int, rate: float):
    """(accel g, gyro deg/s) at time t within the scenario."""
    a, g = [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]
    if name == "tilt":                                  # slow 0.25 Hz rocking about y, +/-30 deg
        th = math.radians(30) * math.sin(2 * math.pi * 0.25 * t)
        a = [math.sin(th), 0.0, math.cos(th)]
        g = [0.0, math.degrees(math.radians(30) * 2 * math.pi * 0.25 * math.cos(2 * math.pi * 0.25 * t)), 0.0]
    elif name == "rotate":                              # 300 deg/s about z
        g = [0.0, 0.0, 300.0]
    elif name == "walk":                                # 2 Hz steps
        s = math.sin(4 * math.pi * t)
        a = [0.1 * s, 0.05, 1.0 + 0.4 * s]
    elif name == "taps":                                # a sharp spike every 0.25 s
        a = [0.0, 0.0, 1.0 + (1.2 if i % max(1, int(0.25 * rate)) == 0 else 0.0)]
    elif name == "vibration":                           # 10 Hz, 0.5 g
        a = [0.0, 0.0, 1.0 + 0.5 * math.sin(2 * math.pi * 10 * t)]
    elif name == "freefall":
        a = [0.0, 0.0, 0.0]
    elif name == "shake":                               # near full scale on every axis
        a = [1.9 * math.sin(30 * t), -1.9 * math.cos(30 * t), 1.9 * math.sin(50 * t)]
        g = [1800 * math.sin(20 * t), -1800.0, 1500.0]
    return a, g


def generate(names, rate=200.0, seconds=2.0, accel_fs=2.0, gyro_fs=2000.0, noise=True, seed=1):
    rnd = random.Random(seed)
    a_lsb, g_lsb = 32768 / accel_fs, 32768 / gyro_fs
    out, i = [], 0
    for name in names:
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name!r} (known: {', '.join(SCENARIOS)})")
        for k in range(int(rate * seconds)):
            a, g = physical(name, k / rate, k, rate)
            if noise:
                a = [v + rnd.gauss(0, 0.004) for v in a]
                g = [v + rnd.gauss(0, 0.05) for v in g]
            q = lambda v, lsb: max(-32768, min(32767, int(round(v * lsb))))
            out.append(Sample(q(a[0], a_lsb), q(a[1], a_lsb), q(a[2], a_lsb),
                              q(g[0], g_lsb), q(g[1], g_lsb), q(g[2], g_lsb), temp=-2000))
            i += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m opendmp.emu.gen", description=__doc__.split("\n")[0])
    ap.add_argument("scenarios", help=f"comma-separated: {', '.join(SCENARIOS)}")
    ap.add_argument("--rate", type=float, default=200)
    ap.add_argument("--seconds", type=float, default=2)
    ap.add_argument("--accel-fs", type=float, default=2)
    ap.add_argument("--gyro-fs", type=float, default=2000)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    s = generate(a.scenarios.split(","), a.rate, a.seconds, a.accel_fs, a.gyro_fs, not a.no_noise)
    write_sensor_csv(a.out, s)
    print(f"wrote {len(s)} samples ({a.scenarios}) at {a.rate:g} Hz to {a.out}")


if __name__ == "__main__":
    main()
