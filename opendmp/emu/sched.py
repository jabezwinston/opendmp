"""The pass timeline: sample ticks, when each pass starts, and which sample it sees.

The hardware writes a sample into DMP RAM on every tick of the sample clock, and the program
runs from the entry point once per pass.  When a pass outruns the sample period the two chips
behave differently, both measured:

* MPU-6050: a pass can only start on a tick, so the rate drops to an exact integer submultiple
  (rate / ceil(pass / period)) [HW, probe38/41].
* ICM-20948: a pending tick starts the next pass as soon as the previous one ends, so the rate
  is simply 1 / pass time [HW, probe49].

A pass starts on the latest sample written at or before its start.  The hardware rewrites the
sample page at every tick, even under a running pass, so a pass longer than the period reads
newer samples part-way through [HW, InvensenseDMP Finding 47]; `run` hands those ticks to
`Machine.run_pass`.  On the MPU each sample write that lands inside a running pass also stalls it
(`Chip.tick_stall_cycles`, measured 14-28 cycles by mix).
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

from .machine import Machine, PassResult, Sample


def run(machine: Machine, samples: Sequence[Sample], rate_hz: float,
        max_passes: int | None = None,
        on_pass: Callable[[PassResult], None] | None = None,
        midpass: bool = True) -> list[PassResult]:
    """Run passes over `samples` (one per sample tick at `rate_hz`) until they run out.  With
    `midpass` (the default) a pass longer than the period sees the samples that arrive while it
    runs, as on silicon; without it, only its starting sample (stalls are then added after)."""
    if rate_hz <= 0:
        raise ValueError("rate must be positive")
    period = 1e9 / rate_hz
    policy = machine.chip.rate_policy
    results: list[PassResult] = []
    t, idx = 0.0, 0
    cycle_ns = machine.chip.cycle_ns

    def later(t0, i0):                              # the ticks after this pass's, in its cycles
        for j in range(i0 + 1, len(samples)):
            yield math.ceil((j * period - t0) / cycle_ns - 1e-9), samples[j]

    while idx < len(samples) and (max_passes is None or len(results) < max_passes):
        res = machine.run_pass(samples[idx], ticks=later(t, idx) if midpass else None)
        res.index, res.sample, res.start_ns = len(results), idx, t
        stall = machine.chip.tick_stall_cycles * machine.chip.cycle_ns
        if stall and not midpass:
            base, n = res.duration_ns, 0
            while True:                               # ticks inside the pass, stalls included
                k = int(math.floor((t + base + n * stall) / period + 1e-9)) - idx
                if k <= n:
                    break
                n = k
            res.duration_ns = base + n * stall
        results.append(res)
        if on_pass:
            on_pass(res)
        end = t + res.duration_ns
        if policy == "tick":
            nxt = math.ceil(end / period - 1e-9) * period
        else:
            nxt = max(end, (idx + 1) * period)
        idx = int(math.floor(nxt / period + 1e-9))
        t = nxt
    return results


def summary(results: Iterable[PassResult], rate_hz: float) -> dict:
    rs = list(results)
    if not rs:
        return {"passes": 0}
    durs = [r.duration_ns / 1000 for r in rs]
    last = rs[-1]
    span_s = (last.start_ns + last.duration_ns) / 1e9
    return {
        "passes": len(rs),
        "samples_seen": len({r.sample for r in rs}),
        "samples_skipped": last.sample + 1 - len({r.sample for r in rs}),
        "pass_us_mean": sum(durs) / len(durs),
        "pass_us_max": max(durs),
        "period_us": 1e6 / rate_hz,
        "passes_per_s": len(rs) / span_s if span_s else float("nan"),
        "fifo_bytes": sum(len(r.fifo) for r in rs),
        "fifo_bytes_per_pass": sorted({len(r.fifo) for r in rs}),
        "ints": sum(r.ints for r in rs),
        "steps_max": max(r.steps for r in rs),
    }
