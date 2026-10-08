"""Run DMP machine code over sensor data from files.

    python -m opendmp.emu [run.toml] [options]

Every option can also be set in a TOML file (same names, dashes or underscores); options on the
command line override the file.  Example run.toml:

    chip = "mpu6050"
    rate = 200                      # sample rate, Hz: one CSV row per tick
    sensors = "walk.csv"
    units = "raw"                   # or "si": g, deg/s, deg C
    load = ["tilt.bin@0x400"]
    poke = ["0x100=0.0873q,1.0q,0.98q,0.02q", "0x120=0.25q"]
    entry = 0x400
    log = "passes.csv"              # one row per pass
    fifo = "out.fifo"               # the raw FIFO stream
    trace = "trace.txt"             # instruction trace ...
    trace_passes = 2                # ... of the first N passes
    watch = ["0x11c"]               # RAM words logged each pass (Q30 in the log)
    model = ["mac_precision=q30"]   # see opendmp.emu.machine.Model (defaults: measured)

With no `sensors`, every pass sees one fixed sample: `sample = "ax=100,gz=-5"` (zeros if
omitted), for `passes` passes (default 1).
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
import tomllib

from ..isa.core import chip_name
from ..q30 import from_q30
from .chips import get_chip
from .files import parse_addr, parse_load, parse_poke, parse_sample, read_asm, read_program, \
    read_sensor_csv
from .machine import DmpFault, Machine, Model, Unmodelled
from .sched import run, summary

MODEL_FIELDS = {"mac_restart", "mac_precision", "icm_f0", "carry", "max_steps"}


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m opendmp.emu",
                                 description="Timed DMP emulator fed from files.")
    ap.add_argument("config", nargs="?", help="TOML file with any of the options below")
    ap.add_argument("--chip", help="mpu6050 or icm20948 (default: a .asm's .chip, else mpu6050)")
    ap.add_argument("--rate", type=float, help="sample rate in Hz (default 200)")
    ap.add_argument("--sensors", help="sensor CSV, one row per sample tick")
    ap.add_argument("--sample", help="no CSV: every pass sees this sample, e.g. ax=100,gz=-5 "
                                     "(default: zeros)")
    ap.add_argument("--units", choices=("raw", "si"))
    ap.add_argument("--accel-fs", type=float, help="accel full scale in g for --units si (default 2)")
    ap.add_argument("--gyro-fs", type=float, help="gyro full scale in deg/s for --units si (default 2000)")
    ap.add_argument("--load", action="append", metavar="FILE@ADDR", help="binary, .hex or .s at ADDR; an opendmp .asm needs no @ADDR")
    ap.add_argument("--poke", action="append", metavar="ADDR=W[,W..]", help="words: hex or 0.5q")
    ap.add_argument("--entry", help="PRGM_START (default 0x400 MPU, 0x1000 ICM)")
    ap.add_argument("--passes", type=int, help="stop after N passes")
    ap.add_argument("--regs", help="DMP3 I/O space: JSON {\"bank:reg\": value}")
    ap.add_argument("--fsync", action="store_true", default=None, help="DMP3: FSYNC into temp bit 0")
    ap.add_argument("--model", action="append", metavar="KEY=VALUE", help="emulator assumption")
    ap.add_argument("--log", help="per-pass CSV log")
    ap.add_argument("--fifo", help="raw FIFO output file")
    ap.add_argument("--trace", help="instruction trace file")
    ap.add_argument("--trace-passes", type=int, help="trace the first N passes (default 1)")
    ap.add_argument("--watch", action="append", metavar="ADDR", help="log this RAM word each pass")
    ap.add_argument("--dump", help="write DMP memory after the run")
    ap.add_argument("--costs", action="store_true", default=None, help="print the cycle table")
    ap.add_argument("--clock-mhz", type=float, help="DMP clock of your part (default: the bench's)")
    return ap


def merged_options(argv):
    ap = build_parser()
    a = ap.parse_args(argv)
    opts = {}
    if a.config:
        with open(a.config, "rb") as f:
            opts = {k.replace("-", "_"): v for k, v in tomllib.load(f).items()}
        base = pathlib.Path(a.config).resolve().parent
        for k in ("sensors", "log", "fifo", "trace", "dump", "regs"):
            if k in opts:
                opts[k] = str(base / opts[k])
        if "load" in opts:
            opts["load"] = [str(base / f) + ("" if ad is None else "@" + str(ad))
                            for f, ad in map(parse_load, opts["load"])]
    for k, v in vars(a).items():
        if k != "config" and v is not None:
            opts[k] = v
    return opts


def make_model(items) -> Model:
    kw = {}
    for it in items or []:
        k, v = it.split("=", 1)
        k = k.strip()
        if k not in MODEL_FIELDS:
            raise SystemExit(f"unknown model key {k!r} (known: {', '.join(sorted(MODEL_FIELDS))})")
        if k == "carry":
            v = tuple(x.strip() for x in v.split(",") if x.strip()) if v.strip() not in ("", "none") else ()
        elif k == "max_steps":
            v = int(v)
        kw[k] = v.strip() if isinstance(v, str) else v
    try:
        return Model(**kw)
    except ValueError as e:
        raise SystemExit(f"--model: {e}") from None


def asm_chip(loads):
    """The .chip of the first opendmp .asm loaded, if it names one."""
    for spec in loads:
        f, addr = parse_load(spec)
        if addr is None:
            m = re.search(r"^\s*\.chip\s+(\w+)", pathlib.Path(f).read_text(), re.M)
            return chip_name(m.group(1)) if m else None
    return None


def main(argv=None):
    o = merged_options(sys.argv[1:] if argv is None else argv)
    chip = get_chip(o.get("chip") or asm_chip(o.get("load", [])) or "mpu6050",
                    float(o["clock_mhz"]) * 1e6 if o.get("clock_mhz") else None)
    if o.get("costs"):
        print(f"{chip.name}: {chip.clock_hz / 1e6:.3f} MHz ({chip.cycle_ns:.2f} ns/cycle); pass starts "
              f"{chip.start_cycles} cycles after the tick; {chip.tick_stall_cycles} cycles per tick stall")
        for k, c, ns, src in chip.cost_report():
            print(f"  {k:8s} {c:3d} cycles  {ns / 1000:7.3f} us   {src}")
        if not o.get("load"):
            return 0
    if not o.get("load"):
        raise SystemExit("nothing to run: give --load FILE (see --help)")
    if o.get("sensors") and o.get("sample") is not None:
        raise SystemExit("--sensors and --sample: give one")
    rate = float(o.get("rate", 200))
    regs = None
    if o.get("regs"):
        raw = json.loads(pathlib.Path(o["regs"]).read_text())
        regs = {tuple(int(x, 0) for x in k.split(":")): int(str(v), 0) for k, v in raw.items()}
    m = Machine(chip, make_model(o.get("model")), regs=regs)
    m.fsync_enabled = bool(o.get("fsync"))
    for spec in o.get("load", []):
        f, addr = parse_load(spec)
        if addr is None:                                # an opendmp .asm places itself
            prog = read_asm(f, chip.name)
            for a, data in prog.segments():
                m.load(data, a)
            m.entry = prog.entry
        else:
            m.load(read_program(f, chip.name), addr)
    for spec in o.get("poke", []):
        addr, words = parse_poke(spec)
        m.poke_words(addr, words)
    if "entry" in o:
        m.entry = parse_addr(o["entry"])
    if o.get("sensors"):
        samples = read_sensor_csv(o["sensors"], o.get("units", "raw"), o.get("accel_fs", 2),
                                  o.get("gyro_fs", 2000), chip.name)
    else:
        try:
            one = parse_sample(o.get("sample", ""), o.get("units", "raw"), o.get("accel_fs", 2),
                               o.get("gyro_fs", 2000), chip.name)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        samples = [one] * int(o.get("passes") or 1)
    watch = [parse_addr(w) for w in o.get("watch", [])]

    trace_f = open(o["trace"], "w") if o.get("trace") else None
    trace_n = int(o.get("trace_passes", 1))
    log_f = open(o["log"], "w", newline="") if o.get("log") else None
    log = csv.writer(log_f) if log_f else None
    if log:
        log.writerow(["pass", "sample", "start_us", "pass_us", "cycles", "steps", "ints", "fifo_bytes",
                      "fifo_hex"]
                     + [f"{a:#06x}" for a in watch])
    fifo_f = open(o["fifo"], "wb") if o.get("fifo") else None
    state = {"pass": 0}

    def tracer(mach, pc, text, key, note):
        n = len(mach.mem[pc:mach.pc]) if 0 < mach.pc - pc <= 3 else mach.length(pc) if key == "skip" else 1
        raw = mach.mem[pc:pc + max(1, min(n, 3))].hex(" ")
        trace_f.write(f"{state['pass']:5d}  {pc:04x}  {raw:9s} {text:34s} "
                      f"{mach.chip.cost(key) / 1000:6.2f}  {key:10s} {note}\n")

    if trace_f:
        m.tracer = tracer

    def on_pass(r):
        if log:
            log.writerow([r.index, r.sample, f"{r.start_ns / 1000:.3f}", f"{r.duration_ns / 1000:.3f}",
                          r.cycles, r.steps, r.ints, len(r.fifo), r.fifo.hex()]
                         + [f"{from_q30(m.peek_word(a)):.9f}" for a in watch])
        if fifo_f:
            fifo_f.write(r.fifo)
        m.drain_fifo()
        state["pass"] = r.index + 1
        if trace_f and state["pass"] >= trace_n:
            m.tracer = None

    status = 0
    try:
        results = run(m, samples, rate, o.get("passes"), on_pass)
    except (DmpFault, Unmodelled) as e:
        print(f"stopped in pass {state['pass']}: {type(e).__name__}: {e}", file=sys.stderr)
        results, status = None, 2
    finally:
        for f in (trace_f, log_f, fifo_f):
            if f:
                f.close()
    if o.get("dump"):
        pathlib.Path(o["dump"]).write_bytes(bytes(m.mem))
    if results is not None:
        s = summary(results, rate)
        print(f"{chip.name}: {s['passes']} passes over {len(samples)} samples at {rate:g} Hz "
              f"({s['samples_skipped']} samples never seen)")
        print(f"  pass time {s['pass_us_mean']:.1f} us mean, {s['pass_us_max']:.1f} us max, "
              f"period {s['period_us']:.1f} us -> {s['passes_per_s']:.1f} passes/s")
        print(f"  FIFO {s['fifo_bytes']} B ({'/'.join(map(str, s['fifo_bytes_per_pass']))} B per pass), "
              f"{s['ints']} interrupts, at most {s['steps_max']} instructions per pass")
    return status


if __name__ == "__main__":
    sys.exit(main())
