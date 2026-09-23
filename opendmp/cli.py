"""opendmp command line.

    python -m opendmp build prog.py --chip mpu6050  compile: .lst .asm .bin .json in build/
    python -m opendmp run prog.py --port /dev/ttyUSB0 [--verify]   on a bench rig (ESP32 bridge)
    python -m opendmp check prog.py                 parse, type, and look for values that can wrap
    python -m opendmp sim prog.py --gen walk,taps   run it on sensor data (or --sensors data.csv;
                                                    --engine emu:icm20948 runs the compiled bytes)
    python -m opendmp ir prog.py                    what the front end made of it
    python -m opendmp dis image.bin --chip mpu6050 --org 0x400     annotated disassembly
    python -m opendmp asm prog.asm [-o prog.bin]    assemble, print the listing
    python -m opendmp facts [--chip icm20948]       the machine facts and their evidence

The emulator is `python -m opendmp.emu`.
"""
import argparse
import csv
import pathlib
import runpy
import sys

from .emu.chips import get_chip
from .isa import AsmError, assemble, chip_name, cost_key, disassemble, effect, to_text
from .isa import facts as _facts


def listing(lines, chip, title: str = "") -> str:
    """addr, bytes, asm, effect, cycles and us for each line (cycles as executed, not skipped;
    a MPU branch as taken)."""
    c = get_chip(chip_name(chip))
    out = [f"; {title}{c.name} ({c.clock_hz / 1e6:.3f} MHz)",
           "; addr  bytes     asm                    effect                                   cycles     us"]
    total = 0
    for ln in lines:
        if ln.insn is None:
            words = [ln.data[i:i + 4].hex() for i in range(0, len(ln.data), 4)]
            out.append(f"{ln.addr:04x}  .dw {' '.join(words)}")
            continue
        cyc = c.cyc(cost_key(ln.insn, ln.mode if ln.mode is not None else 1))
        total += cyc
        out.append(f"{ln.addr:04x}  {ln.data.hex(' '):9s} {ln.insn.text():22s} {effect(ln.insn):40s} "
                   f"{cyc:6d} {cyc * c.cycle_ns / 1000:6.2f}")
    code = sum(len(ln.data) for ln in lines if ln.insn is not None)
    out.append(f"; {code} B of code; straight through: {total} cycles = {total * c.cycle_ns / 1000:.1f} us"
               f" (+{c.start_cycles} after the tick)")
    return "\n".join(out)


def cmd_dis(a):
    data = pathlib.Path(a.file).read_bytes()
    chip = chip_name(a.chip)
    start = int(a.start, 0) if a.start else 0
    org = int(a.org, 0) if a.org else 0
    data = data[start:start + int(a.length, 0)] if a.length else data[start:]
    lines = disassemble(data, chip, org=org + start, mode=a.mode)
    if a.asm:
        sys.stdout.write(to_text(lines, chip))
    else:
        print(listing(lines, chip, f"{pathlib.Path(a.file).name}  "))
    return 0


def cmd_asm(a):
    try:
        prog = assemble(pathlib.Path(a.file).read_text(), a.chip)
    except AsmError as e:
        print(f"{a.file}: {e}", file=sys.stderr)
        return 1
    print(listing(prog.lines, prog.chip, f"{pathlib.Path(a.file).name}  entry {prog.entry:#06x}  "))
    if a.output:
        img = prog.image()
        end = max(addr + len(d) for addr, d in prog.segments())
        pathlib.Path(a.output).write_bytes(bytes(img[:end]))
        print(f"; wrote {a.output}: memory image 0x0000-{end - 1:#06x}, load at 0 "
              f"(PRGM_START = {prog.entry:#06x})")
    return 0


def load_programs(path, name=None):
    ns = runpy.run_path(str(path), run_name="__opendmp__")
    from .api import Program
    progs = [v for v in ns.values() if isinstance(v, Program)]
    if name:
        progs = [p for p in progs if p.name == name]
    if not progs:
        raise SystemExit(f"{path}: no @dmp.program" + (f" named {name!r}" if name else ""))
    return progs


def describe(ir) -> str:
    o = ir.opts
    ns, np_ = len(ir.states), len(ir.params)
    parts = [f"{ir.name}: {o['rate']:g} Hz, gyro +/-{o['gyro_fs']} dps, accel +/-{o['accel_fs']} g",
             f"{ns} state word{'s' if ns != 1 else ''}, {np_} param{'s' if np_ != 1 else ''}"]
    if ir.subs:
        parts.append("subs " + ", ".join(ir.subs))
    head = "; ".join(parts)
    pk = ", ".join(f"{f.name} {'i16' if f.type == 's16' else f.type}" for f in ir.packet)
    return f"{head}\n  packet {ir.packet_size} B: {pk}"


def _lowered(p):
    from .lang.errors import DmpError
    try:
        return p.ir
    except DmpError as e:
        print(e, file=sys.stderr)
        return None


def cmd_check(a):
    from .ir.ranges import analyse
    rc = 0
    for p in load_programs(a.file, a.name):
        ir = _lowered(p)
        if ir is None:
            rc = 1
            continue
        print(describe(ir))
        ws = analyse(ir).warnings
        for w in ws:
            print(w)
        print(f"  {len(ws)} range warning{'s' if len(ws) != 1 else ''}" if ws else "  nothing can wrap")
        if ws and a.strict:
            rc = 1
    return rc


def cmd_ir(a):
    from .ir.show import program
    rc = 0
    for p in load_programs(a.file, a.name):
        ir = _lowered(p)
        if ir is None:
            rc = 1
            continue
        print(program(ir))
    return rc


def cmd_sim(a):
    from .emu.files import read_sensor_csv
    from .emu.gen import generate
    progs = load_programs(a.file, a.name)
    if len(progs) > 1:
        raise SystemExit(f"{a.file} has {len(progs)} programs; pick one with --name")
    p = progs[0]
    if _lowered(p) is None:
        return 1
    o = p.opts
    if a.sensors:
        samples = read_sensor_csv(a.sensors, a.units, o["accel_fs"], o["gyro_fs"])
    else:
        samples = generate((a.gen or "rest,tilt,walk,taps,shake").split(","), rate=o["rate"],
                           seconds=a.seconds, accel_fs=o["accel_fs"], gyro_fs=o["gyro_fs"])
    params = {}
    for kv in a.param or []:
        k, v = kv.split("=", 1)
        params[k] = float(v) if "." in v or "e" in v.lower() else int(v, 0)
    res = p.simulate(samples, a.engine, params)
    print(res.summary())
    if a.output:
        with open(a.output, "w", newline="") as f:
            w = csv.writer(f)
            names = [fl.name for fl in res.packet]
            w.writerow(["pass", "irq"] + names)
            for i, (d, irq) in enumerate(zip(res.fields(), res.irqs)):
                w.writerow([i, int(irq)] + [d[n] for n in names])
        print(f"  wrote {a.output}")
    return 0


def cmd_build(a):
    from .lang.errors import DmpError
    rc = 0
    for p in load_programs(a.file, a.name):
        if _lowered(p) is None:
            rc = 1
            continue
        for chip in a.chip:
            try:
                b = p.build(chip)
            except DmpError as e:
                print(e, file=sys.stderr)
                rc = 1
                continue
            if a.list:
                print(b.listing())
            files = b.write(a.output)
            from .emu.chips import get_chip
            c = get_chip(b.chip)
            us = b.cycles_straight() * c.cycle_ns / 1000
            print(f"{p.name} for the {b.chip}: {b.size} B of code, packet {p.ir.packet_size} B, "
                  f"at most {us:.1f} us per pass ({100 * us * p.opts['rate'] / 1e6:.1f} % of the period)"
                  f"  -> {', '.join(str(f) for f in files)}")
    return rc


def cmd_run(a):
    from .host import Rig, verify
    from .sim import decode
    progs = load_programs(a.file, a.name)
    if len(progs) > 1:
        raise SystemExit(f"{a.file} has {len(progs)} programs; pick one with --name")
    p = progs[0]
    if _lowered(p) is None:
        return 1
    rig = Rig(a.port)
    try:
        if a.verify:
            v = verify(p, a.port, seconds=a.seconds, rate=a.rate, rig=rig)
            print(v.summary())
            fields = v.fields
            ok = v.ok
        else:
            b = p.build(rig.chip)
            rig.load(b, a.rate)
            packets, ints = rig.run(a.seconds)
            fields = [decode(p.ir.packet, pk) for pk in packets]
            print(f"{p.name} on the {rig.chip}: {len(packets)} packets at {rig.rate:.1f} Hz, {ints} interrupts")
            ok = True
    finally:
        rig.close()
    for f in p.ir.packet:
        col = [d[f.name] for d in fields]
        if col:
            print(f"  {f.name:12s} min {min(col):>12.6g}  max {max(col):>12.6g}  last {col[-1]:>12.6g}")
    if a.output and fields:
        with open(a.output, "w", newline="") as fh:
            w = csv.writer(fh)
            names = [f.name for f in p.ir.packet]
            w.writerow(["pass"] + names)
            for i, d in enumerate(fields):
                w.writerow([i] + [d[n] for n in names])
        print(f"  wrote {a.output}")
    return 0 if ok else 1


def cmd_facts(a):
    print(_facts.table(chip_name(a.chip) if a.chip else None))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m opendmp", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dis", help="disassemble a binary")
    d.add_argument("file")
    d.add_argument("--chip", default="mpu6050")
    d.add_argument("--org", help="address of the file's first byte (default 0)")
    d.add_argument("--start", help="offset into the file to start at")
    d.add_argument("--length", help="bytes to disassemble")
    d.add_argument("--mode", type=int, default=1, help="mode register at the start (default 1)")
    d.add_argument("--asm", action="store_true", help="print reassemblable .asm instead of a listing")
    d.set_defaults(fn=cmd_dis)
    s = sub.add_parser("asm", help="assemble an opendmp .asm file")
    s.add_argument("file")
    s.add_argument("--chip", help="override the file's .chip")
    s.add_argument("-o", "--output", help="write the memory image (load at 0)")
    s.set_defaults(fn=cmd_asm)
    bl = sub.add_parser("build", help="compile a program for a chip")
    bl.add_argument("file")
    bl.add_argument("--name")
    bl.add_argument("--chip", action="append", choices=("mpu6050", "icm20948", "mpu", "icm"),
                    help="the target (repeatable; default both)")
    bl.add_argument("-o", "--output", default="build", help="output directory (default build/)")
    bl.add_argument("--list", action="store_true", help="print the listing")
    bl.set_defaults(fn=cmd_build)
    for name, fn, hlp in (("check", cmd_check, "parse a program and report what can wrap"),
                          ("ir", cmd_ir, "print the IR the front end made")):
        c = sub.add_parser(name, help=hlp)
        c.add_argument("file")
        c.add_argument("--name", help="the program, when the file has several")
        if name == "check":
            c.add_argument("--strict", action="store_true", help="exit 1 on a range warning")
        c.set_defaults(fn=fn)
    m = sub.add_parser("sim", help="simulate a program on sensor data (the reference semantics)")
    m.add_argument("file")
    m.add_argument("--name")
    m.add_argument("--sensors", help="sensor CSV (python -m opendmp.emu.gen makes one)")
    m.add_argument("--units", choices=("raw", "si"), default="raw")
    m.add_argument("--gen", help="synthetic motion instead: "
                   "rest,tilt,rotate,walk,taps,vibration,freefall,shake")
    m.add_argument("--seconds", type=float, default=1.0, help="per --gen scenario")
    m.add_argument("--engine", choices=("reference", "ir", "emu:mpu6050", "emu:icm20948"),
                   default="reference")
    m.add_argument("--param", action="append", metavar="NAME=VALUE", help="set a dmp.param")
    m.add_argument("-o", "--output", help="write the decoded packets as CSV")
    m.set_defaults(fn=cmd_sim)
    r = sub.add_parser("run", help="run a program on a bench rig (the ESP32 bridge)")
    r.add_argument("file")
    r.add_argument("--name")
    r.add_argument("--port", required=True, help="the bridge's serial port (the part is detected)")
    r.add_argument("--seconds", type=float, default=3.0)
    r.add_argument("--rate", type=float, help="sample rate override (the bridge reads ~5 KB/s)")
    r.add_argument("--verify", action="store_true",
                   help="a trace build, replayed in the emulator and the reference: bit-exact or not")
    r.add_argument("-o", "--output", help="write the decoded packets as CSV")
    r.set_defaults(fn=cmd_run)
    f = sub.add_parser("facts", help="the machine facts the tools rely on, with their evidence")
    f.add_argument("--chip")
    f.set_defaults(fn=cmd_facts)
    a = ap.parse_args(argv)
    if getattr(a, "fn", None) is cmd_build and not a.chip:
        a.chip = ["mpu6050", "icm20948"]
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
