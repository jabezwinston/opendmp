"""opendmp command line.

    python -m opendmp dis image.bin --chip mpu6050 --org 0x400     annotated disassembly
    python -m opendmp dis image.bin --chip icm20948 --org 0x1000 --asm > out.asm
    python -m opendmp asm prog.asm [-o prog.bin]                    assemble, print the listing
    python -m opendmp facts [--chip icm20948]                       the machine facts and evidence

The emulator is `python -m opendmp.emu`.  `build`, `sim`, `run` and `check` come with the
compiler (PLAN.md section 9).
"""
import argparse
import pathlib
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
    f = sub.add_parser("facts", help="the machine facts the tools rely on, with their evidence")
    f.add_argument("--chip")
    f.set_defaults(fn=cmd_facts)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
