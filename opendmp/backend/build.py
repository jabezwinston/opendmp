"""build(program, chip): the IR -> a loadable image, a listing, assembly, and a packet schema.

    b = prog.build("mpu6050")
    print(b.listing())
    b.emulate(samples)          # the bytes, run in opendmp.emu: must equal prog.simulate(samples)
"""
import json
import linecache
import pathlib
import re
from dataclasses import dataclass, field

from ..emu.chips import get_chip
from ..isa.core import Insn, chip_name, cost_key, effect, encode
from ..lang.errors import DmpError
from ..q30 import MASK, ONE
from . import tac as T
from .layout import optimise
from .opt import defs, optimize, uses
from .sched import Emitted, Scheduler, schedule

ENTRY = {"mpu6050": 0x0400, "icm20948": 0x1000}
CODE_END = {"mpu6050": 0x0C00, "icm20948": 0x3000}     # MPU: end of RAM; ICM: the FIFO at 0x3000
MEM = {"mpu6050": 0x0C00, "icm20948": 0x4000}


@dataclass
class Build:
    program: object
    ir: object
    chip: str
    entry: int
    image: bytearray                  # the whole memory image: data and code
    code: bytes
    out: list                         # [Emitted], with addresses
    words: list                       # [tac.Word], placed
    labels: dict = field(default_factory=dict)
    hops: int = 0
    trace: bool = False               # each packet starts with the pass's raw sample words

    # ------------------------------------------------------------------ numbers
    @property
    def size(self):
        return len(self.code)

    @property
    def trace_bytes(self):
        return 4 * len(T.TRACE_WORDS[self.chip]) if self.trace else 0

    @property
    def packet_size(self):
        return self.trace_bytes + self.ir.packet_size

    def trace_sample(self, packet: bytes):
        """The sensor sample a trace build's packet carries."""
        from ..emu import Sample
        w = [packet[i:i + 4] for i in range(0, self.trace_bytes, 4)]
        h = lambda b, k: int.from_bytes(b[2 * k:2 * k + 2], "big", signed=True)
        if self.chip == "mpu6050":           # ax|ay az|temp gx|gy gz|-
            return Sample(ax=h(w[0], 0), ay=h(w[0], 1), az=h(w[1], 0), temp=h(w[1], 1),
                          gx=h(w[2], 0), gy=h(w[2], 1), gz=h(w[3], 0))
        return Sample(ax=h(w[0], 0), ay=h(w[0], 1), az=h(w[1], 0), gx=h(w[1], 1),   # ax|ay az|gx
                      gy=h(w[2], 0), gz=h(w[2], 1), temp=h(w[3], 0))                 # gy|gz temp|-

    @property
    def bounded(self) -> bool:
        """True when no instruction can run twice in a pass: no backward branch, no call.  Then
        cycles_straight() is an upper bound on the pass."""
        for e in self.out:
            if e.insn is None:
                continue
            if e.insn.op == "call":
                return False
            if e.target is not None and self.labels[e.target] <= e.addr:
                return False
        return True

    def cycles_straight(self) -> int:
        """Every instruction once, the start latency included: an upper bound on a pass for
        code that only branches forward (each instruction runs at most once)."""
        chip = get_chip(self.chip)
        return chip.start_cycles + sum(chip.cyc(cost_key(e.insn, e.mode if e.mode is not None else 1))
                                       for e in self.out if e.insn is not None)

    def symbols(self) -> dict:
        return {w.name: w.addr for w in self.words if w.kind in ("state", "param")}

    # ------------------------------------------------------------------ running it
    def machine(self):
        from ..emu import Machine
        m = Machine(self.chip)
        m.load(bytes(self.image), 0)
        m.entry = self.entry
        return m

    def emulate(self, samples, params: dict | None = None):
        """One pass per sample in the emulator: the packets, interrupts and cycles."""
        from ..sim import SimResult
        m = self.machine()
        for name, value in (params or {}).items():
            addr = self.symbols()[name]
            m.poke_words(addr, [value if isinstance(value, int) else round(value * ONE)])
        res = SimResult(self.ir.name, f"emu:{self.chip}", self.ir.packet)
        for s in samples:
            r = m.run_pass(s)
            res.packets.append(r.fifo)
            res.irqs.append(r.ints > 0)
            res.cycles.append(r.cycles)
        return res

    # ------------------------------------------------------------------ text
    def listing(self) -> str:
        chip = get_chip(self.chip)
        o = self.ir.opts
        lines = [f"; {self.ir.name}  opendmp  chip {self.chip} ({chip.clock_hz / 1e6:.3f} MHz)  "
                 f"entry {self.entry:#06x}  {o['rate']:g} Hz",
                 "; addr  bytes     asm                    effect                                   cycles"]
        last = None
        for e in self.out:
            if e.loc is not None and (last is None or (e.loc.file, e.loc.line) != (last.file, last.line)):
                src = linecache.getline(e.loc.file, e.loc.line).strip()
                lines.append(f"        ; {pathlib.Path(e.loc.file).name}:{e.loc.line}  {src}")
                last = e.loc
            if e.insn is None:
                lines.append(f"      {e.label}:")
                continue
            cyc = chip.cyc(cost_key(e.insn, e.mode if e.mode is not None else 1))
            text = self.text(e)
            eff = e.note or effect(e.insn)
            lines.append(f"{e.addr:04x}  {e.bytes.hex(' '):9s} {text:22s} {eff:40s} {cyc:5d}")
        lines.append(self.data_listing())
        pk = "   ".join(f"{f.name} {'i16' if f.type == 's16' else f.type} [{f.offset}:{f.offset + f.width}]"
                       for f in self.ir.packet)
        lines.append(f"; packet {self.ir.packet_size} B   {pk}")
        straight = self.cycles_straight()
        period = 1e6 / o["rate"]
        us = straight * chip.cycle_ns / 1000
        if self.bounded:
            lines.append(f"; pass   {self.size} B of code; at most {straight} cycles = {us:.1f} us after the "
                         f"tick ({100 * us / period:.1f} % of {period:.0f} us)"
                         + (f"; {self.hops} branch hops" if self.hops else ""))
        else:
            lines.append(f"; pass   {self.size} B of code; every instruction once: {straight} cycles = "
                         f"{us:.1f} us; loops and calls run longer (the emulator counts them)"
                         + (f"; {self.hops} branch hops" if self.hops else ""))
        return "\n".join(lines)

    def text(self, e):
        i = e.insn
        if e.target is not None:
            return f"{i.op} {e.target}"
        return i.text()

    def data_listing(self) -> str:
        rows = []
        for w in sorted((w for w in self.words if w.addr is not None and w.kind != "sample"),
                        key=lambda w: w.addr):
            val = "" if w.kind == "temp" else f" = {w.init & MASK:08x} ({w.init / ONE:.6g})"
            rows.append(f";   {w.addr:#05x} {w.kind:5s} {w.name}{val}")
        return "; data\n" + "\n".join(rows)

    def asm(self) -> str:
        """opendmp assembly that reassembles to this image (python -m opendmp asm)."""
        names = {}
        for w in self.words:
            if w.addr is not None and w.kind != "sample":
                base = re.sub(r"\W", "_", w.name.lstrip("=")) or "w"
                base = ("k_" if w.kind == "const" else "") + (base if base[0].isalpha() or base[0] == "_"
                                                              else "_" + base)
                n, k = base, 1
                while n in names.values():
                    k += 1
                    n = f"{base}_{k}"
                names[w] = n
        out = [f"; {self.ir.name} for the {self.chip}: opendmp build output", f".chip {self.chip}",
               f".entry {self.entry:#06x}", "", "; data"]
        prev = None
        for w in sorted(names, key=lambda w: w.addr):
            if prev is None or w.addr != prev + 4:
                out.append(f"        .org {w.addr:#06x}")
            out.append(f"{names[w]}: .dw {w.init & MASK:#010x}      ; {w.kind} {w.name}")
            prev = w.addr
        out += ["", "; code", f"        .org {self.entry:#06x}"]
        last = None
        for e in self.out:
            if e.loc is not None and (last is None or (e.loc.file, e.loc.line) != (last.file, last.line)):
                out.append(f"        ; {pathlib.Path(e.loc.file).name}:{e.loc.line}  "
                           f"{linecache.getline(e.loc.file, e.loc.line).strip()}")
                last = e.loc
            if e.insn is None:
                out.append(f"{e.label}:")
            else:
                note = e.note or effect(e.insn)
                out.append(f"        {self.text(e):24s} ; {note}")
        return "\n".join(out) + "\n"

    def schema(self) -> dict:
        o = self.ir.opts
        return {"program": self.ir.name, "chip": self.chip, "entry": self.entry, "rate": o["rate"],
                "gyro_fs": o["gyro_fs"], "accel_fs": o["accel_fs"],
                "packet": [{"name": f.name, "type": f.type, "offset": f.offset, "width": f.width}
                           for f in self.ir.packet],
                "packet_size": self.ir.packet_size,
                "symbols": {w.name: {"addr": w.addr, "kind": w.kind, "type": w.var.type if w.var else None,
                                     "init": w.init} for w in self.words
                            if w.kind in ("state", "param")},
                "code_size": self.size, "cycles_once": self.cycles_straight(),
                "cycles_max": self.cycles_straight() if self.bounded else None}

    def write(self, directory, stem=None) -> list:
        d = pathlib.Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        stem = stem or f"{self.ir.name}.{self.chip}"
        files = {f"{stem}.lst": self.listing(), f"{stem}.asm": self.asm(),
                 f"{stem}.json": json.dumps(self.schema(), indent=2)}
        for name, text in files.items():
            (d / name).write_text(text)
        end = self.entry + self.size
        (d / f"{stem}.bin").write_bytes(bytes(self.image[:end]))
        return [d / n for n in list(files) + [f"{stem}.bin"]]


def size_of(e: Emitted, chip) -> int:
    op = e.insn.op
    if op == "branch.m4":
        return 1
    if op == "branch":
        return 2
    if op in ("jump", "call"):
        return 3
    return len(encode(e.insn, chip))


def mode_at(out: list, j: int):
    """The mode register in force just before out[j], from the instruction before it."""
    from ..isa.core import F4_OPS
    for k in range(j - 1, -1, -1):
        e = out[k]
        if e.insn is None:
            continue
        if e.insn.op == "fmode":
            return e.insn.args[0]
        if e.insn.op in F4_OPS:
            return 4
        return e.mode
    return None


def assemble(out: list, chip: str, entry: int):
    """Addresses, branch relaxation and the bytes.

    An MPU branch reaches 31 bytes forward and 32 back (a signed 6-bit field, probe101).  A
    longer one goes through a hop: a branch placed
    where no predicate is armed, which the code falling through steps over.  Far branches to
    the same target share hops (a branch takes the furthest hop to its target it can reach), so
    a target far away costs one hop every ~60 bytes, not one per branch."""
    icm = chip == "icm20948"
    for e in out:
        # in mode 4 (after any f4) an MPU branch is one byte: the f4 is implied (Finding 42)
        if not icm and e.insn is not None and e.insn.op == "branch" and e.mode == 4:
            e.insn = Insn("branch.m4", e.insn.args)
        e.size = 0 if e.insn is None else size_of(e, chip)
    hops = 0
    hop_target: dict = {}                      # hop label -> the label it continues to
    for _ in range(100000):
        addr, labels = entry, {}
        for e in out:
            e.addr = addr
            if e.insn is None:
                labels[e.label] = addr
            addr += e.size
        if icm:
            break
        far = None
        for i, e in enumerate(out):
            if e.target is not None:
                d = labels[e.target] - (e.addr + e.size)
                if not -32 <= d <= 31:
                    far = i
                    break
        if far is None:
            break
        e = out[far]
        final = hop_target.get(e.target, e.target)
        back = labels[e.target] < e.addr
        lo_reach, hi_reach = e.addr + e.size - 32, e.addr + e.size + 31
        shared = [h for h, t in hop_target.items() if t == final and lo_reach <= labels[h] <= hi_reach
                  and (labels[h] < e.addr if back else labels[h] > e.addr + e.size)]
        if shared:                         # the hop to the same target that gets furthest
            e.target = (min if back else max)(shared, key=lambda h: labels[h])
            e.insn = Insn(e.insn.op, (e.target,))
            continue
        if back:                           # a hop at the earliest safe point it can reach, allowing
            # for the 5 bytes the hop itself adds between it and the branch
            cands = [k for k in range(far) if out[k].safe and out[k].addr + 2 >= lo_reach + 5
                     and out[k].addr >= labels[final]]
            j = cands[0] if cands else None
        else:
            cands = [k for k in range(far + 1, len(out)) if out[k].safe and out[k].addr + 2 <= hi_reach]
            j = cands[-1] if cands else None
        if j is None:
            raise DmpError(f"a branch to {e.target} spans more than 32 bytes with no place for a hop")
        hops += 1
        hop, over = f"hop.{hops}", f"over.{hops}"
        loc = out[j].loc
        m = mode_at(out, j)
        ins = [Emitted(Insn("branch.m4" if m == 4 else "branch", (over,)), target=over, safe=True,
                       loc=loc, note="step over the hop", mode=m),
               Emitted(None, label=hop, loc=loc),
               Emitted(Insn("branch.m4", (final,)), target=final, loc=loc,     # one byte: mode 4
                       note=f"hop on to {final}", mode=4),
               Emitted(None, label=over, loc=loc)]
        if m not in (None, 4):             # the step-over f4 left mode 4: restore what was in force
            ins.append(Emitted(Insn("fmode", (m,)), loc=loc, note="restore the mode after the hop",
                               mode=4))
        for x in ins:
            x.size = 0 if x.insn is None else size_of(x, chip)
        out[j:j] = ins
        hop_target[hop] = final
        e.target = hop
        e.insn = Insn(e.insn.op, (hop,))
    else:
        raise DmpError("internal: branch relaxation did not settle")
    code = bytearray()
    for e in out:
        if e.insn is None:
            continue
        insn = Insn(e.insn.op, (labels[e.target],)) if e.target is not None else e.insn
        e.bytes = encode(insn, chip, e.addr)
        code += e.bytes
    return bytes(code), labels, hops


def build(program, chip, trace: bool = False) -> Build:
    chip = chip_name(chip)
    ir = program.ir
    sel = T.Selector(ir, chip, trace)
    ops = optimize(sel.run())
    used = {id(w) for op in ops for w in uses(op) + defs(op)}
    words = optimise([w for w in sel.words if id(w) in used or w.kind in ("state", "param")], ops, chip)
    out = schedule(ops, chip)
    entry = ENTRY[chip]
    code, labels, hops = assemble(out, chip, entry)
    if entry + len(code) > CODE_END[chip]:
        raise DmpError(f"{ir.name} needs {len(code)} bytes of code; the {chip} has "
                       f"{CODE_END[chip] - entry} from the entry point")
    image = bytearray(MEM[chip])
    for w in words:
        if w.kind in ("state", "param", "const"):
            image[w.addr:w.addr + 4] = (w.init & MASK).to_bytes(4, "big")
    image[entry:entry + len(code)] = code
    return Build(program, ir, chip, entry, image, code, out, words, labels, hops, trace)
