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
    hop_target: dict = field(default_factory=dict)    # hop label -> the label it continues to
    notes: list = field(default_factory=list)          # [(Loc, text)]: the method used, per construct
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
            s = Sample(ax=h(w[0], 0), ay=h(w[0], 1), az=h(w[1], 0), temp=h(w[1], 1),
                       gx=h(w[2], 0), gy=h(w[2], 1), gz=h(w[3], 0))
        else:
            s = Sample(ax=h(w[0], 0), ay=h(w[0], 1), az=h(w[1], 0), gx=h(w[1], 1),   # ax|ay az|gx
                       gy=h(w[2], 0), gz=h(w[2], 1), temp=h(w[3], 0))                 # gy|gz temp|-
        if self.fsync:
            s.fsync = s.temp & 1             # the temperature's bit 0 carries it
        return s

    @property
    def fsync(self) -> bool:
        """The program reads s.fsync: the runtime sets EXT_SYNC_SET = 1, which puts the FSYNC
        level in bit 0 of the temperature sample."""
        return "fsync" in self.ir.uses

    @property
    def bounded(self) -> bool:
        """True when a pass has an upper bound: every loop has a trip count (no `while`)."""
        return self.cycles_max() is not None

    def cycles_max(self):
        """An upper bound on a pass, in cycles after the sample tick, or None when a `while`
        loop leaves it open.  The longest path through the code as emitted, where:
          - a branch costs its taken cost, and a far branch the hops on the way;
          - an instruction a false predicate skips is charged as executed (which costs more);
          - a loop costs its trip count times its costliest trip, a break counting as a trip;
          - a call costs the routine's own bound.
        The MPU's sample-write stalls (14-28 cycles each, only in a pass that overruns its
        period) are not included."""
        if "_cmax" not in self.__dict__:
            self.__dict__["_cmax"] = _wcet(self)
        return self.__dict__["_cmax"]

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
        m.fsync_enabled = self.fsync
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
        from ..lang.errors import snippet
        by_line = {}
        for loc, text in self.notes:
            by_line.setdefault((loc.file, loc.line), []).append(f"`{snippet(loc, 36)}`: {text}")
        shown, last = set(), None
        for e in self.out:
            if e.loc is not None and (last is None or (e.loc.file, e.loc.line) != (last.file, last.line)):
                src = linecache.getline(e.loc.file, e.loc.line).strip()
                lines.append(f"        ; {pathlib.Path(e.loc.file).name}:{e.loc.line}  {src}")
                key = (e.loc.file, e.loc.line)
                if key not in shown:
                    shown.add(key)
                    lines += [f"        ;   -> {n}" for n in by_line.get(key, [])]
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
        lines.append("; pass   " + self.budget() + (f"; {self.hops} branch hops" if self.hops else ""))
        return "\n".join(lines)

    def explain(self) -> list:
        """One line per construct: where it is, what it says, and how it was compiled."""
        from ..lang.errors import snippet
        return [f"{pathlib.Path(loc.file).name}:{loc.line}  `{snippet(loc)}`: {text}" for loc, text in self.notes]

    def budget(self) -> str:
        """The pass's cost against the sample period, in words."""
        chip = get_chip(self.chip)
        period = 1e6 / self.ir.opts["rate"]
        bound = self.cycles_max()
        if bound is None:
            once = self.cycles_straight()
            return (f"{self.size} B of code; no bound: a while loop runs as long as its condition "
                    f"holds (every instruction once: {once} cycles = {once * chip.cycle_ns / 1000:.1f} us)")
        us = bound * chip.cycle_ns / 1000
        loops = any(e.loop_head for e in self.out if e.insn is None) or \
            any(e.insn is not None and e.insn.op == "call" for e in self.out)
        text = (f"{self.size} B of code; at most {bound} cycles = {us:.1f} us after the tick "
                f"({100 * us / period:.1f} % of {period:.0f} us)" + ("; loops at their trip counts" if loops else ""))
        if us > period:
            text += ("; OVER THE PERIOD: a pass can overrun, and " +
                     ("the MPU then drops to a lower rate" if self.chip == "mpu6050" else "the ICM runs slower"))
        return text

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
                "cycles_max": self.cycles_max()}

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


def _wcet(b: "Build"):
    """Worst-case cycles of a pass (Build.cycles_max): the longest path through the code as
    emitted, loops collapsed innermost first, routines leaf first."""
    chip = get_chip(b.chip)
    out = b.out
    ins = [i for i, e in enumerate(out) if e.insn is not None]
    n = len(ins)
    at_label, info, k = {}, {}, 0
    for e in out:                                   # a label -> the instruction after it
        if e.insn is None:
            at_label[e.label], info[e.label] = k, e
        else:
            k += 1
    E = [out[i] for i in ins]
    call_extra = {}                                 # instruction -> its routine's bound

    def cost(k):
        e = E[k]
        return chip.cyc(cost_key(e.insn, e.mode if e.mode is not None else 1)) + call_extra.get(k, 0)

    def final(label):                               # through a chain of hops
        extra = 0
        while label in b.hop_target:
            extra += cost(at_label[label])
            label = b.hop_target[label]
        return label, extra

    # successors [(instruction, extra cycles on the edge)]; a branch under an armed predicate
    # (an `if` not yet ended by `always`) may also fall through; backward branches close loops
    succ, back, armed = [], {}, False
    for k in range(n):
        op = E[k].insn.op
        armed = True if op in ("if f", "if nf") else False if op == "always" else armed
        s = []
        if op in ("end", "ret"):
            pass
        elif E[k].target is not None and op != "call":
            lab, extra = final(E[k].target)
            t = at_label[lab]
            if t <= k:
                head = info[lab]
                if not head.loop_head or head.trips is None:
                    return None                     # a while loop: the pass has no bound
                back[t] = (max(back.get(t, (0, k))[0], k), head.trips)
            else:
                s.append((t, extra))
            if armed:
                s.append((k + 1, 0))
        else:
            s.append((k + 1, 0))
        succ.append([(t, x) for t, x in s if t < n])
    loops = {h: end for h, (end, _) in back.items()}
    collapsed = {}                                  # loop head -> (cost, exits)

    def longest(start, end, stops):
        """The costliest path from `start` within [start, end] to a node in `stops` or out of the
        region, collapsed inner loops counted whole; and where it can leave."""
        dist, best, exits = {start: 0}, 0, []
        k = start
        while k <= end:
            inner = k in collapsed and k != start
            if k in dist:
                c, nexts = collapsed[k] if inner else (cost(k), succ[k])
                here = dist[k] + c
                if k in stops:
                    best = max(best, here)
                for t, x in nexts:
                    if start < t <= end and t > k:
                        dist[t] = max(dist.get(t, -1), here + x)
                    elif not start <= t <= end:
                        best = max(best, here + x)
                        exits.append(t)
            k = loops[k] + 1 if inner else k + 1
        return best, exits

    def region(start, end):
        """A region's bound: its calls first, then its loops innermost first."""
        for k in range(start, end + 1):
            if E[k].insn.op == "call":
                call_extra[k] = routine(E[k].target)
        for h in sorted((h for h in loops if start <= h <= end), key=lambda h: loops[h] - h):
            trip, exits = longest(h, loops[h], {loops[h]})
            collapsed[h] = (back[h][1] * trip, [(t, 0) for t in exits] + [(loops[h] + 1, 0)])
        return longest(start, end, {end})[0]

    bounds = {}

    def routine(label):                             # entry .. its ret, each routine once
        if label not in bounds:
            start = at_label[label]
            end = next(k for k in range(start, n) if E[k].insn.op == "ret")
            bounds[label] = region(start, end)
        return bounds[label]

    end_main = next(k for k in range(n) if E[k].insn.op == "end")
    return chip.start_cycles + region(0, end_main)


def size_of(e: Emitted, chip) -> int:
    op = e.insn.op
    if op == "branch.m4":
        return 1
    if op == "branch":
        return 2
    if op in ("jump", "call"):
        return 3
    return len(encode(e.insn, chip))


def dead_points(out: list) -> list:
    """Indices k where nothing falls through into out[k]: the instruction before it (with no
    label between) is an unconditional branch or jump, the end of the pass, or a return."""
    dead, armed = [], False
    prev = None                                  # the previous entry, label or instruction
    for k, e in enumerate(out):
        if prev is not None and prev.insn is not None and (
                prev.insn.op in ("end", "ret") or
                (prev.insn.op in ("branch", "branch.m4", "jump") and not prev_armed)):
            dead.append(k)
        if e.insn is not None:
            prev_armed = armed
            if e.insn.op in ("if f", "if nf"):
                armed = True
            elif e.insn.op == "always":
                armed = False
        prev = e
    return dead


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
    longer one goes through a hop, a one-byte branch on to the target (Finding 42):
      - best, at a dead point, just after an unconditional branch or the end, where no code
        falls through: the hop costs one byte and nothing on any other path;
      - else where no predicate is armed, with a branch that steps the code falling through
        over it (and restores the mode that step-over's f4 disturbed).
    Far branches to the same target share hops (a branch takes the furthest hop to its target
    it can reach), so a target far away costs one hop every ~60 bytes, not one per branch."""
    icm = chip == "icm20948"
    for e in out:
        # in mode 4 (after any f4) an MPU branch is one byte: the f4 is implied (Finding 42)
        if not icm and e.insn is not None and e.insn.op == "branch" and e.mode == 4:
            e.insn = Insn("branch.m4", e.insn.args)
        e.size = 0 if e.insn is None else size_of(e, chip)
    hops = 0
    hop_target: dict = {}                      # hop label -> the label it continues to
    notes = []
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
        if e.target not in hop_target and e.loc is not None:
            notes.append((e.loc, "a branch from here reaches past the MPU's +-32 bytes: it goes on through "
                                 "hops (one-byte branches, Finding 42)"))
        back = labels[e.target] < e.addr
        lo_reach, hi_reach = e.addr + e.size - 32, e.addr + e.size + 31
        shared = [h for h, t in hop_target.items() if t == final and lo_reach <= labels[h] <= hi_reach
                  and (labels[h] < e.addr if back else labels[h] > e.addr + e.size)]
        if shared:                         # the hop to the same target that gets furthest
            e.target = (min if back else max)(shared, key=lambda h: labels[h])
            e.insn = Insn(e.insn.op, (e.target,))
            continue
        dead = dead_points(out)
        pick = None
        for need, pool in ((1, dead), (5, [k for k in range(len(out)) if out[k].safe])):
            # a hop at a dead point adds 1 byte, a live one 5 (the step-over, the hop, a mode byte)
            # a hop must bring the branch at least 4 bytes nearer its target, or relaxation
            # would only move the problem (a dead point right after the branch itself, say)
            if back:                       # the earliest point it can reach
                cands = [k for k in pool if k < far and out[k].addr + 2 >= lo_reach + need
                         and out[k].addr >= labels[final] and out[k].addr <= e.addr - 4]
                j = cands[0] if cands else None
            else:                          # the furthest point it can reach
                cands = [k for k in pool if k > far and out[k].addr + need <= hi_reach + 1
                         and out[k].addr >= e.addr + e.size + 4]
                j = cands[-1] if cands else None
            if j is not None:
                pick = (j, pool is dead)
                break
        if pick is None:
            raise DmpError(f"a branch to {e.target} spans more than 32 bytes with no place for a hop")
        j, at_dead = pick
        hops += 1
        hop, over = f"hop.{hops}", f"over.{hops}"
        loc = out[j].loc
        m = mode_at(out, j)
        hop_insn = Emitted(Insn("branch.m4", (final,)), target=final, loc=loc,      # one byte: mode 4
                           note=f"hop on to {final}", mode=4)
        if at_dead:                        # nothing falls through here: no step-over needed
            ins = [Emitted(None, label=hop, loc=loc), hop_insn]
        else:
            ins = [Emitted(Insn("branch.m4" if m == 4 else "branch", (over,)), target=over, safe=True,
                           loc=loc, note="step over the hop", mode=m),
                   Emitted(None, label=hop, loc=loc), hop_insn,
                   Emitted(None, label=over, loc=loc)]
            if m not in (None, 4):         # the step-over f4 left mode 4: restore what was in force
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
    return bytes(code), labels, hops, hop_target, notes


def build(program, chip, trace: bool = False) -> Build:
    chip = chip_name(chip)
    ir = program.ir
    sel = T.Selector(ir, chip, trace)
    ops = optimize(sel.run())
    used = {id(w) for op in ops for w in uses(op) + defs(op)}
    words = optimise([w for w in sel.words if id(w) in used or w.kind in ("state", "param")], ops, chip)
    out = schedule(ops, chip)
    entry = ENTRY[chip]
    code, labels, hops, hop_target, asm_notes = assemble(out, chip, entry)
    if entry + len(code) > CODE_END[chip]:
        raise DmpError(f"{ir.name} needs {len(code)} bytes of code; the {chip} has "
                       f"{CODE_END[chip] - entry} from the entry point")
    image = bytearray(MEM[chip])
    for w in words:
        if w.kind in ("state", "param", "const"):
            image[w.addr:w.addr + 4] = (w.init & MASK).to_bytes(4, "big")
    image[entry:entry + len(code)] = code
    b = Build(program, ir, chip, entry, image, code, out, words, labels, hops, hop_target, trace=trace)
    b.notes = _merge_notes(ir.notes + sel.notes + asm_notes)
    return b


def _merge_notes(notes):
    """One entry per construct and method, in source order; not opendmp's own library code."""
    from ..lang.errors import library_file
    seen, out = set(), []
    for loc, text in sorted(notes, key=lambda n: (n[0].file, n[0].line, n[0].col)):
        if library_file(loc.file):
            continue
        key = (loc.file, loc.line, loc.col, text)
        if key not in seen:
            seen.add(key)
            out.append((loc, text))
    return out
