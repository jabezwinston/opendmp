"""opendmp assembly: parse, assemble, disassemble.

The syntax is one instruction per line, as `Insn.text()` prints it, plus:

    .chip mpu6050            the target (or icm20948); required before the first instruction
    .entry 0x400             where PRGM_START points (default: the chip's)
    .org 0x100               continue at this address
    label:                   a label, alone or before an instruction
    .dw 0.25q, -1.0q, 0x7fffffff    32-bit big-endian words (Q30 with a q suffix)
    .db 0x12, 0x34           raw bytes
    .mode 1                  assert that the mode register holds 1 here (emits nothing)
    ; comment

Branch, jump and call targets are labels or absolute addresses (`branch 0x042b`); a
disassembly prints absolute addresses, so it reassembles to the same bytes.

The assembler tracks the mode register as the code falls through, and refuses an ALU mnemonic
from the wrong family (`min` needs f5-f7, `mac` needs f0, `add` needs f1-f3) and, on DMP3, any
ALU op while mode 4 is set.  At a label that a branch, jump or call reaches, the mode is
whatever the fall-through path left only if every path agrees; otherwise it is unknown and the
code there must set it before its first ALU op.
"""
import re
from dataclasses import dataclass, field

from ..q30 import parse_word
from .core import ALU_OPS, FAMILY_MODES, ICM, MPU, EncodeError, Insn, chip_name, decode, encode, \
    length, next_mode  # noqa: F401 (length is re-exported)

DEFAULT_ENTRY = {MPU: 0x0400, ICM: 0x1000}
MEM_SIZE = {MPU: 0xC00, ICM: 0x4000}
_TARGETED = ("branch", "jump", "call", "jump.m4", "call.m4")


class AsmError(Exception):
    pass


@dataclass
class Line:
    addr: int
    insn: Insn | None          # None for data
    data: bytes
    lineno: int = 0
    mode: int | None = None    # the mode in force before this line (code only)
    source: str = ""


@dataclass
class Program:
    chip: str
    entry: int
    lines: list = field(default_factory=list)
    symbols: dict = field(default_factory=dict)

    def segments(self) -> list[tuple[int, bytes]]:
        """Contiguous (address, bytes) runs, in address order."""
        out: list[list] = []
        for ln in sorted((x for x in self.lines if x.data), key=lambda x: x.addr):
            if out and out[-1][0] + len(out[-1][1]) == ln.addr:
                out[-1][1] += ln.data
            else:
                out.append([ln.addr, bytearray(ln.data)])
        return [(a, bytes(d)) for a, d in out]

    def image(self, size: int | None = None) -> bytearray:
        mem = bytearray(size or MEM_SIZE[self.chip])
        for a, d in self.segments():
            mem[a:a + len(d)] = d
        return mem

    def code(self) -> bytes:
        """The bytes of the segment that holds the entry point."""
        for a, d in self.segments():
            if a <= self.entry < a + len(d):
                return d[self.entry - a:]
        return b""


def _int(tok: str) -> int:
    return int(tok.strip(), 0)


def _parse_args(op: str, arg: str):
    toks = [t.strip() for t in arg.split(",")] if arg.strip() else []
    if op in ALU_OPS:
        if len(toks) not in (2, 3) or (len(toks) == 3 and toks[2].lower() not in ("inc", "adv")):
            raise AsmError(f"{op} takes A,B or A,B,inc")
        return (_int(toks[0]), _int(toks[1]), len(toks) == 3)
    if op in ("mov", "movx", "mov16"):
        if len(toks) not in (1, 2) or (len(toks) == 2 and toks[1].lower() not in ("inc", "adv")):
            raise AsmError(f"{op} takes k or k,inc")
        return (_int(toks[0]), len(toks) == 2)
    if op in ("shl", "sar"):
        if len(toks) not in (1, 2) or (len(toks) == 2 and toks[1].lower() not in ("adv", "inc")):
            raise AsmError(f"{op} takes s or s,adv")
        return (_int(toks[0]), len(toks) == 2)
    if op in _TARGETED:
        if len(toks) != 1:
            raise AsmError(f"{op} takes one target")
        t = toks[0]
        return (t,) if re.fullmatch(r"[A-Za-z_][\w.]*", t) else (_int(t),)
    if op == ".db":
        return tuple(_int(t) & 0xFF if _int(t) >= 0 else _int(t) for t in toks)
    return tuple(_int(t) for t in toks)


def parse(text: str, chip=None) -> list[tuple[int, str, str, str | None]]:
    """(lineno, op, arg, label) per source line with content."""
    out = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split(";")[0].strip()
        if not line:
            continue
        label = None
        m = re.match(r"([A-Za-z_][\w.]*):\s*(.*)$", line)
        if m:
            label, line = m.group(1), m.group(2).strip()
        if not line:
            out.append((lineno, "", "", label))
            continue
        parts = line.split(None, 1)
        op, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
        if op == "if":                              # `if f` / `if nf` / `if f.alt`
            op, arg = f"if {arg.lower()}", ""
        op = {"disable": "end", "mov16": "movx", "swap": "xchg01"}.get(op, op)
        out.append((lineno, op, arg, label))
    return out


def assemble(text: str, chip=None) -> Program:
    """Assemble opendmp assembly.  `chip` overrides/replaces a `.chip` directive."""
    rows = parse(text)
    chip = chip_name(chip) if chip else None
    entry = None
    for _, op, arg, _ in rows:                     # directives that fix the target first
        if op == ".chip" and chip is None:
            chip = chip_name(arg)
        elif op == ".entry":
            entry = _int(arg)
    if chip is None:
        raise AsmError("no .chip directive and no chip given")
    prog = Program(chip, DEFAULT_ENTRY[chip] if entry is None else entry)

    # pass 1: addresses and labels
    pc, items = prog.entry, []
    for lineno, op, arg, label in rows:
        try:
            if label:
                if label in prog.symbols:
                    raise AsmError(f"duplicate label {label!r}")
                prog.symbols[label] = pc
            if not op or op in (".chip", ".entry"):
                continue
            if op == ".org":
                pc = _int(arg)
                continue
            if op == ".mode":                   # assert the mode register here (emits nothing)
                m = _int(arg)
                if m not in (0, 1, 2, 3, 4, 5, 6, 7):
                    raise AsmError(".mode takes 0-7")
                if label:
                    raise AsmError("put the label on its own line before .mode")
                items.append((lineno, pc, None, None, m))
                continue
            if op == ".dw":
                data = b"".join((parse_word(t.strip()) & 0xFFFFFFFF).to_bytes(4, "big")
                                for t in arg.split(","))
                items.append((lineno, pc, None, data, label))
                pc += len(data)
                continue
            insn = Insn(op, _parse_args(op, arg))
            n = (2 if insn.op in ("branch", "jump.m4", "call.m4") else 3) if _needs_label(insn) \
                else len(encode(insn, chip, pc) if insn.op != "branch" else b"..")
            items.append((lineno, pc, insn, None, label))
            pc += n
        except (AsmError, EncodeError, ValueError) as e:
            raise AsmError(f"line {lineno}: {e}") from None

    # pass 2: resolve labels, encode, and track the mode register.  A jump target's mode is
    # known only when every way in agrees: the fall-through, each forward jump or branch (the
    # mode at the jump), each backward jump (always mode 4: on DMP3 a jump leaves it set) and
    # each call's return point (unknown: the callee may change it).
    resolved = []
    for lineno, addr, insn, data, label in items:
        if insn is not None and _needs_label(insn) and insn.args[0] not in prog.symbols:
            raise AsmError(f"line {lineno}: undefined label {insn.args[0]!r}")
        resolved.append((lineno, addr, None if insn is None else _resolve(insn, prog.symbols),
                         data, label))
    backward, calls = set(), set()
    for lineno, addr, insn, data, label in resolved:
        if insn is not None and insn.op in _TARGETED:
            if insn.args[0] <= addr:
                backward.add(insn.args[0])
            if insn.op.startswith("call"):
                calls.add(addr + (2 if insn.op == "call.m4" else 3))
    arriving: dict[int, set] = {}                   # address -> modes carried there by forward jumps
    mode: int | None = None                         # unknown at the entry, unless `.mode` says
    asserted = None                                 # a `.mode` here overrides the join rule
    for lineno, addr, insn, data, label in resolved:
        try:
            if addr != asserted and (addr in arriving or addr in backward or addr in calls):
                ways = set(arriving.get(addr, ())) | {mode}
                if addr in backward:
                    ways.add(4 if chip == ICM else None)
                if addr in calls:
                    ways.add(None)
                mode = ways.pop() if len(ways) == 1 else None
            if insn is None:
                if data is None:                    # a .mode assertion
                    mode, asserted = label, addr
                    continue
                prog.lines.append(Line(addr, None, data, lineno))
                continue
            _check_mode(insn, chip, mode)
            code = encode(insn, chip, addr)
            prog.lines.append(Line(addr, insn, code, lineno, mode, insn.text()))
            after = next_mode(insn, chip, mode) if mode is not None else \
                (insn.args[0] if insn.op == "fmode" else
                 4 if chip == ICM and insn.op in ("jump", "call", "mode4") else None)
            if insn.op in _TARGETED and insn.args[0] > addr:
                arriving.setdefault(insn.args[0], set()).add(
                    4 if chip == ICM else mode)
            mode = after
        except (AsmError, EncodeError) as e:
            raise AsmError(f"line {lineno}: {e}") from None
    _check_following(prog)
    return prog


def _needs_label(insn: Insn) -> bool:
    return insn.op in _TARGETED and isinstance(insn.args[0], str)


def _resolve(insn: Insn, symbols: dict) -> Insn:
    if _needs_label(insn):
        return Insn(insn.op, (symbols[insn.args[0]],) + insn.args[1:])
    return insn


def _check_mode(insn: Insn, chip: str, mode: int | None):
    if insn.op in ALU_OPS:
        fam = ALU_OPS[insn.op][0]
        if mode is None:
            raise AsmError(f"{insn.text()}: the mode register is unknown here (at the entry, or "
                           "at a label reached with different modes); set it with fmode first")
        if chip == ICM and mode == 4:
            raise AsmError(f"{insn.text()} in mode 4: a byte below 0x80 is a jump target here; "
                           "end mode 4 with an fmode first")
        if mode not in FAMILY_MODES[fam]:
            need = {"mac": "f0", "arith": "f1-f3", "logic": "f5-f7"}[fam]
            raise AsmError(f"{insn.op} needs mode {need}, but fmode {mode} is in force")
    if insn.op in ("jump.m4", "call.m4") and mode != 4:
        raise AsmError(f"{insn.op} is a mode-4 target form; mode 4 is not in force")
    if chip == ICM and mode == 4 and insn.op == ".db" and insn.args and insn.args[0] < 0x80:
        raise AsmError(".db below 0x80 in mode 4 would decode as a jump target")


def _check_following(prog: Program):
    """A bare f4 (DMP3 `mode4`, MPU `nop.f4`) is one byte only if the next byte is >= 0x80."""
    code = sorted((ln for ln in prog.lines if ln.insn is not None), key=lambda x: x.addr)
    mem = {}
    for ln in prog.lines:
        for i, v in enumerate(ln.data):
            mem[ln.addr + i] = v
    for ln in code:
        if ln.insn.op in ("mode4", "nop.f4"):
            nxt = mem.get(ln.addr + 1)
            if nxt is None or nxt < 0x80:
                raise AsmError(f"line {ln.lineno}: {ln.insn.op} must be followed by a byte >= 0x80 "
                               "(otherwise f4 reads it as an operand)")


def disassemble(data: bytes, chip, org: int = 0, mode: int = 1) -> list[Line]:
    """Linear sweep from `org`, tracking the mode register as the code falls through."""
    chip = chip_name(chip)
    buf = bytes(data)
    out, off = [], 0
    while off < len(buf):
        insn, n = decode(buf, off, chip, mode)
        if insn.op == "branch":                     # decode() counts from the buffer's start
            insn = Insn("branch", (org + insn.args[0],))
        out.append(Line(org + off, insn, buf[off:off + n], mode=mode, source=insn.text()))
        mode = next_mode(insn, chip, mode)
        off += n
    return out


def to_text(lines: list[Line], chip, entry: int | None = None) -> str:
    """Assembly text for `lines` (from `disassemble`), which reassembles to the same bytes."""
    chip = chip_name(chip)
    out = [f".chip {chip}"]
    if entry is not None:
        out.append(f".entry {entry:#06x}")
    # where the assembler cannot know the mode, state the one the linear decode assumed
    joins = set()
    for ln in lines:
        if ln.insn is not None and ln.insn.op in _TARGETED:
            joins.add(ln.insn.args[0])
            if ln.insn.op.startswith("call"):
                joins.add(ln.addr + len(ln.data))
    pc = None
    for ln in lines:
        if pc != ln.addr:
            out.append(f".org {ln.addr:#06x}")
        if ln.mode is not None and (pc != ln.addr or ln.addr in joins):
            out.append(f".mode {ln.mode}")
        text = ln.insn.text() if ln.insn else ".db " + ",".join(f"{v:#04x}" for v in ln.data)
        out.append(f"        {text}")
        pc = ln.addr + len(ln.data)
    return "\n".join(out) + "\n"


__all__ = ["AsmError", "Line", "Program", "assemble", "disassemble", "parse", "to_text", "length"]
