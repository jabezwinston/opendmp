"""The two instruction sets as tables: one instruction model, an encoder and a decoder.

Every byte value decodes to exactly one canonical instruction, and every instruction encodes to
exactly one byte string, so `decode` and `encode` are inverses.  The tests check this against
the vendor images.

Two things make the ISA stateful, and the decoder tracks both:

- **The mode register** (`f0`-`f7`, sticky).  It names the ALU byte: under f1/f2/f3 op 0-3 is
  add/rsub/mul/nmul, under f5/f6/f7 it is min/cmp/and/xor, and under f0 it is mac/macn
  [HW, probe70/74, Finding 30].
- **DMP3 mode 4**, entered by any `f4`.  While it is set, a byte below 0x80 is not an ALU op
  but a two-byte jump or call target, until a width byte ends the mode [HW, probe54b/64].

Duplicates (bytes that do the same thing as a canonical one) keep their byte in the name, for
example `clr.e5` on the MPU, so a disassembly always reassembles to the same bytes.  A byte with
no measured meaning decodes as `.db`.
"""
from dataclasses import dataclass

MPU, ICM = "mpu6050", "icm20948"
_ALIASES = {"mpu": MPU, "mpu6050": MPU, "mantis": MPU, "icm": ICM, "icm20948": ICM, "dmp3": ICM}


def chip_name(chip) -> str:
    name = getattr(chip, "name", chip)
    try:
        return _ALIASES[str(name).lower()]
    except KeyError:
        raise ValueError(f"unknown chip {chip!r} (known: mpu6050, icm20948)") from None


ARITH = ("add", "rsub", "mul", "nmul")                  # f1 f2 f3
LOGIC = ("min", "cmp", "and", "xor")                    # f5 f6 f7   [HW, probe70/86]
MACS = ("mac", "macn", "mac.m", "macn.m")               # f0: acc +/- A*B; op2/3 = op0/1 [HW, F30]
FAMILY = {0: MACS, 1: ARITH, 2: ARITH, 3: ARITH, 5: LOGIC, 6: LOGIC, 7: LOGIC}
ALU_OPS = {name: (fam, i) for fam, names in (("mac", MACS), ("arith", ARITH), ("logic", LOGIC))
           for i, name in enumerate(names)}
FAMILY_MODES = {"mac": {0}, "arith": {1, 2, 3}, "logic": {5, 6, 7}}
COND = ("eq", "gt", "lt", "ne")                         # DMP3 fc cK Y, Y & 3   [HW, probe83b]

# Single-byte instructions with no operands: byte -> name, per chip.
_COMMON = {0xD0: "p2inc2", 0xD1: "test", 0xD8: "always", 0xD9: "if f", 0xDA: "if nf",
           0xDB: "if f.alt", 0xDC: "store", 0xDD: "fifo", 0xDE: "clr", 0xDF: "clr.next",
           0xF8: "inc", 0xF9: "dec", 0xFA: "inc.next", 0xFB: "dec.next", 0xFF: "end",
           0xBC: "p0blk", 0xBD: "p1blk", 0xBE: "p2blk"}
FIXED = {
    MPU: {**_COMMON, 0xFC: "xchg01", 0xFD: "xchg02", 0xFE: "int",
          **{b: f"clr.{b:02x}" for b in range(0xD2, 0xD8)},        # [HW, probe68]
          **{b: f"clr.{b:02x}" for b in range(0xE0, 0xF0)}},       # [HW, probe69]
    ICM: {**_COMMON, 0xD2: "lo", 0xD3: "lo.next", 0xE0: "ret", 0xE2: "int", 0xE7: "clr.e7",
          0xBF: "movz", 0xFE: "int.fe"},
}
FIXED_BYTE = {chip: {name: b for b, name in t.items()} for chip, t in FIXED.items()}

# What each instruction does, for listings (ALU ops and operand forms are described in `effect`).
DOC = {
    "p2inc2": "P2 word += 2", "test": "F := (last result == 0)", "always": "predicate off",
    "if f": "run while F (result < 0)", "if nf": "run while not F", "if f.alt": "as if f (db)",
    "store": "route -> RAM", "fifo": "route -> FIFO", "clr": "M[P2] := 0",
    "clr.next": "P2 word += 1; M[P2] := 0", "inc": "M[P2] += 1 lsb", "dec": "M[P2] -= 1 lsb",
    "inc.next": "P2 word += 1; M[P2] += 1", "dec.next": "P2 word += 1; M[P2] -= 1",
    "end": "end of pass", "p0blk": "P0 += 1 KB", "p1blk": "P1 += 1 KB", "p2blk": "P2 += 1 KB",
    "xchg01": "toggle: A and MOV via P1, B via P0", "xchg02": "toggle: A and MOV via P2, store to P0's page",
    "int": "raise the interrupt", "int.fe": "raise the interrupt (fe, INFER on DMP3)",
    "lo": "store the product's low word", "lo.next": "P2 word += 1; store the product's low word",
    "ret": "return", "movz": "toggle: MOV stores zero", "mode4": "enter mode 4",
    "nop.f4": "f4 with no operand: nothing", "nop.f440": "f4 40: nothing",
}


@dataclass(frozen=True)
class Insn:
    """One instruction.  `op` is the canonical mnemonic; `args` are ints, except that a jump,
    call or branch target may be a label (str) until it is resolved."""
    op: str
    args: tuple = ()

    def text(self) -> str:
        a = self.args
        if self.op == ".db":
            return ".db " + ",".join(f"{v:#04x}" for v in a)
        if self.op in ALU_OPS or self.op in ("mov", "movx"):
            *ops, adv = a
            return f"{self.op} {','.join(str(x) for x in ops)}{',inc' if adv else ''}"
        if self.op in ("shl", "sar"):
            return f"{self.op} {a[0]}{',adv' if a[1] else ''}"
        if self.op in ("branch", "jump", "call", "jump.m4", "call.m4"):
            t = a[0]
            return f"{self.op} {t if isinstance(t, str) else f'{t:#06x}'}"
        if a:
            return f"{self.op} {','.join(str(x) for x in a)}"
        return self.op

    __str__ = text


class EncodeError(ValueError):
    pass


def _chk(cond, msg):
    if not cond:
        raise EncodeError(msg)


def _rng(v, lo, hi, what):
    _chk(isinstance(v, int) and lo <= v <= hi, f"{what} must be {lo}..{hi}, not {v!r}")


def length(buf, pc: int, chip, mode: int = 1) -> int:
    """How many bytes the instruction at pc takes, in the given mode."""
    chip = chip_name(chip)
    b = buf[pc]
    nxt = buf[pc + 1] if pc + 1 < len(buf) else 0xFF
    if chip == ICM:
        if mode == 4 and b < 0x80:
            return 2
        if b == 0xF4:
            return 3 if nxt < 0x80 else 1
        return 3 if b == 0xFC else 2 if b == 0xFD else 1
    if b == 0xF4:
        return 2 if nxt < 0x80 else 1
    return 1


def next_mode(insn: Insn, chip, mode: int) -> int:
    """The mode register after `insn`.  Mode bytes act even under a false predicate, and so
    does DMP3's f4 [HW, probe64], so this holds on every path."""
    if insn.op == "fmode":
        return insn.args[0]
    if chip_name(chip) == ICM and insn.op in ("jump", "call", "mode4"):
        return 4
    return mode


def decode(buf, pc: int, chip, mode: int = 1) -> tuple[Insn, int]:
    """(instruction, length) at pc.  `mode` is the mode register in force (it names the ALU
    byte, and on DMP3 mode 4 turns low bytes into jump targets)."""
    chip = chip_name(chip)
    n = length(buf, pc, chip, mode)
    raw = bytes(buf[pc:pc + n])
    if len(raw) < n:
        return Insn(".db", tuple(raw)), len(raw)
    b = raw[0]
    icm = chip == ICM

    if icm and mode == 4 and b < 0x80:                  # a branch target with no f4 of its own
        t = ((b << 8) | raw[1]) & 0x3FFF
        return Insn("call.m4" if b & 0x40 else "jump.m4", (t,)), 2
    if b < 0x80:
        names = FAMILY.get(mode)
        if names is None:
            return Insn(".db", (b,)), 1
        return Insn(names[b & 3], ((b >> 5) & 3, (b >> 3) & 3, bool(b & 4))), 1
    hi = b >> 4
    if hi in (0x8, 0x9, 0xA):
        return Insn(f"p{hi - 8}page", (b & 0xF,)), 1
    if hi == 0xB and (b >> 2) & 3 < 3:
        return Insn(f"p{(b >> 2) & 3}bank", (b & 3,)), 1
    if hi == 0xC:
        return Insn("movx" if b & 8 else "mov", ((b >> 1) & 3, bool(b & 1))), 1
    if 0xF0 <= b <= 0xF7 and b != 0xF4:
        return Insn("fmode", (b & 7,)), 1
    if icm and 0xD4 <= b <= 0xD7:
        return Insn("load", (b - 0xD4,)), 1
    if b in FIXED[chip]:
        return Insn(FIXED[chip][b]), 1
    if b == 0xF4:
        if icm:
            if n == 1:
                return Insn("mode4"), 1
            t = ((raw[1] << 8) | raw[2]) & 0x3FFF
            return Insn("call" if raw[1] & 0x40 else "jump", (t,)), 3
        if n == 1:
            return Insn("nop.f4"), 1
        x = raw[1]
        if x < 0x40:
            return Insn("branch", (pc + 2 + x,)), 2     # [HW, probe26/27; 6-bit, probe70]
        if x == 0x40:
            return Insn("nop.f440"), 2
        if x & 0x1F:
            return Insn("int.f4", (x,)), 2              # f4 41-7f raise DMP_INT [HW, probe69]
        return Insn(".db", tuple(raw)), 2               # f4 60: unmeasured
    if icm and b == 0xFC:
        x, y = raw[1], raw[2]
        if x & 0xFC == 0xC0:
            return Insn(f"cmp.{COND[y & 3]}", (x & 3, y >> 2)), 3
        return Insn(".db", tuple(raw)), 3               # fc X Y, X below 0xc0: unmapped
    if icm and b == 0xFD:
        v = raw[1]
        if v & 0x80 or (not v & 0x20 and not v & 0x1F):
            return Insn(".db", tuple(raw)), 2
        adv = bool(v & 0x40)
        if v & 0x20:
            return Insn("sar", (32 - (v & 0x1F), adv)), 2
        return Insn("shl", (v & 0x1F, adv)), 2
    return Insn(".db", tuple(raw)), n


def encode(insn: Insn, chip, pc: int | None = None) -> bytes:
    """The bytes of `insn`.  A branch needs `pc` (its own address).  Mode consistency (an ALU
    mnemonic against the mode in force) is the assembler's check, not this function's."""
    chip = chip_name(chip)
    icm = chip == ICM
    op, a = insn.op, insn.args
    if op == ".db":
        for v in a:
            _rng(v, 0, 255, ".db byte")
        return bytes(a)
    if op in ALU_OPS:
        _chk(len(a) == 3, f"{op} takes A,B[,inc]")
        _rng(a[0], 0, 3, f"{op}: A")
        _rng(a[1], 0, 3, f"{op}: B")
        return bytes([(a[0] << 5) | (a[1] << 3) | ALU_OPS[op][1] | (4 if a[2] else 0)])
    if op in ("mov", "movx"):
        _chk(len(a) == 2, f"{op} takes k[,inc]")
        _rng(a[0], 0, 3, f"{op}: k")
        return bytes([0xC0 | (8 if op == "movx" else 0) | (a[0] << 1) | (1 if a[1] else 0)])
    if op in ("p0page", "p1page", "p2page"):
        _rng(a[0] if a else None, 0, 15, f"{op}: page")
        return bytes([(0x8 + int(op[1])) << 4 | a[0]])
    if op in ("p0bank", "p1bank", "p2bank"):
        _rng(a[0] if a else None, 0, 3, f"{op}: bank")
        return bytes([0xB0 | (int(op[1]) << 2) | a[0]])
    if op == "fmode":
        _chk(len(a) == 1 and a[0] in (0, 1, 2, 3, 5, 6, 7), "fmode takes 0-3 or 5-7 (4 is f4)")
        return bytes([0xF0 | a[0]])
    if op in FIXED_BYTE[chip]:
        _chk(not a, f"{op} takes no operands")
        return bytes([FIXED_BYTE[chip][op]])
    if icm:
        if op == "load":
            _rng(a[0] if a else None, 0, 3, "load: k")
            return bytes([0xD4 + a[0]])
        if op in ("jump", "call", "jump.m4", "call.m4"):
            t = a[0] if a else None
            _chk(isinstance(t, int), f"{op}: unresolved target {t!r}")
            _rng(t, 0, 0x3FFF, f"{op}: target")
            hi = (t >> 8) | (0x40 if op.startswith("call") else 0)
            return bytes(([0xF4] if "." not in op else []) + [hi, t & 0xFF])
        if op == "mode4":
            return b"\xf4"
        if op.startswith("cmp.") and op[4:] in COND:
            _rng(a[0], 0, 3, f"{op}: k")
            _rng(a[1], 0, 63, f"{op}: t")
            return bytes([0xFC, 0xC0 | a[0], (a[1] << 2) | COND.index(op[4:])])
        if op == "shl":
            _rng(a[0], 1, 31, "shl: s")
            return bytes([0xFD, a[0] | (0x40 if a[1] else 0)])
        if op == "sar":
            _rng(a[0], 1, 32, "sar: s")
            return bytes([0xFD, 0x20 | ((32 - a[0]) & 0x1F) | (0x40 if a[1] else 0)])
    else:
        if op == "branch":
            t = a[0] if a else None
            _chk(isinstance(t, int), f"branch: unresolved target {t!r}")
            _chk(pc is not None, "branch: encoding needs the instruction's own address")
            d = t - (pc + 2)
            _chk(0 <= d <= 63, f"branch to {t:#06x} from {pc:#06x}: the displacement {d} is "
                 "outside 0..63 (forward only, 6 bits; f4 40-7f is not a branch)")
            return bytes([0xF4, d])
        if op == "nop.f4":
            return b"\xf4"
        if op == "nop.f440":
            return b"\xf4\x40"
        if op == "int.f4":
            _chk(len(a) == 1 and 0x41 <= a[0] <= 0x7F and a[0] & 0x1F, "int.f4 takes 0x41-0x7f "
                 "with a non-zero low five bits")
            return bytes([0xF4, a[0]])
    raise EncodeError(f"{op!r} is not an instruction on the {chip}")


def effect(insn: Insn) -> str:
    """A short description of what the instruction does, for listings."""
    op, a = insn.op, insn.args
    if op in ALU_OPS:
        fam, i = ALU_OPS[op]
        A, B = f"P0.{a[0]}", f"P1.{a[1]}"
        body = {"add": f"{A} + {B}", "rsub": f"{B} - {A}", "mul": f"{A} * {B}",
                "nmul": f"-{A} * {B}", "min": f"min({A}, {B}); F := {A} > {B}",
                "cmp": f"F := {A} - {B} < 0 (no store)", "and": f"{A} & {B}", "xor": f"{A} ^ {B}",
                "mac": f"acc + {A} * {B}", "macn": f"acc - {A} * {B}",
                "mac.m": f"acc + {A} * {B}", "macn.m": f"acc - {A} * {B}"}[op]
        return ("P2 word += 1; " if a[2] else "") + body
    if op in ("mov", "movx"):
        pre = "P2 word += 1; " if a[1] else ""
        return pre + (f"P0.{a[0]} halves swapped" if op == "movx" else f"P0.{a[0]}")
    if op.endswith("page"):
        return f"{op[:2].upper()} page {a[0]}"
    if op.endswith("bank"):
        return f"{op[:2].upper()} bank {a[0]}"
    if op == "fmode":
        return {0: "multiply-accumulate", 1: "32-bit", 2: "high 16-bit lane", 3: "low 16-bit lane",
                5: "min/cmp/and/xor, 32-bit", 6: "min/cmp/and/xor, high lane",
                7: "min/cmp/and/xor, low lane"}[a[0]]
    if op in ("branch", "jump", "call", "jump.m4", "call.m4"):
        return f"{op.split('.')[0]} {a[0] if isinstance(a[0], str) else f'{a[0]:#06x}'}"
    if op.startswith("cmp."):
        return f"F := (P1.{a[0]} & 0x3f) {dict(eq='==', gt='>', lt='<', ne='!=')[op[4:]]} {a[1]}"
    if op == "load":
        return f"shifter := P0 page, select word {a[0]}"
    if op == "shl":
        return f"shift left {a[0]}" + ("; P2 word += 1" if a[1] else "")
    if op == "sar":
        return f"shift right {a[0]}" + ("; P2 word += 1" if a[1] else "")
    if op == "int.f4":
        return f"raise the interrupt (f4 {a[0]:02x})"
    if op.startswith("clr."):
        return f"M[P2] := 0 ({op[4:]})"
    if op == ".db":
        return "no measured meaning"
    return DOC.get(op, "")


_CHEAP = {"fmode", "always", "if f", "if nf", "if f.alt", "store", "fifo", "p2inc2", "test",
          "xchg01", "xchg02", "movz", "mode4", "nop.f4", "nop.f440", "p0bank", "p1bank", "p2bank",
          "p0blk", "p1blk", "p2blk", ".db"}
_KEYS = {"mov": "mov", "movx": "movx", "clr": "clr", "clr.next": "clr", "lo": "lo", "lo.next": "lo",
         "load": "sload", "shl": "shift", "sar": "shift", "inc": "inc", "dec": "inc",
         "inc.next": "inc", "dec.next": "inc", "branch": "branch", "jump": "branch",
         "jump.m4": "branch", "call": "call", "call.m4": "call", "ret": "ret", "int": "int",
         "int.fe": "int", "int.f4": "int2", "end": "halt", "p0page": "page", "p1page": "page",
         "p2page": "page", "min": "min", "cmp": "cmp", "and": "logic", "xor": "logic"}


def cost_key(insn: Insn, mode: int) -> str:
    """The cycle-table class (opendmp.emu.chips) of `insn` executed under `mode`: the same key
    the emulator charges, less its '.fifo' suffix.  A skipped instruction is 'skip' per byte."""
    op = insn.op
    if op in ("add", "rsub", "mul", "nmul"):
        return ("add" if op in ("add", "rsub") else "mul") + ("16" if mode in (2, 3) else "")
    if op in ("mac", "macn", "mac.m", "macn.m"):
        return "mac"
    if op.startswith("clr."):
        return "clr"
    if op.startswith("cmp."):
        return "fccmp"
    if op == ".db" and insn.args[:1] == (0xFD,):
        return "shift"                  # DMP3 fd n in an unmeasured form: the emulator shifts
    if op in _CHEAP:
        return "cheap"
    return _KEYS[op]
