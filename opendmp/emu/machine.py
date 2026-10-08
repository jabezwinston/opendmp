"""The DMP machine: state, one-instruction step, and one sample pass.

Semantics follow the InvensenseDMP programmer's manual (docs/manual/: chapter 5 defines every
instruction for both the MPU-6050 and the ICM-20948 DMP3 core).  The emulator is strict in two ways:

* `Unmodelled` is raised for any byte or state combination with no measured meaning (MPU `bf`,
  DMP3 `fc X Y` with X < 0xc0, both operand exchanges at once, ...).  It never guesses.
* Where the manual leaves a behaviour unmeasured but a program cannot avoid it (what survives
  from one pass to the next, how the multiply-accumulator rounds, ...), the choice is a named
  field of `Model`, so a result can always be traced to the assumption it rests on.
"""
from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass, field

from ..q30 import MASK, LO30, s32, sx16, sx30
from .chips import Chip, get_chip
from .disasm import disasm


UNKNOWN_F = object()      # F after an op whose flag rule is not measured


class Unmodelled(Exception):
    """The program did something the emulator has no measured model for."""


class DmpFault(Exception):
    """The program did something that stops the real DMP (a runaway pass, RET with no caller,
    a pointer or PC outside memory)."""


CARRY_ALL = ("pointers", "p2word", "route", "mode", "xchg", "predicate", "flags")
CARRY_MEASURED = ("pointers", "p2word", "route", "mode", "predicate", "flags", "acc")


@dataclass
class Model:
    """Behaviours the emulator must pick because the silicon has not settled them.

    mac_restart    what starts a fresh multiply-accumulate sum.  "chip" (default) = measured:
                   on the MPU any move of the destination (op&4, a P2 page load, d0, df, an
                   advancing MOV) [HW, probe86 B]; on DMP3 only an ALU op's own op&4
                   pre-advance [HW, probe88].  Or force "move", "op" or "none".
    mac_precision  "q60": products are summed exactly and the sum is floored once [HW both,
                   probe86 B / probe88].  "q30": each product is floored before it is added.
    icm_f0         DMP3 `f0`: "mac" (default: the multiply-accumulator, as on the MPU [HW,
                   probe88]) or "width" (a plain 32-bit store, the decoder's old reading).
    carry          machine state that survives from one pass to the next.  Measured on the
                   MPU (probe86 C): pointers, the P2 word index, route, mode and flags survive;
                   the fc/fd exchanges do not.  Always: the MAC accumulator resets [HW]; DMP3
                   blocks reset and its mode register (mode 4) survives [HW]; the MPU block
                   field survives [HW].
    max_steps      instructions after which a pass is declared runaway.
    """
    mac_restart: str = "chip"
    mac_precision: str = "q60"
    icm_f0: str = "mac"
    carry: tuple = CARRY_MEASURED
    max_steps: int = 200_000


@dataclass
class Ptr:
    page: int = 0
    bank: int = 0
    block: int = 0
    io: bool = False          # DMP3: loaded under mode 4, so it addresses the I/O space

    def addr(self) -> int:
        return self.block * 1024 + self.bank * 256 + self.page * 16


@dataclass
class Sample:
    """One sensor sample in raw int16 counts, as the hardware writes it into DMP RAM."""
    ax: int = 0
    ay: int = 0
    az: int = 0
    gx: int = 0
    gy: int = 0
    gz: int = 0
    temp: int = 0
    fsync: int = 0


@dataclass
class PassResult:
    fifo: bytes
    duration_ns: float        # tick -> halt, including tick stalls added by the scheduler
    steps: int
    ints: int
    costs: Counter            # instructions per cost class
    cycles: int = 0           # DMP clock cycles, tick -> halt (without tick stalls)
    index: int = 0            # pass number, set by the scheduler
    sample: int = 0           # sample row this pass saw
    start_ns: float = 0.0


def _i16(v: int) -> int:
    return max(-32768, min(32767, int(v)))


class Machine:
    def __init__(self, chip="mpu6050", model: Model | None = None, regs: dict | None = None,
                 clock_hz: float | None = None):
        self.chip: Chip = get_chip(chip, clock_hz)
        self.icm = self.chip.icm
        self.model = model or Model()
        self.mem = bytearray(self.chip.ram_size)
        self.entry = self.chip.entry
        # DMP3 I/O space = the host register file, one register per word (Finding 34).
        # Keyed (bank, register).  Only WHO_AM_I is known without a rig.
        self.regs = dict(regs) if regs is not None else ({(0, 0x00): 0xEA} if self.icm else {})
        self.io_log: list[tuple[int, int, int]] = []
        self.fsync_enabled = False
        self.tracer = None        # callable(Machine, pc, length, key, note) or None
        self.power_on()

    # ------------------------------------------------------------------ state
    def power_on(self):
        """The state after DMP_RST, as far as it is known (pointers and modes: INFER)."""
        self.p = [Ptr(), Ptr(), Ptr()]
        self.p2word = 0
        self.route_ram = True     # RAM after DMP_RST [HW, probe113]
        self.mode = 1
        self.xchg01 = self.xchg02 = self.bf = False
        self.armed = False        # a d9/da/db predicate is in force
        self.skip = False         # ... and it evaluated false
        self.f = False            # the one flag: result < 0, or == 0 after d1 [HW, probe86 G/H]
        self.acc = 0              # the result register R: the MAC accumulator and what d1 tests
                                  # (64-bit Q60, or Q30 under mac_precision="q30") [HW, probe106/107]
        self.sh = [0, 0, 0, 0]    # DMP3 shifter register
        self.sh_sel = 0
        self.sh_loaded = False
        self.stack = [None] * 4   # DMP3 return stack, circular [HW, probe80]
        self.sp = 0
        self.fifo = bytearray()
        self._fifo_wr = 0         # bytes written since the FIFO reset (where the next one lands)
        self.ints = 0
        self.pc = self.entry
        self.halted = False
        self._fresh = True        # the destination moved since the last MAC op (MPU rule)
        self._note = ""
        self._skip_len = 1

    # ------------------------------------------------------------------ host side
    def load(self, data: bytes, addr: int):
        if addr < 0 or addr + len(data) > len(self.mem):
            raise ValueError(f"{len(data)} B at {addr:#06x} does not fit the {self.chip.name}'s "
                             f"{len(self.mem):#x} B of DMP memory")
        self.mem[addr:addr + len(data)] = data

    def poke_words(self, addr: int, words):
        for i, w in enumerate(words):
            self.load((w & MASK).to_bytes(4, "big"), addr + 4 * i)

    def peek_word(self, addr: int) -> int:
        return s32(int.from_bytes(self.mem[addr:addr + 4], "big"))

    def write_sample(self, s: Sample):
        """What the hardware writes into RAM before every pass."""
        if self.icm:
            # 0x0000-0x0003 cleared, then accel, gyro, temperature as int16 BE, nothing else
            # in RAM but the FIFO's own bytes [HW, probe82b, probe91 B2];
            # with EXT_SYNC_SET = 1 the FSYNC level is bit 0 of the temperature [HW, probe83]
            temp = _i16(s.temp)
            if self.fsync_enabled:
                temp = (temp & ~1) | (s.fsync & 1)
            self.mem[0:4] = bytes(4)
            struct.pack_into(">7h", self.mem, 4, _i16(s.ax), _i16(s.ay), _i16(s.az),
                             _i16(s.gx), _i16(s.gy), _i16(s.gz), temp)
        else:
            # 0x000-0x00d in register-burst order, nothing else [HW, probe84, probe91 B2].
            # 0x00e is not written: probe84's "constant 12" was its own tag's low half.  With
            # EXT_SYNC_SET = 1 the temperature's bit 0 is the FSYNC level here too [HW, probe105]
            temp = _i16(s.temp)
            if self.fsync_enabled:
                temp = (temp & ~1) | (s.fsync & 1)
            struct.pack_into(">7h", self.mem, 0, _i16(s.ax), _i16(s.ay), _i16(s.az),
                             temp, _i16(s.gx), _i16(s.gy), _i16(s.gz))

    # ------------------------------------------------------------------ passes
    def begin_pass(self, sample: Sample | None = None):
        if sample is not None:
            self.write_sample(sample)
        carry = set(self.model.carry)
        if "acc" not in carry:                              # R (and a MAC sum) survives the pass
            self.acc = 0                                    # boundary on both parts [HW, probe113]
        if self.icm:
            for p in self.p:
                p.block = 0                                 # [HW, probe47]
            # the return stack and its pointer survive `end`: a pass that ends inside a call
            # leaves its return addresses for later `ret`s [HW, probe109 D]
        if "pointers" not in carry or self.icm:             # DMP3 starts every pass with all
            for p in self.p:                                # three pointers at 0x0000, word 0
                p.page = p.bank = 0                         # [HW, probe113]; the MPU keeps them
                p.io = False
        if "p2word" not in carry or self.icm:
            self.p2word = 0
        if "route" not in carry:
            self.route_ram = True
        if "mode" not in carry and not self.icm:            # DMP3 mode 4 survives [HW, probe54b]
            self.mode = 1
        if "xchg" not in carry:
            self.xchg01 = self.xchg02 = self.bf = False
        if "predicate" not in carry:
            self.armed = self.skip = False
        if "flags" not in carry:
            self.f = False
        self.pc = self.entry
        self.halted = False
        if "acc" not in carry:
            self._fresh = True

    def run_pass(self, sample: Sample | None = None) -> PassResult:
        self.begin_pass(sample)
        f0, i0 = len(self.fifo), self.ints
        costs: Counter = Counter()
        chip = self.chip
        cyc = chip.start_cycles
        steps = 0
        while not self.halted:
            key = self.step()
            costs[key] += 1
            cyc += chip.cyc(key) * (self._skip_len if key == "skip" else 1)
            steps += 1
            if steps > self.model.max_steps:
                raise DmpFault(f"pass did not halt within {self.model.max_steps} instructions "
                               f"(pc={self.pc:#06x}); the real DMP would stop serving samples")
        return PassResult(bytes(self.fifo[f0:]), cyc * chip.cycle_ns, steps, self.ints - i0, costs, cyc)

    def drain_fifo(self) -> bytes:
        out = bytes(self.fifo)
        self.fifo.clear()
        return out

    # ------------------------------------------------------------------ operands
    def _range(self, a: int, what: str):
        if not 0 <= a <= self.chip.data_limit - 4:
            raise DmpFault(f"{what} at {a:#06x} is outside the {self.chip.name}'s data space")

    def _read(self, pi: int, word: int) -> int:
        P = self.p[pi]
        a = P.addr() + 4 * word
        if P.io:
            return self._io_read(a)
        self._range(a, "read")
        return s32(int.from_bytes(self.mem[a:a + 4], "big"))

    def _io_read(self, a: int) -> int:
        if a >= 0x800:                                      # blocks 2-3 reach nothing [HW]
            return 0
        k = a >> 2
        return self.regs.get((k >> 7, k & 0x7F), 0) & 0xFF

    def _io_write(self, a: int, v: int):
        if a >= 0x800:
            return
        k = a >> 2
        self.regs[(k >> 7, k & 0x7F)] = v & 0xFF
        self.io_log.append((k >> 7, k & 0x7F, v & 0xFF))
        self._note = f"reg[{k >> 7}:{k & 0x7F:#04x}] := {v & 0xFF:#04x}"

    def _srcs(self):
        """(A pointer, B pointer, MOV source pointer) under the exchange toggles."""
        if self.xchg01 and self.xchg02:
            raise Unmodelled("fc and fd exchanges both in force: never measured")
        if self.xchg01:
            return 1, 0, 1                                  # [HW, probe35/68]
        if self.xchg02:
            return 2, 1, 2                                  # [HW, probe67/68]
        return 0, 1, 0

    def _dest(self):
        return (0 if self.xchg02 else 2), self.p2word       # fd: stores into P0's page

    def _width(self):
        m = self.mode
        return "hi" if m in (2, 6) else "lo" if m in (3, 7) else 32

    def _store(self, val: int) -> str:
        """Write a result through the route.  Returns '.fifo' or '' for the cost key."""
        w = self._width()
        if not self.route_ram:
            if w == 32:
                self._fifo_put((val & MASK).to_bytes(4, "big"))
            elif w == "hi":
                self._fifo_put(((val >> 16) & 0xFFFF).to_bytes(2, "big"))
            else:
                self._fifo_put((val & 0xFFFF).to_bytes(2, "big"))
            if self.tracer:
                self._note = f"fifo <- {val & MASK:08x}" + ("" if w == 32 else f" ({w})")
            return ".fifo"
        pi, word = self._dest()
        P = self.p[pi]
        a = P.addr() + 4 * word
        if P.io:
            self._io_write(a, val)
            return ""
        self._range(a, "store")
        if w == 32:
            self.mem[a:a + 4] = (val & MASK).to_bytes(4, "big")
        elif w == "hi":
            self.mem[a:a + 2] = ((val >> 16) & 0xFFFF).to_bytes(2, "big")
        else:
            self.mem[a + 2:a + 4] = (val & 0xFFFF).to_bytes(2, "big")
        if self.tracer:
            self._note = f"M[{a:#05x}] := {int.from_bytes(self.mem[a:a + 4], 'big'):08x}"
        return ""

    def _fifo_put(self, data: bytes, at: int | None = None):
        """Emit to the FIFO.  On DMP3 the FIFO is DMP RAM from 0x3000, filled in order from a
        FIFO reset [HW, probe91], so an emit also lands there (and would clobber code placed
        there); the wrap at 4 KB is INFER (stream mode overwrites the oldest bytes)."""
        if at is None:
            at = self._fifo_wr
            self.fifo += data
            self._fifo_wr += len(data)
        else:
            self.fifo[len(self.fifo) - len(data):] = data
        base = self.chip.fifo_base
        if base is not None:
            for i, v in enumerate(data):
                self.mem[base + (at + i) % self.chip.fifo_size] = v

    def fifo_reset(self):
        """The host's FIFO reset: empty it and restart the fill at the base."""
        self.fifo.clear()
        self._fifo_wr = 0

    def _adv(self, n: int):
        self.p2word = (self.p2word + n) & 3
        self._moved()

    def _mac_rule(self) -> str:
        rule = self.model.mac_restart
        if rule == "chip":
            rule = "op" if self.icm else "move"
        return rule

    def _moved(self):
        """The destination moved (a P2 page or bank load, d0, df, an op&4, an advancing MOV).  On
        the MPU that restarts a MAC sum by zeroing R, so d1 then reads zero; DMP3 keeps R (only an
        ALU op's own op&4 restarts its sum) [HW, probe86 B, probe88, probe107]."""
        self._fresh = True
        if self._mac_rule() == "move":
            self.acc = 0

    def _set_r(self, v: int):
        """A result replaces R (what a later MAC adds to and d1 tests)."""
        self.acc = (v << 30) if self.model.mac_precision == "q60" else v

    def _r_lo(self) -> int:
        """R's low 30 bits, sign-extended from bit 29 (DMP3 `lo`)."""
        return sx30(self.acc & LO30) if self.model.mac_precision == "q60" else 0

    def _load_shifter(self):
        """DMP3: every op that writes R loads the shifter from it: word 0 = R's word, word 1 = its
        low 30 bits [HW, probe64/110: CLAMP loads its compare D, INC its result, a MAC run its
        sum; AND, XOR and the sign test load nothing]."""
        if self.icm:
            self.sh, self.sh_sel, self.sh_loaded = [self._r_word(), self._r_lo() & MASK, 0, 0], 0, True

    def _r_word(self) -> int:
        return ((self.acc >> 30) if self.model.mac_precision == "q60" else self.acc) & MASK

    def _flags(self, f):
        """F := f (None leaves it); frozen under a predicate [HW].

        F is the sign of the EXACT result, not of the wrapped word: 1.0 + 1.5 stores -1.5 and
        leaves F clear, so a compare never overflows [HW, probe86 E].  There is no Z flag on
        either part: a zero result sets neither d9 nor da alone; d1 turns F into `R == 0`
        [HW, probe86 G/H; Finding 28's "d9 fires on 0" was the op's own zero emit]."""
        if self.armed:
            return
        if f is not None:
            self.f = f

    def _cond(self, b: int) -> bool:
        if self.f is UNKNOWN_F:
            raise Unmodelled("a predicate reads F where its rule is not modelled")
        return (not self.f) if b == 0xDA else self.f

    # ------------------------------------------------------------------ execution
    def length(self, pc: int) -> int:
        """Instruction length at pc in the current mode (how far a false predicate skips)."""
        m, b = self.mem, self.mem[pc]
        if self.icm:
            if self.mode == 4 and b < 0x80:
                return 2
            if b == 0xF4:
                return 3 if m[pc + 1] < 0x80 else 1
            return 3 if b == 0xFC else 2 if b == 0xFD else 1
        if b == 0xF4:
            return 2 if m[pc + 1] < 0x80 else 1
        return 1

    def step(self) -> str:
        """Execute one instruction; return its cost key."""
        pc = self.pc
        if not 0 <= pc < len(self.mem) - 2:
            raise DmpFault(f"PC ran off the end of memory ({pc:#06x})")
        if self.tracer:
            self._note = ""
            text = disasm(self, pc)
        key = self._step(pc)
        if self.tracer:
            self.tracer(self, pc, text, key, self._note)
        return key

    def _step(self, pc: int) -> str:
        m = self.mem
        b = m[pc]
        icm = self.icm

        # --- executes whatever the predicate says -------------------------------------
        if 0xD8 <= b <= 0xDB:                     # ALWAYS / IF: sticky until the next one [HW]
            self.armed = b != 0xD8
            self.skip = self.armed and not self._cond(b)
            self.pc = pc + 1
            return "cheap"
        if 0xF0 <= b <= 0xF7 and b != 0xF4:       # mode bytes are transparent to a predicate
            if icm and b == 0xF0 and self.model.icm_f0 == "width":
                self.mode = 1                     # DMP3 f0 read as a 32-bit width (B6)
            else:
                self.mode = b & 7
            self.pc = pc + 1
            return "cheap"
        if b == 0xF4:                             # every f4 enters mode 4, even when skipped:
            self.mode = 4                         # DMP3 [HW, probe64]; the MPU too [HW, probe95-97]
        if icm and b == 0xF4:
            n = self.length(pc)
            if self.skip or n == 1:
                self.pc = pc + n
                self._skip_len = n
                return "skip" if self.skip else "cheap"
            return self._jump(m[pc + 1], m[pc + 2], pc + 3)
        if self.skip:
            self._skip_len = self.length(pc)      # a skipped instruction costs per byte [HW]
            self.pc = pc + self._skip_len
            return "skip"

        # --- executed ---------------------------------------------------------------
        if icm and self.mode == 4 and b < 0x80:   # a branch target with no f4 of its own [HW]
            return self._jump(b, m[pc + 1], pc + 2) + "1"     # 9 cycles, bare [HW, probe111]
        if not icm and self.mode == 4 and b < 0x80:
            # the MPU's mode 4: a byte below 0x40 is a one-byte branch, the f4's signed 6-bit
            # displacement with the f4 implied: to pc + 1 + d, 10 cycles taken (ISA2 Finding 42,
            # probe104; the vendor's eMD 6.12 images chain their hops with it)
            if b < 0x40:
                self.pc = pc + 1 + (b if b < 0x20 else b - 64)
                return "branch1"
            raise Unmodelled(f"MPU byte {b:02x} in mode 4 at {pc:#06x}: no branch and no interrupt "
                             "(probe104), anything else unmeasured")
        self.pc = pc + 1
        if b < 0x80:
            return self._alu(b)
        hi = b >> 4
        if hi in (0x8, 0x9, 0xA):
            i = hi - 0x8
            P = self.p[i]
            P.page = b & 0xF
            P.io = icm and self.mode == 4          # I/O space [HW, probe82]
            if i == 2:
                self.p2word = 0                     # [HW, probe61/69]
                self._moved()
            return "page"
        if hi == 0xB:
            sel, v = (b >> 2) & 3, b & 3
            if sel < 3:
                self.p[sel].bank = v
                if sel == 2:
                    self._moved()                   # a P2 bank load too [HW, probe107]
            elif v < 3:
                P = self.p[v]
                P.block = (P.block + 1) % self.chip.block_mod   # [HW, probe47/84]
            elif icm:
                self.bf = not self.bf               # MOV stores zero while set [HW, probe83d/e]
            else:
                raise Unmodelled("bf on the MPU corrupts stores unpredictably (probe84b)")
            return "cheap"
        if hi == 0xC:
            return self._mov(b)
        if b == 0xD0:
            self._adv(2)
            return "cheap"
        if b == 0xD1:                               # TEST: F := (R's word == 0), at the lane
            w = self._r_word()                      # width in force when d1 runs [HW, probe107]
            if self.mode in (3, 7):
                w &= 0xFFFF
            elif self.mode in (2, 6):
                w &= 0xFFFF0000
            self.f = w == 0
            return "cheap"
        if 0xD2 <= b <= 0xD7:
            if not icm:
                return "clrx" + self._store(0)      # a clear on the MPU [HW, probe68]
            if b <= 0xD3:                           # R's low 30 bits: a product's, a MAC sum's
                if b == 0xD3:                       # [HW, probe51, probe110]
                    self._adv(1)
                return "lo" + self._store(self._r_lo())
            src = self._srcs()[2]
            self.sh = [self._read(src, k) & MASK for k in range(4)]
            self.sh_sel, self.sh_loaded = b - 0xD4, True
            self._set_r(s32(self.sh[self.sh_sel]))  # R := the selected word, F untouched [HW word 0,
            return "sload"                          # probe107]
        if b == 0xDC:
            self.route_ram = True
            return "cheap"
        if b == 0xDD:
            self.route_ram = False
            return "cheap"
        if b == 0xDE:
            return "clr" + self._store(0)
        if b == 0xDF:
            self._adv(1)
            return "clr" + self._store(0)
        if hi == 0xE:
            if not icm:
                return "clrx" + self._store(0)      # e0-ef clear on the MPU [HW, probe69]
            if b == 0xE0:
                return self._ret()
            if b == 0xE2:
                self.ints += 1                      # DMP_INT1 [HW, probe53]
                return "int"
            if b == 0xE7:
                return "clrx" + self._store(0)
            raise Unmodelled(f"{b:02x} on DMP3 has no measured meaning")
        if b == 0xF4:                               # the MPU branch family [HW, probe26/27/69/70]
            imm = m[pc + 1]
            if imm >= 0x80:
                return "cheap"                      # one byte; the next byte executes
            self.pc = pc + 2
            if imm < 0x40:                          # signed 6 bits: -32..+31 [HW, probe101]
                self.pc = pc + 2 + (imm if imm < 0x20 else imm - 64)
                return "branch"
            if imm & 0x1F:
                self.ints += 1
                return "int2"
            if imm == 0x40:
                return "cheap"
            raise Unmodelled(f"f4 {imm:02x}: only f4 40 is measured in this range")
        if 0xF8 <= b <= 0xFB:
            return self._incdec(b)
        if b == 0xFC:
            if not icm:
                self.xchg01 = not self.xchg01
                return "cheap"
            return self._fc(pc)
        if b == 0xFD:
            if not icm:
                self.xchg02 = not self.xchg02
                return "cheap"
            return self._shift(m[pc + 1])
        if b == 0xFE:
            if icm:
                raise Unmodelled("fe on DMP3 raises no interrupt (probe109); its effect is unmeasured")
            self.ints += 1                          # [HW, probe69/70]
            return "int"
        if b == 0xFF:
            self.halted = True
            return "halt"
        raise Unmodelled(f"byte {b:02x} at {pc:#06x}")

    # ------------------------------------------------------------------ instruction bodies
    def _lanes(self, a: int, b: int):
        """The operands as the width mode presents them to the ALU, and the lane width.

        f3/f7: the LOW half-words, sign-extended, as integers -- so a MUL there is
               (lo(a) * lo(b)) >> 30, tiny.  f2/f6: the HIGH half-words in place (low halves
               cleared).  Otherwise the whole words.  [HW MPU, probe86 A/E/F]"""
        m = self.mode
        if m in (3, 7):
            return sx16(a), sx16(b), 16
        if m in (2, 6):
            return s32(a & 0xFFFF0000), s32(b & 0xFFFF0000), 32
        return a, b, 32

    def _alu(self, v: int) -> str:
        A, B, op = (v >> 5) & 3, (v >> 3) & 3, v & 7
        if op & 4:
            self._adv(1)                            # PRE-advance [HW, probe32]
        pa, pb, _ = self._srcs()
        a, b = self._read(pa, A), self._read(pb, B)
        mode, k = self.mode, op & 3
        q60 = self.model.mac_precision == "q60"
        store = True
        racc = None
        if mode == 0:                               # multiply-accumulate [HW, probe74/86]
            prod = -(a * b) if op & 1 else a * b
            rule = self._mac_rule()
            if (rule == "move" and self._fresh) or (rule == "op" and op & 4):
                self.acc = 0
            if q60:                                 # a 64-bit register: Q60 in [-8, 8), wrapping;
                acc = (self.acc + prod) & ((1 << 64) - 1)   # F is its sign bit [HW, probe106]
                self.acc = acc - (1 << 64) if acc >> 63 else acc
                r = self.acc >> 30
            else:
                self.acc += prod >> 30
                r = self.acc
            self._flags(r < 0)
            key = "mac"
        elif mode in (5, 6, 7):                     # CLAMP / CMP / AND / XOR [HW, probe70/86]
            a, b, w = self._lanes(a, b)
            if k == 0:                              # CLAMP A to [-B, B] (the vendor's "MIN")
                x = min(a, b) if a >= 0 else max(a, -b)     # -B exact: B = -2.0 gives +2.0
                r = x - (1 << w) if x >= 1 << (w - 1) else x
                if r < -(1 << (w - 1)):
                    r += 1 << w
                # A compare inside: D = B - A when A >= 0, A + B when A < 0.  F := D < 0, i.e. A
                # was clamped [HW, probe103: 1920 cells per part, lanes included]; R := D, so d1
                # then tests whether A sat on its bound [HW, probe107]
                d = b - a if a >= 0 else a + b
                self._flags(d < 0)
                racc = d
                key = "logic"
            elif k == 1:                            # "CMP": a sign test, not a compare, both
                r, store = a - b, False             # cores: F := a < 0 and b >= 0 (probe98/99,
                self._flags(a < 0 <= b)             # 56 cells); no store; R untouched (probe107)
                key = "cmp"
            else:                                   # AND / XOR: F and R untouched [HW, probe107]
                r = (a & b) if k == 2 else (a ^ b)
                key = "logic"
            if k == 0:
                key = "clamp"
            if mode in (6, 7):
                key += "16"                         # half-word lanes run faster [HW, probe111]
        else:
            a, b, _ = self._lanes(a, b)
            if k == 0:
                r, key = a + b, "add"
            elif k == 1:                            # B - A as B + (-A), -A wrapped to the lane:
                w = 16 if mode in (3, 7) else 32    # so A = -2.0 is added, not subtracted, and F
                neg = -a                            # is the sign of that exact sum (probe100)
                if neg >= 1 << (w - 1):
                    neg -= 1 << w
                r, key = b + neg, "add"
            else:
                prod = a * b if k == 2 else -(a * b)
                r, key = prod >> 30, "mul"
            if mode in (2, 3):
                key += "16"                         # half-word lanes run faster [HW, probe87]
            self._flags(r < 0)
            racc = r
        if mode != 0 and racc is not None:
            if key.startswith("mul") and q60:
                self.acc = prod                     # R keeps the whole product [HW, probe110]
            else:
                self._set_r(racc)                   # the op's exact result replaces R
        self._fresh = False
        if mode == 0 or racc is not None:
            self._load_shifter()
        if store:
            key += self._store(r)
        elif self.tracer:
            self._note = f"flags from {r:#x}"
        return key

    def _mov(self, v: int) -> str:
        k = (v >> 1) & 3
        if v & 1:
            self._adv(1)
        val = self._read(self._srcs()[2], k)
        if v & 8:                                   # MOVX: half-words exchanged [HW]
            val = s32(((val & 0xFFFF) << 16) | ((val >> 16) & 0xFFFF))
        if self.icm and self.bf:
            val = 0
        return ("movx" if v & 8 else "mov") + self._store(val)

    def _incdec(self, b: int) -> str:
        """INC/DEC by one LSB of the lane: the low half-word under f3/f7, the high half-word
        under f2/f6 (+-0x10000), else the word.  F is the exact result's sign [HW, probe86 A/F]."""
        if b & 2:
            self._adv(1)
        d = -1 if b & 1 else 1
        m = self.mode
        if self.route_ram or not self.icm:
            # the operand is M[P2]; under FIFO routing the MPU emits the result as a new item of
            # the lane's width and leaves RAM alone [HW, probe112]
            pi, word = self._dest()
            P = self.p[pi]
            a = P.addr() + 4 * word
            old = self._io_read(a) if P.io else None
            if old is None:
                self._range(a, "inc/dec")
                old = s32(int.from_bytes(self.mem[a:a + 4], "big"))
        else:
            # DMP3 under FIFO routing changes the last emitted word [HW, probe52/112]; under a
            # lane mode the result is irregular (0000ffff + 1 under f3 gave ffff0001, probe112)
            if m in (2, 3, 6, 7):
                raise Unmodelled("DMP3 INC/DEC on the FIFO under a lane mode is irregular (probe112)")
            if len(self.fifo) < 4:
                raise Unmodelled("INC/DEC under FIFO routing with no word in the FIFO")
            old = s32(int.from_bytes(self.fifo[-4:], "big"))
        if m in (3, 7):
            r = sx16(old) + d
            new = (old & 0xFFFF0000) | (r & 0xFFFF)
        elif m in (2, 6):
            r = s32(old & 0xFFFF0000) + (d << 16)
            new = (r & 0xFFFF0000) | (old & 0xFFFF)
        else:
            r = old + d
            new = r
        new &= MASK
        if self.route_ram:
            if P.io:
                self._io_write(a, new)
            else:
                self.mem[a:a + 4] = new.to_bytes(4, "big")
                if self.tracer:
                    self._note = f"M[{a:#05x}] := {new:08x}"
        elif not self.icm:
            self._store(new)                        # a new FIFO item, the lane's width
        else:
            self._fifo_put(new.to_bytes(4, "big"), at=self._fifo_wr - 4)
        self._flags(r < 0)
        self._set_r(r)                              # R := the exact result [HW, probe107]
        self._load_shifter()                        # [HW, probe110]
        return "inc16" if m in (2, 3, 6, 7) else "inc"

    def _jump(self, hi: int, lo: int, ret: int) -> str:
        t = ((hi << 8) | lo) & 0x3FFF
        if hi & 0x40:                               # bit 14: call [HW, probe46]
            self.stack[self.sp] = ret
            self.sp = (self.sp + 1) & 3
        self.pc = t
        if self.tracer:
            self._note = ("call " if hi & 0x40 else "jump ") + f"{t:#06x}"
        return "call" if hi & 0x40 else "branch"

    def _ret(self) -> str:
        self.sp = (self.sp - 1) & 3
        r = self.stack[self.sp]
        if r is None:
            raise DmpFault("e0 with no return address: the DMP stops after this pass (manual: RET)")
        self.pc = r
        return "ret"

    def _fc(self, pc: int) -> str:
        x, y = self.mem[pc + 1], self.mem[pc + 2]
        self.pc = pc + 3
        if x < 0xC0:
            raise Unmodelled(f"fc {x:02x} {y:02x}: X below 0xc0 is only partly mapped (UNKNOWNS.md)")
        w = self._read(1, x & 3) & 0x3F
        t, c = y >> 2, y & 3
        self.f = (w == t, w > t, w < t, w != t)[c]  # [HW, probe83b]; updates F even when armed
        return "fccmp"

    def _shift(self, n: int) -> str:
        self.pc += 1
        s = n & 0x1F
        if n & 0x40:
            self._adv(1)
        if not self.sh_loaded:
            val = 0                                 # nothing loaded since DMP_RST [HW, probe51]
        else:
            w = self.sh
            if n & 0x20:                            # arithmetic right by 32 - s, per word
                new = [(s32(x) >> (32 - s)) & MASK for x in w]
            else:                                   # left by s, pulling the next word's fraction in
                new = [((w[i] << s) | (((w[(i + 1) & 3] << 2) & MASK) >> (32 - s))) & MASK
                       for i in range(4)]
            self.sh = new
            val = s32(new[self.sh_sel])
        self._set_r(val)                            # R := the shifted word, F untouched [HW, probe107]
        return "shift" + self._store(val)
