"""Run the IR, pass by pass, with the Q30 machine's arithmetic.

This is the second executable meaning of a program: `opendmp.lang.reference` evaluates the
source, this evaluates what the front end made of it, and they must agree bit for bit.  (The
emulator running the compiled bytes is the third.)
"""
from collections import Counter

from ..q30 import MASK, ONE, s32
from . import nodes as N


class _Break(Exception):
    def __init__(self, label):
        self.label = label


class Interp:
    MAX_WHILE = 100_000

    def __init__(self, ir: N.ProgramIR):
        self.ir = ir
        self.env: dict = {}
        self.assume_fails: Counter = Counter()
        self.reset()

    def reset(self):
        self.env.clear()
        for v in self.ir.vars:
            if v.kind in ("state", "param"):
                self.env[v] = v.init

    def set_param(self, name: str, raw: int):
        for v in self.ir.params:
            if v.name == name:
                self.env[v] = raw
                return
        raise KeyError(f"{self.ir.name} has no param {name!r}")

    def peek(self, name: str) -> int:
        for v in self.ir.vars:
            if v.name == name:
                return self.env[v]
        raise KeyError(name)

    # ------------------------------------------------------------------ a pass
    def run_pass(self, sample):
        """(packet bytes, interrupt raised?) for one sensor sample (opendmp.emu.Sample)."""
        self.sample = sample
        self.packet = bytearray()
        self.irq = False
        try:
            self.block(self.ir.body)
        except _Break as b:                         # only `pass` (a return) can escape
            if b.label != "pass":
                raise
        return bytes(self.packet), self.irq

    def block(self, stmts):
        for s in stmts:
            self.stmt(s)

    def stmt(self, s):
        if isinstance(s, N.Assign):
            self.env[s.var] = self.ev(s.value)
        elif isinstance(s, N.If):
            self.block(s.then if self.cond(s.cond) else s.els)
        elif isinstance(s, N.Block):
            try:
                self.block(s.body)
            except _Break as b:
                if b.label != s.label:
                    raise
        elif isinstance(s, N.Loop):
            n = 0
            try:
                while s.count is None or n < s.count:
                    n += 1
                    if s.count is None and n > self.MAX_WHILE:
                        raise RuntimeError(f"while loop at {s.loc} ran {self.MAX_WHILE} times")
                    self.block(s.body)
            except _Break as b:
                if b.label != s.label:
                    raise
        elif isinstance(s, N.Break):
            raise _Break(s.label)
        elif isinstance(s, N.Emit):
            for name, e in s.fields:
                v = self.ev(e)
                if e.type == N.S16:
                    self.packet += ((v >> 16) & 0xFFFF).to_bytes(2, "big")
                else:
                    self.packet += (v & MASK).to_bytes(4, "big")
        elif isinstance(s, N.Irq):
            self.irq = True
        elif isinstance(s, N.Check):
            self.check(self.ev(s.x), s.x.type, s.lo, s.hi, s.loc)
        elif isinstance(s, N.CallSub):
            args = [self.ev(a) for a in s.args]
            for p, v in zip(s.sub.params, args):
                self.env[p] = v
            try:
                self.block(s.sub.body)
            except _Break as b:
                if not b.label.startswith(s.sub.name + ".ret"):
                    raise
            for o, r in zip(s.outs, s.sub.results):
                self.env[o] = self.env[r]
        else:
            raise TypeError(f"unknown statement {type(s).__name__}")

    def check(self, v, typ, lo, hi, loc):
        x = v / ONE if typ in (N.Q30, N.S16) else v
        if not lo <= x <= hi:
            self.assume_fails[loc] += 1

    # ------------------------------------------------------------------ values
    def ev(self, e) -> int:
        if isinstance(e, N.Const):
            return e.value
        if isinstance(e, N.Load):
            return self.env[e.var]
        if isinstance(e, N.Sensor):
            s = self.sample
            if e.field == "fsync":
                return s.fsync & 1
            raw = {"ax": s.ax, "ay": s.ay, "az": s.az, "gx": s.gx, "gy": s.gy, "gz": s.gz,
                   "temp": s.temp}[e.field]
            return s32(raw << 16)
        if isinstance(e, N.Sum):
            total = 0
            for t in e.terms:
                if isinstance(t.x, N.Prod):
                    total += t.coef * self.ev(t.x.a) * self.ev(t.x.b)
                elif e.type == N.INT:
                    total += t.coef * self.ev(t.x)
                else:
                    total += (t.coef * self.ev(t.x)) << 30
            return s32(total if e.type == N.INT else total >> 30)
        if isinstance(e, N.MinMax):
            a, b = self.ev(e.a), self.ev(e.b)
            return min(a, b) if e.op == "min" else max(a, b)
        if isinstance(e, N.Abs):
            v = self.ev(e.x)
            return s32(-v) if v < 0 else v
        if isinstance(e, N.Select):
            c = self.cond(e.cond)
            a, b = self.ev(e.a), self.ev(e.b)
            return a if c else b
        if isinstance(e, N.BitOp):
            a, b = self.ev(e.a) & MASK, self.ev(e.b) & MASK
            return s32({"and": a & b, "xor": a ^ b, "or": a | b}[e.op])
        if isinstance(e, N.Shift):
            v = self.ev(e.x)
            return s32(v << e.k) if e.op == "shl" else v >> e.k
        if isinstance(e, N.Assume):
            v = self.ev(e.x)
            self.check(v, e.x.type, e.lo, e.hi, e.loc)
            return v
        raise TypeError(f"unknown expression {type(e).__name__}")

    def cond(self, c) -> bool:
        if isinstance(c, N.Cmp):
            a, b = self.ev(c.a), self.ev(c.b)
            return {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b, "eq": a == b,
                    "ne": a != b}[c.op]
        if isinstance(c, N.BoolOp):
            a, b = self.cond(c.a), self.cond(c.b)
            return (a and b) if c.op == "and" else (a or b)
        if isinstance(c, N.Not):
            return not self.cond(c.x)
        raise TypeError(f"unknown condition {type(c).__name__}")
