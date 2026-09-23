"""Instruction selection: the IR -> three-address code over abstract words.

A Word is one 32-bit RAM word: a state, a param, a temp, a pooled constant, or a word of the
hardware's sample page.  The TAC operations map one-to-one onto machine operations; which
pointers and modes they need is the scheduler's business (sched.py).

The one subtle selection is the Sum.  It is exact and floored once (PLAN.md section 4), which is
what one multiply-accumulate run computes when it starts fresh: every product is one MAC op, a
plain term t is t * 1.0 (exact), and the run's destination must not move until it ends.  Two-term
sums without products are one ADD or RSUB, a lone product one MUL or NMUL.
"""
from dataclasses import dataclass, field

from ..ir import nodes as N
from ..ir.ranges import analyse
from ..q30 import ONE, s32
from ..lang.errors import DmpError

MIN_WORD = -(1 << 31)             # -2.0

# where each sensor lives in the sample page: (word index, "hi" | "lo")
SAMPLE_MPU = {"ax": (0, "hi"), "ay": (0, "lo"), "az": (1, "hi"), "temp": (1, "lo"),
              "gx": (2, "hi"), "gy": (2, "lo"), "gz": (3, "hi")}
SAMPLE_ICM = {"ax": (1, "hi"), "ay": (1, "lo"), "az": (2, "hi"), "gx": (2, "lo"),
              "gy": (3, "hi"), "gz": (3, "lo"), "temp": (4, "hi")}


@dataclass(eq=False)
class Word:
    name: str
    kind: str                     # "state" "param" "temp" "const" "sample"
    init: int = 0                 # raw load-time value (state, param, const)
    addr: int | None = None       # set by the layout (fixed for sample words)
    mac_dst: bool = False         # the destination of a MAC run (the ICM wants it at an odd word)
    var: object = None

    def __repr__(self):
        return f"<{self.name}@{self.addr:#05x}>" if self.addr is not None else f"<{self.name}>"


FIFO = Word("fifo", "fifo")       # a destination: the FIFO route


# ------------------------------------------------------------------ operations
@dataclass(eq=False)
class Op:
    loc: object = field(default=None, kw_only=True)


@dataclass(eq=False)
class Alu(Op):
    op: str                       # add rsub mul nmul (mode 1) | min and xor (mode 5)
    dst: Word
    a: Word                       # P0
    b: Word                       # P1


@dataclass(eq=False)
class MacRun(Op):
    """dst := floor(sum of +-a*b), exactly: one MAC run from a fresh accumulator."""
    dst: Word
    terms: list                   # [(negative: bool, a: Word, b: Word)]


@dataclass(eq=False)
class Mov(Op):
    dst: Word                     # a word or FIFO
    src: Word
    swap: bool = False            # MOVX: the half-words exchanged
    lane: str | None = None       # None: 32 bits; "hi": f2 (upper half); "lo": f3 (lower half)


@dataclass(eq=False)
class Clr(Op):
    dst: Word


@dataclass(eq=False)
class IncDec(Op):
    dst: Word
    dec: bool


@dataclass(eq=False)
class Cmp(Op):
    """F := a - b < 0, as `rsub` (B - A with A = b) into a scratch word: the exact sign unless
    b is -2.0, which the hardware adds instead of subtracting (probe100).  The Selector guards
    that case.  (The `f5` op 1 "compare" only tests signs: F := A < 0 <= B, probe98/99.)"""
    a: Word
    b: Word
    scratch: Word


@dataclass(eq=False)
class EqTest(Op):
    """F := a == b: RSUB into a scratch word, then `test`."""
    a: Word
    b: Word
    scratch: Word


@dataclass(eq=False)
class BranchIf(Op):
    label: str
    when: bool                    # branch if F == when


@dataclass(eq=False)
class Branch(Op):
    label: str


@dataclass(eq=False)
class Label(Op):
    name: str
    loop_head: bool = False       # reached by a backward jump: nothing is known there


@dataclass(eq=False)
class Irq(Op):
    pass


@dataclass(eq=False)
class End(Op):
    pass


@dataclass(eq=False)
class Source(Op):
    """A listing marker: the statements below come from this source location."""
    pass


# ------------------------------------------------------------------ selection
TRACE_WORDS = {"mpu6050": (0, 1, 2, 3), "icm20948": (1, 2, 3, 4)}   # the sample page, as words


class Selector:
    def __init__(self, ir: N.ProgramIR, chip: str, trace: bool = False):
        self.ir, self.chip, self.trace = ir, chip, trace
        self.icm = chip == "icm20948"
        self.ops: list = []
        self.words: list = []
        self.var_words: dict = {}
        self.consts: dict = {}
        self.sensor_words: dict = {}
        self.sample_words: dict = {}
        self.n = 0
        self.labels: list = []            # label renaming scopes (unrolled copies, inlined subs)
        self.scratch = None
        self.sample_map = SAMPLE_ICM if self.icm else SAMPLE_MPU
        self.intervals = analyse(ir).intervals      # id(expr) -> (lo, hi): which compares need a guard

    def may_be_min(self, e) -> bool:
        """Can e be exactly -2.0 (the one value `rsub` cannot subtract)?"""
        if isinstance(e, N.Const):
            return e.value == MIN_WORD
        if isinstance(e, N.Assume):
            return e.lo <= -2.0 and self.may_be_min(e.x)
        iv = self.intervals.get(id(e))
        return iv is None or iv[0] <= MIN_WORD

    # ------------------------------------------------------------------ words
    def word(self, v: N.Var) -> Word:
        w = self.var_words.get(v)
        if w is None:
            w = Word(v.name, v.kind if v.kind in ("state", "param") else "temp", v.init, var=v)
            self.var_words[v] = w
            self.words.append(w)
        return w

    def const(self, raw: int) -> Word:
        raw = s32(raw)
        w = self.consts.get(raw)
        if w is None:
            w = Word(f"={raw / ONE:.6g}" if raw else "=0", "const", raw)
            self.consts[raw] = w
            self.words.append(w)
        return w

    def temp(self, hint="t") -> Word:
        self.n += 1
        w = Word(f"{hint}#{self.n}", "temp")
        self.words.append(w)
        return w

    def sample_word(self, k: int) -> Word:
        w = self.sample_words.get(k)
        if w is None:
            w = Word(f"sample.{k}", "sample", addr=4 * k)
            self.sample_words[k] = w
        return w

    def sensor(self, f: str) -> Word:
        """A word holding the sensor's Q30 value (raw << 16), extracted in the prologue."""
        if f == "fsync":
            raise DmpError("s.fsync is not supported by the backend yet")
        w = self.sensor_words.get(f)
        if w is None:
            w = Word(f"s.{f}", "temp")
            self.sensor_words[f] = w
            self.words.append(w)
        return w

    def get_scratch(self) -> Word:
        if self.scratch is None:
            self.scratch = Word("scratch", "temp")
            self.words.append(self.scratch)
        return self.scratch

    def emit(self, op):
        self.ops.append(op)
        return op

    def new_label(self, hint):
        self.n += 1
        return f"{hint}.{self.n}"

    # ------------------------------------------------------------------ program
    def run(self):
        body = []
        saved, self.ops = self.ops, body
        self.block(self.ir.body)
        self.ops = saved
        # the prologue: pull every sensor used as a value out of the sample page
        for f, w in self.sensor_words.items():
            k, half = self.sample_map[f]
            self.emit(Clr(w))
            self.emit(Mov(w, self.sample_word(k), swap=(half == "lo"), lane="hi"))
        if self.trace:                              # a trace build: the pass's raw sample first
            for k in TRACE_WORDS[self.chip]:
                self.emit(Mov(FIFO, self.sample_word(k)))
        self.ops += body
        self.emit(End())
        return self.ops

    def block(self, stmts):
        for s in stmts:
            self.stmt(s)

    def stmt(self, s):
        if getattr(s, "loc", None) is not None:
            self.emit(Source(loc=s.loc))
        if isinstance(s, N.Assign):
            self.assign(self.word(s.var), s.value, s.loc)
        elif isinstance(s, N.If):
            self.if_(s)
        elif isinstance(s, N.Block):
            end = self.scoped(s.label)
            self.block(s.body)
            self.labels.pop()
            self.emit(Label(end))
        elif isinstance(s, N.Loop):
            self.loop(s)
        elif isinstance(s, N.Break):
            self.emit(Branch(self.resolve(s.label), loc=s.loc))
        elif isinstance(s, N.Emit):
            for name, e in s.fields:
                self.emit_field(e, s.loc)
        elif isinstance(s, N.Irq):
            self.emit(Irq(loc=s.loc))
        elif isinstance(s, N.Check):
            pass
        elif isinstance(s, N.CallSub):
            self.call_sub(s)
        else:
            raise TypeError(type(s).__name__)

    # labels: unrolled loop bodies and inlined subs get fresh names for the labels they define
    def scoped(self, label):
        new = self.new_label(label)
        self.labels.append({label: new})
        return new

    def resolve(self, label):
        for scope in reversed(self.labels):
            if label in scope:
                return scope[label]
        return label

    # ------------------------------------------------------------------ control flow
    def if_(self, s: N.If):
        end = self.new_label("endif")
        if s.els:
            other = self.new_label("else")
            self.branch_if(s.cond, False, other)
            self.block(s.then)
            self.emit(Branch(end))
            self.emit(Label(other))
            self.block(s.els)
        else:
            self.branch_if(s.cond, False, end)
            self.block(s.then)
        self.emit(Label(end))

    def loop(self, s: N.Loop):
        if s.count is None:
            if not self.icm:
                raise DmpError("while needs a backward jump, and the MPU-6050 has none", s.loc,
                               "use `for _ in dmp.loop(N)`, which unrolls")
            end = self.scoped(s.label)
            top = self.new_label("top")
            self.emit(Label(top, loop_head=True))
            self.block(s.body)
            self.emit(Branch(top))
            self.labels.pop()
            self.emit(Label(end))
            return
        end = self.new_label(s.label)
        if s.count == 1:
            self.labels.append({s.label: end})
            self.block(s.body)
            self.labels.pop()
            self.emit(Label(end))
            return
        # a real loop, on both cores: the MPU branches back up to 32 bytes (probe101), and
        # relaxation chains hops for longer bodies.  The counter runs N-1 .. -1; DEC's F (the
        # exact sign of its result) ends the loop.
        counter = self.temp("n")
        self.emit(Mov(counter, self.const(s.count - 1), loc=s.loc))
        top = self.new_label("top")
        self.emit(Label(top, loop_head=True))
        self.labels.append({s.label: end})
        self.block(s.body)
        self.labels.pop()
        self.emit(IncDec(counter, True, loc=s.loc))
        self.emit(BranchIf(top, False, loc=s.loc))     # back while the count is not below 0
        self.emit(Label(end))

    def call_sub(self, s: N.CallSub):
        sub = s.sub
        vals = [self.operand(a) for a in s.args]
        for p, v in zip(sub.params, vals):
            self.move(self.word(p), v)
        self.block(sub.body)
        for o, r in zip(s.outs, sub.results):
            self.move(self.word(o), self.word(r))

    def branch_if(self, c, when: bool, label: str):
        """Branch to label when the condition c evaluates to `when`."""
        if isinstance(c, N.Not):
            return self.branch_if(c.x, not when, label)
        if isinstance(c, N.BoolOp):
            is_and = c.op == "and"
            if is_and != when:                     # and/False or or/True: either part decides
                self.branch_if(c.a, when, label)
                self.branch_if(c.b, when, label)
            else:
                skip = self.new_label("cond")
                self.branch_if(c.a, not when, skip)
                self.branch_if(c.b, when, label)
                self.emit(Label(skip))
            return
        a, b = self.operand(c.a), self.operand(c.b)
        op = c.op
        if op in ("eq", "ne"):
            self.eq(c.a, c.b, a, b, when if op == "eq" else not when, label, c.loc)
        elif op in ("lt", "ge"):
            self.lt(c.b, a, b, when if op == "lt" else not when, label, c.loc)
        else:                                      # gt / le: b < a
            self.lt(c.a, b, a, when if op == "gt" else not when, label, c.loc)

    def lt(self, y_expr, x: Word, y: Word, when: bool, label: str, loc=None):
        """Branch to label when (x < y) == when.  rsub cannot subtract -2.0, so where y can be
        exactly -2.0 a guard decides that case first (nothing is less than -2.0)."""
        if not self.may_be_min(y_expr):
            self.emit(Cmp(x, y, self.get_scratch(), loc=loc))
            self.emit(BranchIf(label, when, loc=loc))
            return
        # F := y - (-2.0 + 1 lsb) < 0, i.e. y == -2.0 (the subtrahend here is safe)
        self.emit(Cmp(y, self.const(MIN_WORD + 1), self.get_scratch(), loc=loc))
        if when:
            skip = self.new_label("lt")
            self.emit(BranchIf(skip, True, loc=loc))
            self.emit(Cmp(x, y, self.get_scratch(), loc=loc))
            self.emit(BranchIf(label, True, loc=loc))
            self.emit(Label(skip))
        else:
            self.emit(BranchIf(label, True, loc=loc))
            self.emit(Cmp(x, y, self.get_scratch(), loc=loc))
            self.emit(BranchIf(label, False, loc=loc))

    def eq(self, a_expr, b_expr, a: Word, b: Word, when: bool, label: str, loc=None):
        """Branch to label when (a == b) == when: rsub with a subtrahend that cannot be -2.0,
        or, when both can, the XOR of the two against zero."""
        if not self.may_be_min(a_expr):
            self.emit(EqTest(a, b, self.get_scratch(), loc=loc))
        elif not self.may_be_min(b_expr):
            self.emit(EqTest(b, a, self.get_scratch(), loc=loc))
        else:
            d = self.temp("xor")
            self.emit(Alu("xor", d, a, b, loc=loc))          # 0 exactly when equal
            self.emit(EqTest(d, self.const(0), self.get_scratch(), loc=loc))
        self.emit(BranchIf(label, when, loc=loc))

    # ------------------------------------------------------------------ values
    def operand(self, e) -> Word:
        """A word that holds e's value (computed into a temp when it is not one already)."""
        if isinstance(e, N.Load):
            return self.word(e.var)
        if isinstance(e, N.Const):
            return self.const(e.value)
        if isinstance(e, N.Sensor):
            return self.sensor(e.field)
        if isinstance(e, N.Assume):
            return self.operand(e.x)
        t = self.temp()
        self.assign(t, e, e.loc)
        return t

    def move(self, dst: Word, src: Word, loc=None):
        if dst is not src:
            if src.kind == "const" and src.init == 0:
                self.emit(Clr(dst, loc=loc))
            else:
                self.emit(Mov(dst, src, loc=loc))

    def emit_field(self, e, loc):
        if isinstance(e, N.Sensor) and e.field != "fsync":
            k, half = self.sample_map[e.field]
            self.emit(Mov(FIFO, self.sample_word(k), lane=half, loc=loc))
            return
        w = self.operand(e)
        if e.type == N.S16:
            self.emit(Mov(FIFO, w, lane="hi", loc=loc))
        else:
            self.emit(Mov(FIFO, w, loc=loc))

    def assign(self, dst: Word, e, loc):
        if isinstance(e, (N.Load, N.Const, N.Sensor)):
            self.move(dst, self.operand(e), loc)
        elif isinstance(e, N.Assume):
            self.assign(dst, e.x, loc)
        elif isinstance(e, N.Sum):
            self.sum_(dst, e, loc)
        elif isinstance(e, N.MinMax):
            a, b = self.operand(e.a), self.operand(e.b)
            self.select_cmp(dst, a, b, e, loc)
        elif isinstance(e, N.Abs):
            a = self.operand(e.x)
            out = self.temp() if dst is a else dst
            self.move(out, a, loc)
            skip = self.new_label("abs")
            self.emit(Cmp(a, self.const(0), self.get_scratch(), loc=loc))     # F := a < 0
            self.emit(BranchIf(skip, False, loc=loc))
            self.emit(Alu("nmul", out, a, self.const(ONE), loc=loc))   # -a, exactly (wraps at -2)
            self.emit(Label(skip))
            self.move(dst, out, loc)
        elif isinstance(e, N.Select):
            a, b = self.operand(e.a), self.operand(e.b)
            out = self.temp() if dst in (a, b) else dst
            self.move(out, b, loc)
            skip = self.new_label("sel")
            self.branch_if(e.cond, False, skip)
            self.move(out, a, loc)
            self.emit(Label(skip))
            self.move(dst, out, loc)
        elif isinstance(e, N.BitOp):
            a, b = self.operand(e.a), self.operand(e.b)
            if e.op == "or":                       # a | b = (a ^ b) ^ (a & b)
                t1, t2 = self.temp(), self.temp()
                self.emit(Alu("and", t1, a, b, loc=loc))
                self.emit(Alu("xor", t2, a, b, loc=loc))
                self.emit(Alu("xor", dst, t2, t1, loc=loc))
            else:
                self.emit(Alu(e.op, dst, a, b, loc=loc))
        elif isinstance(e, N.Shift):
            x = self.operand(e.x)
            if e.op == "shr":                      # x >> k = floor(x * 2^-k): a MUL
                self.emit(Alu("mul", dst, x, self.const(1 << (30 - e.k)), loc=loc))
            elif e.k == 0:
                self.move(dst, x, loc)
            else:                                  # x << k: k doublings
                cur = x
                for i in range(e.k):
                    nxt = dst if i == e.k - 1 else self.temp()
                    self.emit(Alu("add", nxt, cur, cur, loc=loc))
                    cur = nxt
        else:
            raise TypeError(type(e).__name__)

    def select_cmp(self, dst, a, b, e: N.MinMax, loc):
        """dst := min(a, b) or max(a, b), by an exact compare (MIN's flag has no simple rule,
        probe99, and its b - a can overflow)."""
        out = self.temp() if dst in (a, b) else dst
        self.move(out, a, loc)
        skip = self.new_label(e.op)
        if e.op == "min":                          # take b when b < a
            self.lt(e.a, b, a, False, skip, loc)
        else:                                      # take b when a < b
            self.lt(e.b, a, b, False, skip, loc)
        self.move(out, b, loc)
        self.emit(Label(skip))
        self.move(dst, out, loc)

    # ------------------------------------------------------------------ sums
    def sum_(self, dst: Word, e: N.Sum, loc):
        prods, plain = [], []
        for t in e.terms:
            if isinstance(t.x, N.Prod):
                prods.append((t.coef, t.x))
            else:
                plain.append((t.coef, t.x))
        # Int n +- 1 into n itself: INC / DEC
        if not prods and len(plain) == 2 and dst.var is not None:
            (c1, x1), (c2, x2) = plain
            for (ca, xa), (cb, xb) in (((c1, x1), (c2, x2)), ((c2, x2), (c1, x1))):
                if ca == 1 and isinstance(xa, N.Load) and self.word(xa.var) is dst and \
                        isinstance(xb, N.Const) and xb.value == 1 and cb in (1, -1):
                    self.emit(IncDec(dst, cb == -1, loc=loc))
                    return
        # the operands first: nothing may run between the ops of a MAC run
        P = [(c, self.operand(p.a), self.operand(p.b)) for c, p in prods]
        V = [(c, self.operand(x)) for c, x in plain]
        if not P:
            if len(V) == 1:
                (c, x), = V
                if c == 1:
                    return self.move(dst, x, loc)
                if c == -1:
                    return self.emit(Alu("nmul", dst, x, self.const(ONE), loc=loc))
                if c == 2:
                    return self.emit(Alu("add", dst, x, x, loc=loc))
            if len(V) == 2:
                (c1, x1), (c2, x2) = V
                if (c1, c2) == (1, 1):
                    return self.emit(Alu("add", dst, x1, x2, loc=loc))
                if (c1, c2) == (1, -1):
                    return self.emit(Alu("rsub", dst, x2, x1, loc=loc))    # x1 - x2 = B - A
                if (c1, c2) == (-1, 1):
                    return self.emit(Alu("rsub", dst, x1, x2, loc=loc))
        elif len(P) == 1 and not V and abs(P[0][0]) == 1:
            c, a, b = P[0]
            return self.emit(Alu("mul" if c == 1 else "nmul", dst, a, b, loc=loc))
        one = self.const(ONE)
        terms = []
        for c, a, b in P:
            terms += [(c < 0, a, b)] * abs(c)
        for c, x in V:
            terms += [(c < 0, x, one)] * abs(c)
        reads = [t for t in terms if dst in (t[1], t[2])]
        if reads:
            if len(reads) == 1:                    # dst read once: that term goes first
                terms.remove(reads[0])
                terms.insert(0, reads[0])
            else:
                t = self.temp("sum")
                t.mac_dst = True
                self.emit(MacRun(t, terms, loc=loc))
                return self.move(dst, t, loc)
        dst.mac_dst = True
        self.emit(MacRun(dst, terms, loc=loc))
