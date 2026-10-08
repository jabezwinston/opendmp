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
from ..lang.errors import DmpError, library_file

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
    op: str                       # add rsub mul nmul (mode 1) | clamp and xor (mode 5)
    dst: Word
    a: Word                       # P0
    b: Word                       # P1


@dataclass(eq=False)
class IMul(Alu):
    """DMP3: dst := a * b mod 2^32, the integer product, as `mul` then `shl 30`.  The MUL loads the
    shifter with its whole 64-bit product (InvensenseDMP Finding 46), so the two are emitted
    together and nothing may come between them."""


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
    entry: bool = False           # a routine's entry (ICM): reached only by calls
    trips: int | None = None      # a loop head: the most times the body runs (None: a while)


@dataclass(eq=False)
class Call(Op):
    """Call a routine (ICM `f4 hi|0x40 lo`).  reads/writes: the words the routine (and what it
    calls) may touch, filled in once the routine is selected."""
    label: str
    reads: frozenset = frozenset()
    writes: frozenset = frozenset()


@dataclass(eq=False)
class Ret(Op):
    pass


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
def count_calls(stmts, acc: dict) -> dict:
    """How many call sites each sub has in `stmts` (not counting inside the subs themselves)."""
    for s in stmts:
        if isinstance(s, N.CallSub):
            acc[s.sub] = acc.get(s.sub, 0) + 1
        for attr in ("then", "els", "body"):
            inner = getattr(s, attr, None)
            if isinstance(inner, list):
                count_calls(inner, acc)
    return acc


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
        # ICM: a sub called more than once becomes a routine (call/ret), within the 4-deep stack
        self.routines: dict = {}                    # SubIR -> entry label
        self.pending: list = []
        self.calls = count_calls(ir.body, {})
        self.depth = {}
        self.notes: list = []                      # [(Loc, text)]: the method chosen, per construct

    def note(self, loc, text):
        if loc is not None and not library_file(loc.file):
            self.notes.append((loc, text))

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

    def sensor(self, f: str, loc=None) -> Word:
        """A word holding the sensor's Q30 value (raw << 16), extracted in the prologue; for
        fsync, the Int 0 or 1."""
        if f == "fsync":
            self.note(loc, "bit 0 of the temperature sample, masked out once per pass; the runtime "
                           "sets EXT_SYNC_SET so the FSYNC level lands there")
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
            if f == "fsync":
                # with EXT_SYNC_SET = 1 the FSYNC level is bit 0 of the temperature sample
                # [HW: ICM probe83; MPU probe105].  MPU: temp is word 1's low half, so the bit is
                # bit 0; ICM: temp is word 4's high half, so bit 16, swapped down by a MOVX
                if self.icm:
                    self.emit(Alu("and", w, self.sample_word(4), self.const(0x10000)))
                    self.emit(Mov(w, w, swap=True))
                else:
                    self.emit(Alu("and", w, self.sample_word(1), self.const(1)))
                continue
            k, half = self.sample_map[f]
            self.emit(Clr(w))
            self.emit(Mov(w, self.sample_word(k), swap=(half == "lo"), lane="hi"))
        if self.trace:                              # a trace build: the pass's raw sample first
            for k in TRACE_WORDS[self.chip]:
                self.emit(Mov(FIFO, self.sample_word(k)))
        self.ops += body
        self.emit(End())
        while self.pending:                         # the routines, after the pass's end
            sub = self.pending.pop(0)
            self.emit(Label(self.routines[sub], entry=True, loc=sub.loc))
            self.block(sub.body)
            self.emit(Ret(loc=sub.loc))
        return self.ops

    def sub_depth(self, sub) -> int:
        """How deep calls go below `sub`, counting only subs that would be routines."""
        if sub not in self.depth:
            self.depth[sub] = 0
            inner = count_calls(sub.body, {})
            self.depth[sub] = max((1 + self.sub_depth(s) for s in inner if not self.native(s)),
                                  default=0)
        return self.depth[sub]

    def native(self, sub) -> bool:
        """A library sub the target computes in hardware instead: DMP3's integer product."""
        return self.icm and sub.name == "imul" and sub.loc is not None and library_file(sub.loc.file)

    def as_routine(self, sub) -> bool:
        # the return stack is 4 deep and circular (Finding 33): main -> 4 nested routines at most
        return self.icm and self.calls.get(sub, 0) >= 2 and self.sub_depth(sub) <= 2

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
        if s.count is None:                         # while: a backward branch on both cores
            self.note(s.loc, "a real loop (a branch back); no trip count, so no pass-time bound")
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
        self.note(s.loc, f"a real loop: a counter from {s.count - 1}, DEC, a branch back while it is "
                         f"not negative; at most {s.count} trips")
        counter = self.temp("n")
        self.emit(Mov(counter, self.const(s.count - 1), loc=s.loc))
        top = self.new_label("top")
        self.emit(Label(top, loop_head=True, trips=s.count))
        self.labels.append({s.label: end})
        self.block(s.body)
        self.labels.pop()
        self.emit(IncDec(counter, True, loc=s.loc))
        self.emit(BranchIf(top, False, loc=s.loc))     # back while the count is not below 0
        self.emit(Label(end))

    def call_sub(self, s: N.CallSub):
        sub = s.sub
        vals = [self.operand(a) for a in s.args]
        if self.native(sub):
            self.note(s.loc, "n * m: MUL then SHL 30, 2 instructions (DMP3's shifter holds the whole "
                             "64-bit product)")
            self.emit(IMul("mul", self.word(s.outs[0]), vals[0], vals[1], loc=s.loc))
            return
        for p, v in zip(sub.params, vals):
            self.move(self.word(p), v)
        lib = sub.loc is not None and library_file(sub.loc.file)
        what = ("x / y (dmp.div): exact restoring division (the integer bit, then 30 fraction bits, "
                "saturating); " if lib and sub.name == "div" else
                "n // m (dmp.idiv): exact floor division of Int values (32 steps, as Python's //); "
                if lib and sub.name == "idiv" else
                "n * m (dmp.imul): the exact Int product modulo 2^32 by shift-and-add (32 steps; the "
                "MPU's multiply keeps only the Q30 top of a product); " if lib and sub.name == "imul" else
                f"{sub.name}(): ")
        if self.as_routine(sub):
            if sub not in self.routines:
                self.routines[sub] = self.new_label(f"sub.{sub.name}")
                self.pending.append(sub)
            self.note(s.loc, what + f"one routine, called here (call / ret); {self.calls[sub]} call sites share it")
            self.emit(Call(self.routines[sub], loc=s.loc))
        else:
            self.note(s.loc, what + ("inlined here (the MPU has no call)" if not self.icm else
                                     "inlined here (called once)" if self.calls.get(sub, 0) < 2 else
                                     "inlined here (the return stack is 4 deep)"))
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

    def lt(self, y_expr, x: Word, y: Word, when: bool, label: str, loc=None, quiet=False):
        """Branch to label when (x < y) == when.  rsub cannot subtract -2.0, so where y can be
        exactly -2.0 a guard decides that case first (nothing is less than -2.0)."""
        if not self.may_be_min(y_expr):
            if not quiet:
                self.note(loc, "RSUB into a scratch word, then a branch on its sign (the f5 compare "
                               "only tests signs)")
            self.emit(Cmp(x, y, self.get_scratch(), loc=loc))
            self.emit(BranchIf(label, when, loc=loc))
            return
        if not quiet:
            self.note(loc, "RSUB into a scratch word, then a branch on its sign, guarded: a subtrahend "
                           "of exactly -2.0 is possible here, and RSUB would add it (Finding 40)")
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
            self.note(loc, "RSUB, then TEST (F := the difference is 0)")
            self.emit(EqTest(a, b, self.get_scratch(), loc=loc))
        elif not self.may_be_min(b_expr):
            self.note(loc, "RSUB, then TEST (F := the difference is 0)")
            self.emit(EqTest(b, a, self.get_scratch(), loc=loc))
        else:
            self.note(loc, "XOR, then RSUB against 0 and TEST: both sides can be exactly -2.0")
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
            return self.sensor(e.field, e.loc)
        if isinstance(e, (N.Assume, N.Cast)):
            return self.operand(e.x)          # a Cast is the same word: no instruction
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
        if isinstance(e, N.Sensor) and e.field != "fsync":   # a raw sensor emits its int16
            self.note(e.loc or loc, "the sample's int16, 2 B straight from the sample page (a MOV under "
                                    "a 16-bit lane mode)")
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
        elif isinstance(e, (N.Assume, N.Cast)):
            self.assign(dst, e.x, loc)
        elif isinstance(e, N.Sum):
            self.sum_(dst, e, loc)
        elif isinstance(e, N.MinMax):
            if not self.minmax_clamp(dst, e, loc):
                a, b = self.operand(e.a), self.operand(e.b)
                self.select_cmp(dst, a, b, e, loc)
        elif isinstance(e, N.Abs):
            self.note(e.loc or loc, "compare with 0, then NMUL by 1.0 where negative (abs(-2.0) wraps)")
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
            self.note(e.loc or loc, "compare-and-skip: MOV the else value, test, skip the MOV of the other")
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
                self.note(e.loc or loc, "no OR instruction: (a ^ b) ^ (a & b), 3 ops")
                t1, t2 = self.temp(), self.temp()
                self.emit(Alu("and", t1, a, b, loc=loc))
                self.emit(Alu("xor", t2, a, b, loc=loc))
                self.emit(Alu("xor", dst, t2, t1, loc=loc))
            else:
                self.emit(Alu(e.op, dst, a, b, loc=loc))
        elif isinstance(e, N.Shift):
            x = self.operand(e.x)
            k = e.k
            if k == 0:
                self.move(dst, x, loc)
            elif e.op == "shr" and k <= 30:        # x >> k = floor(x * 2^-k): a MUL, exact
                self.note(e.loc or loc, f"a MUL by 2^-{k}")
                self.emit(Alu("mul", dst, x, self.const(1 << (30 - k)), loc=loc))
            elif e.op == "shr":                    # one MUL divides by at most 2^30
                self.note(e.loc or loc, "two MULs: by 2^-30, then by 2^-1 (floor(x / 2^31), 0 or -1)")
                self.emit(Alu("mul", dst, x, self.const(1), loc=loc))
                self.emit(Alu("mul", dst, dst, self.const(1 << 29), loc=loc))
            else:
                self.shl(dst, x, k, e.loc or loc)
        else:
            raise TypeError(type(e).__name__)

    def shl(self, dst: Word, x: Word, k: int, loc):
        """x << k (1 <= k <= 31), modulo 2^32."""
        if self.icm and k >= 2:
            self.note(loc, f"MUL by 2^{k}, then SHL 30: the integer product (2 instructions)")
            self.emit(IMul("mul", dst, x, self.const(1 << k), loc=loc))
        elif k >= 16:                              # MPU: the low half up a lane, then doublings
            self.note(loc, f"the low half moved up by a MOVX under the high lane (x << 16), then "
                           f"{k - 16} doublings")
            out = self.temp() if dst is x else dst
            self.emit(Clr(out, loc=loc))
            self.emit(Mov(out, x, swap=True, lane="hi", loc=loc))
            for _ in range(k - 16):
                self.emit(Alu("add", out, out, out, loc=loc))
            self.move(dst, out, loc)
        else:
            self.note(loc, f"{k} doubling{'s' if k > 1 else ''} (ADD), in place")
            self.emit(Alu("add", dst, x, x, loc=loc))
            for _ in range(k - 1):
                self.emit(Alu("add", dst, dst, dst, loc=loc))

    # ------------------------------------------------------------------ CLAMP (f5 op 0)
    # r = A >= 0 ? min(A, B) : max(A, -B), -B exact: A clamped to [-B, B] (ISA2 Finding 41)
    def interval(self, e):
        if isinstance(e, N.Const):
            return e.value, e.value
        iv = self.intervals.get(id(e))
        return iv if iv is not None else (MIN_WORD, -MIN_WORD - 1)

    @staticmethod
    def const_value(e):
        while isinstance(e, N.Assume):
            e = e.x
        return e.value if isinstance(e, N.Const) else None

    def minmax_clamp(self, dst, e: N.MinMax, loc) -> bool:
        """Compile e to CLAMP where that is exact for every value the operands can take."""
        top = -MIN_WORD - 1
        # dmp.clamp(x, lo, hi) = min(max(x, lo), hi) with constant bounds
        if e.op == "min" and isinstance(e.a, N.MinMax) and e.a.op == "max":
            lo, hi = self.const_value(e.a.b), self.const_value(e.b)
            if lo is not None and hi is not None and lo <= hi:
                x = e.a.a
                if lo == -hi:                      # symmetric: one CLAMP for every x
                    self.note(e.loc or loc, f"one CLAMP instruction (f5 op 0): x clamped to +-{hi / ONE:g}, "
                                            "exact for every value (Finding 41)")
                    self.emit(Alu("clamp", dst, self.operand(x), self.const(hi), loc=loc))
                    return True
                xl, xh = self.interval(x)
                if (lo + hi) % 2 == 0:             # shift to symmetric: clamp(x - m, +-h) + m
                    m, h = (lo + hi) // 2, (hi - lo) // 2
                    if MIN_WORD <= xl - m and xh - m <= top:
                        self.note(e.loc or loc, f"3 instructions: x - {m / ONE:g}, CLAMP to +-{h / ONE:g}, "
                                                f"+ {m / ONE:g} (the range shows x - {m / ONE:g} cannot wrap)")
                        t = self.temp("clamp")
                        self.emit(Alu("rsub", t, self.const(m), self.operand(x), loc=loc))   # x - m
                        self.emit(Alu("clamp", t, t, self.const(h), loc=loc))
                        self.emit(Alu("add", dst, t, self.const(m), loc=loc))
                        return True
        # min(x, c) / max(x, c) with a constant c, where x's range keeps CLAMP exact
        for x, c in ((e.a, e.b), (e.b, e.a)):
            cv = self.const_value(c)
            if cv is None:
                continue
            xl, xh = self.interval(x)
            if e.op == "min" and (xl >= 0 or (cv >= 0 and xl >= -cv)):
                self.note(e.loc or loc, f"one CLAMP instruction: its range keeps x >= {max(0, -cv) / ONE:g}, "
                                        "where the clamp is the minimum")
                self.emit(Alu("clamp", dst, self.operand(x), self.const(cv), loc=loc))
                return True
            if e.op == "max" and cv > MIN_WORD and (xh < 0 or (cv <= 0 and xh <= -cv)):
                self.note(e.loc or loc, "one CLAMP instruction: its range keeps x where the clamp is the maximum")
                self.emit(Alu("clamp", dst, self.operand(x), self.const(-cv), loc=loc))
                return True
        return False

    def select_cmp(self, dst, a, b, e: N.MinMax, loc):
        """dst := min(a, b) or max(a, b), by an exact compare (MIN's flag has no simple rule,
        probe99, and its b - a can overflow)."""
        guard = self.may_be_min(e.a if e.op == "min" else e.b)
        self.note(e.loc or loc, "compare-and-skip: MOV one side, compare by RSUB, skip the MOV of the "
                                "other" + (", with a -2.0 guard" if guard else "") +
                                " (no CLAMP form holds for these ranges)")
        out = self.temp() if dst in (a, b) else dst
        self.move(out, a, loc)
        skip = self.new_label(e.op)
        if e.op == "min":                          # take b when b < a
            self.lt(e.a, b, a, False, skip, loc, quiet=True)
        else:                                      # take b when a < b
            self.lt(e.b, a, b, False, skip, loc, quiet=True)
        self.move(out, b, loc)
        self.emit(Label(skip))
        self.move(dst, out, loc)

    # ------------------------------------------------------------------ sums
    def add_chain(self, dst: Word, V, loc) -> bool:
        """A sum of plain terms (no products) as a chain of ADD / RSUB: the same value as the
        exact sum modulo 2^32, which is what the language defines (the sum wraps once, at its
        end), in one op fewer than it has terms.  The destination may be read only by the first
        op, which reads before it stores."""
        terms = [(c > 0, x) for c, x in V for _ in range(abs(c))]
        if not 2 <= len(terms) <= 8:
            return False
        own = [t for t in terms if t[1] is dst]
        rest = [t for t in terms if t[1] is not dst]
        pos = [t for t in rest if t[0]]
        if len(own) == 2 and own[0][0] and own[1][0]:
            first = own                                   # dst + dst
        elif len(own) == 1 and own[0][0] and rest:
            first = [own[0], rest[0]]                     # dst +- x
        elif len(own) == 1 and pos:
            first = [pos[0], own[0]]                      # p - dst
        elif not own and pos:
            first = [pos[0], next(t for t in terms if t is not pos[0])]
        else:
            return False
        remaining = list(terms)
        for t in first:
            remaining.remove(t)
        (p1, x1), (p2, x2) = first
        if p1 and p2:
            self.emit(Alu("add", dst, x1, x2, loc=loc))
        elif p1:
            self.emit(Alu("rsub", dst, x2, x1, loc=loc))           # x1 - x2 = B - A
        else:
            self.emit(Alu("rsub", dst, x1, x2, loc=loc))           # x2 - x1
        for positive, x in remaining:
            self.emit(Alu("add", dst, dst, x, loc=loc) if positive else Alu("rsub", dst, x, dst, loc=loc))
        return True

    @staticmethod
    def digits(k: int, naf: bool) -> list[int]:
        """k's digits, least significant first: binary, or the non-adjacent form (digits in
        {-1, 0, 1}, no two adjacent nonzero: 7 = 8 - 1)."""
        out, s, k = [], (1 if k > 0 else -1), abs(k)
        while k:
            d = (2 - k % 4 if k % 2 else 0) if naf else k % 2
            out.append(s * d)
            k = (k - d) // 2
        return out

    def scale_plan(self, V, naf: bool):
        """sum(c * x) for integer c, by Horner's rule over the coefficients' digits: from the top
        digit position down, double the accumulator, then add or subtract each x whose digit
        there is +-1.  Every op wraps modulo 2^32, and so does the result, which is what the
        language defines for a sum.  ("acc" stands for the accumulator; "rsub" a, b is b - a.)"""
        ds = [(self.digits(c, naf), x) for c, x in V]
        zero = self.const(0)
        ops, lazy = [], None                      # lazy: (sign, x), the accumulator not yet written
        for pos in range(max(len(d) for d, _ in ds) - 1, -1, -1):
            if lazy is not None and lazy != "acc":
                s, x = lazy
                if s < 0:
                    ops.append(("rsub", x, zero))
                ops.append(("add", x, x) if s > 0 else ("add", "acc", "acc"))
                lazy = "acc"
            elif lazy == "acc":
                ops.append(("add", "acc", "acc"))
            here = [(d[pos], x) for d, x in ds if pos < len(d) and d[pos]]
            for s, y in sorted(here, key=lambda t: -t[0]):
                if lazy is None:
                    lazy = (s, y)
                elif lazy == "acc":
                    ops.append(("add", "acc", y) if s > 0 else ("rsub", y, "acc"))
                else:
                    s0, x = lazy
                    if s0 > 0:
                        ops.append(("add", x, y) if s > 0 else ("rsub", y, x))
                    elif s > 0:
                        ops.append(("rsub", x, y))
                    else:
                        ops += [("rsub", x, zero), ("rsub", y, "acc")]
                    lazy = "acc"
        if lazy not in (None, "acc"):
            s, x = lazy
            ops.append(("mov", x, None) if s > 0 else ("rsub", x, zero))
        elif lazy is None:
            ops.append(("mov", zero, None))
        return ops

    def best_scale(self, V):
        return min((self.scale_plan(V, naf) for naf in (False, True)), key=len)

    def emit_scale(self, dst: Word, plan, xs, loc):
        acc = self.temp("scale") if dst in xs else dst
        for op, a, b in plan:
            a, b = (acc if w == "acc" else w for w in (a, b))
            if op == "mov":
                self.move(acc, a, loc)
            else:
                self.emit(Alu(op, acc, a, b, loc=loc))
        if acc is not dst:
            self.move(dst, acc, loc)

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
                    self.note(e.loc or loc, "one DEC" if cb == -1 else "one INC")
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
                    self.note(e.loc or loc, "negation: one NMUL by 1.0")
                    return self.emit(Alu("nmul", dst, x, self.const(ONE), loc=loc))
                if c == 2:
                    self.note(e.loc or loc, "doubling: one ADD")
                    return self.emit(Alu("add", dst, x, x, loc=loc))
            if len(V) == 2:
                (c1, x1), (c2, x2) = V
                if (c1, c2) == (1, 1):
                    self.note(e.loc or loc, "one ADD")
                    return self.emit(Alu("add", dst, x1, x2, loc=loc))
                if (c1, c2) == (1, -1):
                    self.note(e.loc or loc, "one RSUB")
                    return self.emit(Alu("rsub", dst, x2, x1, loc=loc))    # x1 - x2 = B - A
                if (c1, c2) == (-1, 1):
                    self.note(e.loc or loc, "one RSUB")
                    return self.emit(Alu("rsub", dst, x1, x2, loc=loc))
            n = len(self.ops)
            expanded = sum(abs(c) for c, _ in V)
            plan = self.best_scale(V)
            chain = 2 <= expanded <= 8 and any(c > 0 for c, _ in V)    # add_chain's reach, roughly
            # (an ADD costs no more than a MAC term: 28 vs 60 cycles on the MPU, 26 vs 27 on the ICM)
            if len(plan) < expanded - 1 if chain else len(plan) <= expanded:
                self.emit_scale(dst, plan, [x for _, x in V], loc)
                self.note(e.loc or loc, f"{len(self.ops) - n} ADD/RSUB by doubling (Horner's rule on the "
                                        "integer coefficients): the exact sum, modulo 2^32")
                return
            if self.add_chain(dst, V, loc):
                self.note(e.loc or loc, f"{len(self.ops) - n} ADD/RSUB in a chain (no products): the same "
                                        "value as the exact sum, modulo 2^32")
                return
        elif len(P) == 1 and not V and abs(P[0][0]) == 1:
            c, a, b = P[0]
            self.note(e.loc or loc, "one MUL (floored)" if c == 1 else "one NMUL (floored)")
            return self.emit(Alu("mul" if c == 1 else "nmul", dst, a, b, loc=loc))
        one = self.const(ONE)
        expanded = sum(abs(c) for c, _ in V)
        if P and expanded > 2:                     # the plain part, scaled by doubling, enters once
            plan = self.best_scale(V)
            if len(plan) + 1 < expanded:
                t = self.temp("plain")
                self.emit_scale(t, plan, [x for _, x in V], loc)
                self.note(e.loc or loc, f"the plain terms by {len(plan)} ADD/RSUB (doubling), then "
                                        f"entering the multiply-accumulate run once")
                V = [(1, t)]
        terms = []
        for c, a, b in P:
            terms += [(c < 0, a, b)] * abs(c)
        for c, x in V:
            terms += [(c < 0, x, one)] * abs(c)
        self.note(e.loc or loc, f"one multiply-accumulate run of {len(terms)} terms (a plain value enters as "
                                f"x * 1.0), summed exactly (Q60) and floored once")
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
