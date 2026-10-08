"""The scheduler: three-address code over placed words -> machine instructions.

It tracks the machine state an instruction depends on and emits only what changes:
  the three pointers (bank, page), P2's walked word index, the mode register, the route,
  and on the MPU whether R is 0: a P2 move zeroes it, an ALU op or INC/DEC writes it, and a MAC
  run starts afresh only from a zero R (InvensenseDMP Findings 45 and 54).
At a label the state is what every way in agrees on (recorded at each branch); a label reached
by a conditional branch starts with `always` (the predicate is still armed there), and on the
ICM a label reached by a jump starts by leaving mode 4.

The measured rules it relies on (opendmp.isa.facts): ptr.p2_reset, alu.preadvance,
mac.restart.mpu, mac.restart.icm, pred.sticky, jump.icm, branch.mpu, io.icm.
"""
from dataclasses import dataclass, field

from ..isa.core import Insn
from ..lang.errors import DmpError
from . import tac as T

MODE = {"add": 1, "rsub": 1, "mul": 1, "nmul": 1, "clamp": 5, "and": 5, "xor": 5}
COMMUTATIVE = {"add", "mul", "nmul", "and", "xor"}
WIDE = (0, 1, 5)                  # modes that move and clear whole words
WIDE_MPU = (0, 1, 4, 5)           # the MPU's mode 4 (after any f4) moves and clears too (probe96)


@dataclass
class State:
    p: list = field(default_factory=lambda: [None, None, None])     # (bank, page) per pointer
    p2i: int | None = None         # P2's word index
    mode: int | None = None
    route: str | None = None       # "ram" | "fifo"
    fresh: bool | None = None      # MPU: R is 0 (a P2 move since the last write of R)

    def copy(self):
        return State(list(self.p), self.p2i, self.mode, self.route, self.fresh)


def known_in(a: State, b: State) -> bool:
    """Everything a claims is also true in b (a is safe to assume where b holds)."""
    return all(x is None or x == y for x, y in zip(a.p, b.p)) and \
        (a.p2i is None or (a.p2i == b.p2i and a.p[2] == b.p[2])) and \
        (a.mode is None or a.mode == b.mode) and (a.route is None or a.route == b.route) and \
        (a.fresh is None or a.fresh == b.fresh)


def join(a: State, b: State) -> State:
    return State([x if x == y else None for x, y in zip(a.p, b.p)],
                 a.p2i if a.p2i == b.p2i and a.p[2] == b.p[2] else None,
                 a.mode if a.mode == b.mode else None,
                 a.route if a.route == b.route else None,
                 a.fresh if a.fresh == b.fresh else None)


@dataclass(eq=False)
class Emitted:
    insn: Insn | None              # None: a label
    label: str | None = None
    target: str | None = None      # a branch or jump's label
    safe: bool = False             # a hop may go before this entry (no predicate armed)
    loc: object = None
    note: str = ""
    mode: int | None = None        # the mode register in force before it (for its cost)
    loop_head: bool = False        # a label: a loop's head ...
    trips: int | None = None       # ... whose body runs at most this often (None: unbounded)
    entry: bool = False            # a label: a routine's entry
    addr: int = 0
    size: int = 0
    bytes: bytes = b""


def page_of(addr):
    return (addr >> 8, (addr >> 4) & 15)


class Scheduler:
    def __init__(self, chip: str):
        self.icm = chip == "icm20948"
        self.wide = WIDE if self.icm else WIDE_MPU
        self.out: list = []
        self.st = State()
        self.reachable = True
        self.arrivals: dict = {}   # label -> [(State, conditional)]
        self.loc = None
        self.group = False
        self.note = ""
        self.heads: dict = {}      # loop head label -> the state assumed there
        self.fall: dict = {}       # loop head label -> the state the code falling in brings
        self.back: dict = {}       # loop head label -> [states the backward branches bring]
        self.placed: set = set()   # labels already emitted

    # ------------------------------------------------------------------ output
    def put(self, op, *args, target=None):
        e = Emitted(Insn(op, tuple(args)), target=target, safe=self.group, loc=self.loc,
                    note=self.note, mode=self.st.mode)
        self.group = False
        self.note = ""
        self.out.append(e)
        return e

    def set_mode(self, m, force=False):
        if force or self.st.mode != m:
            self.put("fmode", m)
            self.st.mode = m

    def set_route(self, r):
        if self.st.route != r:
            self.put("store" if r == "ram" else "fifo")
            self.st.route = r

    def set_ptr(self, i, addr, force=False):
        if self.icm and self.st.mode in (None, 4):
            self.set_mode(1)                   # a page load in mode 4 would reach the I/O space
        bank, page = page_of(addr)
        cur = self.st.p[i]
        if cur is None or cur[0] != bank:
            self.put(f"p{i}bank", bank)
        if force or cur is None or cur[1] != page:
            self.put(f"p{i}page", page)
            if i == 2:
                self.st.p2i, self.st.fresh = 0, True
        self.st.p[i] = (bank, page)

    def ptr_cost(self, i, addr):
        cur = self.st.p[i]
        bank, page = page_of(addr)
        return (cur is None or cur[0] != bank) + (cur is None or cur[1] != page)

    def seek(self, w: T.Word) -> bool:
        """Point P2 so that the next store lands on w; True if that op must pre-advance."""
        widx = (w.addr >> 2) & 3
        if self.st.p[2] != page_of(w.addr) or self.st.p2i is None:
            self.set_ptr(2, w.addr, force=True)
        s = (widx - self.st.p2i) % 4
        if s >= 2:
            self.put("p2inc2")
            self.st.p2i = (self.st.p2i + 2) % 4
            self.st.fresh = True
            s -= 2
        return s == 1

    def stored(self, w, inc, alu):
        self.st.p2i = (w.addr >> 2) & 3
        if alu:
            self.st.fresh = False
        elif inc:
            self.st.fresh = True

    @staticmethod
    def idx(w):
        return (w.addr >> 2) & 3

    # ------------------------------------------------------------------ the pass
    def run(self, ops, heads=None):
        self.heads = heads or {}
        if self.icm:
            self.set_mode(1, force=True)        # mode 4 may survive from the last pass
        self.put("always")
        for op in ops:
            if isinstance(op, T.Source):
                self.loc = op.loc
                continue
            if isinstance(op, T.Label):
                self.label(op)
                continue
            if not self.reachable:
                continue                        # after an unconditional branch
            self.group = True
            getattr(self, "op_" + type(op).__name__)(op)
        return self.out

    def op_Alu(self, op):
        self.alu(op.op, op.dst, op.a, op.b)

    def op_IMul(self, op):
        # DMP3: the MUL loads the shifter with the whole product; SHL 30 stores a*b mod 2^32 over
        # the MUL's own word (no advance), and makes it R (InvensenseDMP Finding 46)
        self.alu("mul", op.dst, op.a, op.b)
        self.put("shl", 30, False)

    def alu(self, name, dst, a, b):
        self.set_mode(MODE[name])
        self.set_route("fifo" if dst is T.FIFO else "ram")
        if name in COMMUTATIVE and a is not b and \
                self.ptr_cost(0, b.addr) + self.ptr_cost(1, a.addr) < \
                self.ptr_cost(0, a.addr) + self.ptr_cost(1, b.addr):
            a, b = b, a
        self.set_ptr(0, a.addr)
        self.set_ptr(1, b.addr)
        inc = False if dst is T.FIFO else self.seek(dst)
        self.put(name, self.idx(a), self.idx(b), inc)
        if dst is not T.FIFO:
            self.stored(dst, inc, True)
        else:
            self.st.fresh = False

    def op_Cmp(self, op):
        # F := a - b < 0 as rsub (B - A) with A = b, B = a, into the scratch word (probe99/100)
        self.alu("rsub", op.scratch, op.b, op.a)

    def op_EqTest(self, op):
        self.alu("rsub", op.scratch, op.a, op.b)
        self.put("test")

    def op_MacRun(self, op):
        dst = op.dst
        self.set_route("ram")
        self.set_mode(0)
        widx = self.idx(dst)
        same = self.st.p[2] == page_of(dst.addr) and self.st.p2i is not None
        if self.icm:
            # the first op pre-advances (that is what restarts the sum): P2 at widx - 1
            need = (widx - 1) % 4
            if not (same and (need - self.st.p2i) % 4 in (0, 2)):
                if need not in (0, 2):
                    raise DmpError(f"internal: MAC destination {dst.name} at an even word")
                self.set_ptr(2, dst.addr, force=True)
            if (need - self.st.p2i) % 4 == 2:
                self.put("p2inc2")
                self.st.p2i = need
            first_inc = True
        else:
            # R must be 0: a P2 move zeroes it, and nothing may write it after the move
            if not same:
                self.set_ptr(2, dst.addr, force=True)
            s = (widx - self.st.p2i) % 4
            if s == 0 and not self.st.fresh:
                self.set_ptr(2, dst.addr, force=True)
                s = widx
            if s >= 2:
                self.put("p2inc2")
                self.st.p2i = (self.st.p2i + 2) % 4
                self.st.fresh = True
                s -= 2
            first_inc = s == 1
        for i, (neg, a, b) in enumerate(op.terms):
            if a is not b and self.ptr_cost(0, b.addr) + self.ptr_cost(1, a.addr) < \
                    self.ptr_cost(0, a.addr) + self.ptr_cost(1, b.addr):
                a, b = b, a
            self.set_ptr(0, a.addr)
            self.set_ptr(1, b.addr)
            self.put("macn" if neg else "mac", self.idx(a), self.idx(b), first_inc and i == 0)
        self.stored(dst, False, True)

    def op_Mov(self, op):
        lane = op.lane
        if self.st.mode not in {None: self.wide, "hi": (2, 6), "lo": (3, 7)}[lane]:
            self.set_mode({None: 1, "hi": 2, "lo": 3}[lane])
        self.set_route("fifo" if op.dst is T.FIFO else "ram")
        self.set_ptr(0, op.src.addr)
        inc = False if op.dst is T.FIFO else self.seek(op.dst)
        self.put("movx" if op.swap else "mov", self.idx(op.src), inc)
        if op.dst is not T.FIFO:
            self.stored(op.dst, inc, False)

    def op_Clr(self, op):
        if self.st.mode not in self.wide:
            self.set_mode(1)
        self.set_route("ram")
        inc = self.seek(op.dst)
        self.put("clr.next" if inc else "clr")
        self.stored(op.dst, inc, False)

    def op_IncDec(self, op):
        if self.st.mode not in self.wide:
            self.set_mode(1)
        self.set_route("ram")
        inc = self.seek(op.dst)
        self.put(("dec" if op.dec else "inc") + (".next" if inc else ""))
        self.stored(op.dst, inc, True)          # INC/DEC write R: a MAC would continue from it

    def op_BranchIf(self, op):
        self.put("if f" if op.when else "if nf")
        self.branch(op.label, True)
        self.put("always")
        if self.icm:
            self.set_mode(1, force=True)        # the jump entered mode 4 either way

    def op_Branch(self, op):
        self.branch(op.label, False)
        self.reachable = False

    def branch(self, label, conditional):
        # every f4, taken or not, leaves mode 4 on both cores (MPU: probe95-97), where a low
        # byte is a jump: the next ALU op must set its mode first, on both paths
        st = self.st.copy()
        st.mode = 4
        self.put("jump" if self.icm else "branch", label, target=label)
        self.st.mode = 4
        if label in self.placed:
            self.back.setdefault(label, []).append(st)
        else:
            self.arrivals.setdefault(label, []).append((st, conditional))

    def label(self, op):
        arr = self.arrivals.pop(op.name, [])
        states = [s for s, _ in arr] + ([self.st] if self.reachable else [])
        if not states:
            new = State()
        else:
            new = states[0].copy()
            for s in states[1:]:
                new = join(new, s)
        if op.loop_head:
            # what the backward branches will bring is known only after the body: assume what
            # the last round found (schedule() checks it and iterates), else nothing
            self.fall[op.name] = new
            new = self.heads.get(op.name, State()).copy()
        if op.entry:
            new = State()                   # a routine: reached from call sites in any state
        self.placed.add(op.name)
        self.out.append(Emitted(None, label=op.name, safe=True, loc=self.loc, loop_head=op.loop_head,
                                trips=op.trips, entry=op.entry))
        self.st, self.reachable = new, True
        self.group = False
        if any(c for _, c in arr) or op.loop_head or op.entry:
            self.put("always")
        if self.icm and (arr or op.loop_head or op.entry or self.st.mode in (None, 4)):
            self.set_mode(1, force=True)
        self.group = True

    def op_Call(self, op):
        self.put("call", op.label, target=op.label)
        self.st = State()                   # the routine may leave any pointer, route or mode

    def op_Ret(self, op):
        self.put("ret")
        self.reachable = False

    def op_Irq(self, op):
        self.put("int")

    def op_End(self, op):
        self.put("end")


def schedule(ops, chip: str, rounds: int = 6) -> list:
    """Schedule `ops`, carrying pointer and mode state into loop heads where every way in agrees:
    the code falling in and each backward branch.  The backward branches' states depend on what
    the head assumed, so assume, schedule, check, and weaken the assumption until it holds; if it
    never does, assume nothing at the heads."""
    first = Scheduler(chip)
    out = first.run(ops)
    if not first.fall:
        return out
    heads = {}
    for name, fall in first.fall.items():
        h = fall
        for b in first.back.get(name, []):
            h = join(h, b)
        heads[name] = h
    for _ in range(rounds):
        s = Scheduler(chip)
        o = s.run(ops, heads)
        actual = {}
        for name, fall in s.fall.items():
            h = fall
            for b in s.back.get(name, []):
                h = join(h, b)
            actual[name] = h
        if all(known_in(heads.get(n, State()), actual[n]) for n in actual):
            return o
        heads = {n: join(heads[n], actual[n]) if n in heads else actual[n] for n in actual}
    return out
