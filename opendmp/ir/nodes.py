"""The IR: what a program means, independent of the chip.

A program is a set of scalar variables (states and params that persist, and per-pass temps)
and a structured body of statements over expression trees.  Lists do not exist here: the
front end resolves every constant index, so `z[1][2]` is its own variable.

Values have one of four types:
  Q30   a fixed-point word in [-2, 2); arithmetic wraps
  INT   a raw 32-bit counter (from an int literal)
  BITS  a raw 32-bit mask (dmp.bits)
  S16   a raw sensor sample: a Q30 word whose low half is zero (value = raw / 16384).  Any
        arithmetic makes it Q30; emitting it directly sends its 2-byte int16

The one arithmetic node is `Sum`: a linear combination of terms with integer coefficients,
where a term is a value or an exact product `Prod(a, b)` of two Q30 values.  **A Sum is exact:**
it is computed in full precision (products in Q60), floored once and wrapped once.  That is
what a multiply-accumulate run computes, and it makes the meaning of `K*a + (1-K)*b` exact.  A
lone product is a one-term Sum.
"""
from dataclasses import dataclass, field

Q30, INT, BITS, S16 = "q30", "int", "bits", "s16"
WIDTH = {Q30: 4, INT: 4, BITS: 4, S16: 2}
NUMERIC = (Q30, S16)          # types that take part in Q30 arithmetic


def arith_type(t: str) -> str:
    """The type a value takes in arithmetic: a sensor sample becomes Q30."""
    return Q30 if t == S16 else t


@dataclass(eq=False)
class Var:
    name: str                 # "angle", "z[1][2]", "t.x#3"
    kind: str                 # "state" | "param" | "temp" | "arg" (a sub's parameter)
    type: str
    init: int = 0             # raw load-time value (states and params)
    range: tuple | None = None    # declared (lo, hi), as floats; an assumption
    loc: object = None

    def __repr__(self):
        return f"Var({self.name}:{self.type})"


# ---------------------------------------------------------------- expressions
@dataclass(eq=False)
class Expr:
    loc: object = field(default=None, kw_only=True)


@dataclass(eq=False)
class Const(Expr):
    type: str
    value: int                # raw


@dataclass(eq=False)
class Load(Expr):
    var: Var

    @property
    def type(self):
        return self.var.type


SENSOR_FIELDS = ("ax", "ay", "az", "gx", "gy", "gz", "temp", "fsync")


@dataclass(eq=False)
class Sensor(Expr):
    field: str                # one of SENSOR_FIELDS

    @property
    def type(self):
        return INT if self.field == "fsync" else S16


@dataclass(eq=False)
class Prod:
    """An exact product of two Q30 values; only ever a term of a Sum."""
    a: Expr
    b: Expr


@dataclass(eq=False)
class Term:
    coef: int                 # a non-zero integer
    x: object                 # Expr or Prod


@dataclass(eq=False)
class Sum(Expr):
    type: str                 # Q30 or INT
    terms: list


@dataclass(eq=False)
class MinMax(Expr):
    op: str                   # "min" | "max"
    a: Expr
    b: Expr
    type: str = Q30


@dataclass(eq=False)
class Abs(Expr):
    x: Expr
    type: str = Q30


@dataclass(eq=False)
class Select(Expr):
    cond: object
    a: Expr
    b: Expr
    type: str = Q30


@dataclass(eq=False)
class BitOp(Expr):
    op: str                   # "and" | "xor" | "or"
    a: Expr
    b: Expr
    type: str = BITS


@dataclass(eq=False)
class Shift(Expr):
    op: str                   # "shl" | "shr" (arithmetic)
    x: Expr
    k: int
    type: str = BITS


@dataclass(eq=False)
class Cast(Expr):
    """x's word, read as another type: dmp.raw (Q30 -> Int) and dmp.q30 (Int -> Q30).  The bits
    do not change, so no instruction runs."""
    x: Expr
    type: str


@dataclass(eq=False)
class Assume(Expr):
    """The value of x, which the programmer asserts lies in [lo, hi]."""
    x: Expr
    lo: float
    hi: float

    @property
    def type(self):
        return self.x.type


# ---------------------------------------------------------------- conditions
@dataclass(eq=False)
class Cmp:
    op: str                   # "lt" "le" "gt" "ge" "eq" "ne"
    a: Expr
    b: Expr
    loc: object = None


@dataclass(eq=False)
class BoolOp:
    op: str                   # "and" | "or"
    a: object
    b: object


@dataclass(eq=False)
class Not:
    x: object


# ---------------------------------------------------------------- statements
@dataclass(eq=False)
class Assign:
    var: Var
    value: Expr
    loc: object = None


@dataclass(eq=False)
class If:
    cond: object
    then: list
    els: list
    loc: object = None


@dataclass(eq=False)
class Block:
    """A place `Break(label)` can leave: a helper's body (its `return`), an unrolled
    `for ... in range()` with a `break`, or the whole pass (the program's `return`)."""
    label: str
    body: list


@dataclass(eq=False)
class Loop:
    """`for _ in dmp.loop(count)`: the body runs up to `count` times; `while` has count None
    (the condition is an `If ... Break` at the top of the body)."""
    label: str
    count: int | None
    body: list
    loc: object = None


@dataclass(eq=False)
class Break:
    label: str
    loc: object = None


@dataclass(eq=False)
class Emit:
    fields: list              # [(name, Expr)]
    loc: object = None


@dataclass(eq=False)
class Irq:
    loc: object = None


@dataclass(eq=False)
class Check:
    """`dmp.assume(x, lo, hi)` as a statement."""
    x: Expr
    lo: float
    hi: float
    loc: object = None


@dataclass(eq=False)
class CallSub:
    sub: "SubIR"
    args: list                # [Expr], one per parameter
    outs: list                # [Var], one per result
    loc: object = None


# ---------------------------------------------------------------- units
@dataclass(eq=False)
class Field:
    name: str
    type: str
    offset: int

    @property
    def width(self):
        return WIDTH[self.type]


@dataclass(eq=False)
class SubIR:
    name: str
    params: list              # [Var] kind "arg"
    results: list             # [Var] kind "temp"
    body: list
    loc: object = None


@dataclass(eq=False)
class ProgramIR:
    name: str
    file: str
    opts: dict
    vars: list                # every Var: states and params first, in declaration order
    body: list
    packet: list              # [Field]
    subs: dict = field(default_factory=dict)
    uses: set = field(default_factory=set)    # sensor fields read
    notes: list = field(default_factory=list)  # [(Loc, text)]: how the front end read constructs

    @property
    def states(self):
        return [v for v in self.vars if v.kind == "state"]

    @property
    def params(self):
        return [v for v in self.vars if v.kind == "param"]

    @property
    def packet_size(self):
        return sum(f.width for f in self.packet)
