"""The front end: the examples, the semantics (reference and IR against hand-computed words),
and the rejected constructs with their messages."""
import pathlib
import runpy
import textwrap

import pytest

from opendmp.emu import Sample
from opendmp.emu.gen import generate
from opendmp.ir import nodes as N
from opendmp.ir.ranges import analyse
from opendmp.lang.errors import DmpError
from opendmp.q30 import to_q30

EXAMPLES = sorted((pathlib.Path(__file__).resolve().parent.parent / "examples").glob("*.py"))
MOTION = ["rest", "tilt", "rotate", "walk", "taps", "vibration", "freefall", "shake"]


def programs(path):
    ns = runpy.run_path(str(path))
    return [v for v in ns.values() if type(v).__name__ == "Program"]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example(path):
    """Every example lowers, has no range warning, and the two engines agree bit for bit on
    every kind of motion, with no wrap and no failed assumption."""
    (p,) = programs(path)
    assert analyse(p.ir).warnings == []
    s = generate(MOTION, rate=p.opts["rate"], seconds=0.5)
    a, b = p.simulate(s), p.simulate(s, "ir")
    assert a.packets == b.packets and a.irqs == b.irqs
    assert len(a.packets[0]) == p.ir.packet_size
    assert not a.wraps and not a.assume_fails and not b.assume_fails


# ------------------------------------------------------------------ building test programs
@pytest.fixture
def mk(tmp_path):
    n = [0]

    def make(body, head="", rate=100, args="s"):
        n[0] += 1
        src = "import math\nimport opendmp as dmp\n" + textwrap.dedent(head) + \
            f"\n\n@dmp.program(rate={rate})\ndef prog({args}):\n" + \
            textwrap.indent(textwrap.dedent(body), "    ")
        f = tmp_path / f"p{n[0]}.py"
        f.write_text(src)
        return runpy.run_path(str(f))["prog"]
    return make


def run(p, samples=None, n=1, **kw):
    """Both engines; they must agree.  Returns the reference result."""
    samples = samples or [Sample(ax=16384, ay=-8192, az=4096, gx=100, gy=-200, gz=300, temp=-2000)] * n
    a, b = p.simulate(samples, **kw), p.simulate(samples, "ir", **kw)
    assert a.packets == b.packets and a.irqs == b.irqs, (a.fields(), b.fields())
    return a


def words(res, k=-1):
    return res.fields()[k]


def raw(res, name, k=-1):
    f = next(f for f in res.packet if f.name == name)
    b = res.packets[k][f.offset:f.offset + f.width]
    return int.from_bytes(b, "big", signed=True)


# ------------------------------------------------------------------ semantics
def test_a_sum_is_exact_and_floored_once(mk):
    p = mk("""
        a: dmp.state = 2.0 ** -30          # 1 LSB
        b: dmp.state = 2.0 ** -30
        one = 0.5 * a
        both = 0.5 * a + 0.5 * b           # exact: 1 LSB (per-product floors would give 0)
        nested = (0.5 * a) + (0.5 * b)     # parentheses do not split a sum
        neg = -0.5 * a                     # floor(-0.5 LSB) = -1 LSB
        dmp.emit(one=one, both=both, nested=nested, neg=neg)
    """)
    r = run(p)
    assert [raw(r, k) for k in ("one", "both", "nested", "neg")] == [0, 1, 1, -1]


def test_integer_multiples_are_exact_terms(mk):
    p = mk("""
        x: dmp.state = 2.0 ** -30
        h: dmp.state = 0.75
        exact = 2 * (x * h)                # 2 * 0.75 LSB = 1.5 -> 1 (floor(0.75)*2 would be 0)
        split = 2 * x * h                  # (2x) * h: 2 LSB * 0.75 = 1.5 -> 1
        dmp.emit(exact=exact, split=split)
    """)
    r = run(p)
    assert raw(r, "exact") == 1 and raw(r, "split") == 1


def test_wrapping(mk):
    p = mk("""
        x: dmp.state = 1.5
        m2: dmp.state = -2.0
        add = x + x                        # 3.0 wraps to -1.0
        mul = x * x                        # 2.25 wraps to -1.75
        tri = 3 * x - 3 * x + x + x        # exact inside, 3.0 at the end: wraps once, to -1.0
        ab = abs(m2)                       # abs(-2.0) = -2.0
        dmp.emit(add=add, mul=mul, tri=tri, ab=ab)
    """)
    r = run(p)
    f = words(r)
    assert (f["add"], f["mul"], f["tri"], f["ab"]) == (-1.0, -1.75, -1.0, -2.0)
    assert sum(r.wraps.values()) == 4                  # the reference saw all four wrap


def test_types_constants_and_sensors(mk):
    p = mk("""
        n: dmp.state = 2147483647
        n += 1                             # an Int counter wraps like the word it is
        c = 0.1                            # rounded to nearest once
        k = s.accel.x * 0.5                # a sensor sample is raw / 16384
        dmp.emit(n, c, s.accel.x, k, t=s.temp)
    """)
    r = run(p)
    assert [f.type for f in r.packet] == [N.INT, N.Q30, N.S16, N.Q30, N.S16]
    assert r.packet[-1].name == "t" and r.packet[2].name == "accel_x"
    f = words(r)
    assert raw(r, "n") == -(1 << 31) and raw(r, "c") == to_q30(0.1)
    assert raw(r, "accel_x") == 16384 and f["k"] == 0.5 and raw(r, "t") == -2000


def test_min_max_clamp_select_and_compare(mk):
    p = mk("""
        lo = min(s.accel.x, s.accel.y, 0.25)
        hi = max(s.accel.x, s.accel.y)
        cl = dmp.clamp(s.accel.x * 1.5, -1, 1)          # int bounds meet a Q30 value
        pick = s.accel.x if s.accel.x > s.accel.y else s.accel.y
        flag = 1 if s.accel.x - s.accel.y > 1.25 else 0  # exact: 1.5 > 1.25
        dmp.emit(lo=lo, hi=hi, cl=cl, pick=pick, flag=flag)
    """)
    f = words(run(p))
    assert (f["lo"], f["hi"], f["cl"], f["pick"], f["flag"]) == (-0.5, 1.0, 1.0, 1.0, 1)


def test_control_flow(mk):
    p = mk("""
        x = 0.0
        for _ in dmp.loop(10):
            x = x + 0.125
            if x >= 0.5:
                break
        n = 0
        while n < 3:
            n += 1
        acc = 0.0
        for i in range(4):
            if i == 2:
                break
            acc = acc + FRAC[i]
        dmp.emit(x=x, n=n, acc=acc)
    """, head="FRAC = [0.25, 0.5, 1.0, 1.5]")
    f = words(run(p))
    assert (f["x"], f["n"], f["acc"]) == (0.5, 3, 0.75)
    assert isinstance(p.ir.body[1], N.Loop)


def test_helpers_subs_lists_and_returns(mk):
    p = mk("""
        z: dmp.state = [[0.0, 0.0], [0.0, 0.0]]
        row = z[1]                         # a named list is shared
        row[0] = 0.5
        cp = [z[1][0], z[1][1]]            # a list display is a copy
        cp[1] = 0.75
        a, b = split(s.accel.x)
        c = first_positive(s.accel.y, s.accel.x)
        d = twice(0.25)
        dmp.emit(z10=z[1][0], z11=z[1][1], a=a, b=b, c=c, d=d)
        if s.accel.x > 0:
            dmp.irq()
            return
        dmp.irq()
    """, head='''
        def split(x):
            return x * 0.25, x * 0.75

        def first_positive(a, b):
            if a > 0:
                return a
            return b

        @dmp.sub
        def twice(v):
            return v + v
    ''')
    r = run(p, n=2)
    f = words(r)
    assert (f["z10"], f["z11"], f["a"], f["b"], f["c"], f["d"]) == (0.5, 0.0, 0.25, 0.75, 1.0, 0.5)
    assert r.irqs == [True, True] and list(p.ir.subs) == ["twice"]


def test_params_and_bits(mk):
    p = mk("""
        gain: dmp.param = 0.5
        m: dmp.state = dmp.bits(0xF0)
        m = (m ^ MASK) << 4
        dmp.emit(y=s.accel.x * gain, m=m)
    """, head="MASK = dmp.bits(0xFF)")
    r = run(p)
    assert words(r)["y"] == 0.5 and raw(r, "m") == 0x0F0
    r = run(p, params={"gain": 0.25})
    assert words(r)["y"] == 0.25


def test_assumptions_are_checked(mk):
    p = mk("""
        x = dmp.assume(s.accel.x, -0.5, 0.5)
        dmp.assume(s.accel.y, -0.25, 0.25)
        dmp.emit(x=x)
    """)
    r = run(p)
    assert sorted(loc.line for loc in r.assume_fails) == [8, 9]


def test_range_warning_names_the_fix(mk):
    p = mk("""
        acc: dmp.state = 0.0
        acc = acc + s.gyro.z * 0.5
        dmp.emit(acc=acc)
    """)
    (w,) = p.check()
    assert "can reach" in w.msg and "stored in acc" in w.msg and "declare a range on `acc`" in w.hint
    q = mk("""
        acc: dmp.state(-1.0, 1.0) = 0.0
        acc = dmp.clamp(acc + s.gyro.z * 0.25, -1.0, 1.0)
        dmp.emit(acc=acc)
    """)
    assert q.check() == []


# ------------------------------------------------------------------ what is refused
REJECTS = [
    ("x = s.accel.x / s.accel.y\ndmp.emit(x=x)", "division by a variable", "reciprocal"),
    ("x = math.sqrt(s.accel.x)\ndmp.emit(x=x)", "needs constant arguments", "Newton"),
    ("x = s.accel.x ** 2\ndmp.emit(x=x)", "** is not supported", "x * x"),
    ("h: dmp.state = [0.0, 0.0]\nfor _ in dmp.loop(2):\n    x = h[_]\ndmp.emit(h0=h[0])",
     "list index must be a constant", "unroll"),
    ("for k in dmp.loop(2):\n    x = k * 1.0\ndmp.emit(a=s.accel.x)", "the dmp.loop variable has no value",
     "range(N)"),
    ("h: dmp.state = [0.0, 0.0]\ni = 1\ndmp.emit(x=h[i])", "list index must be a constant",
     "no indirect addressing"),
    ("dmp.emit(x=f(s.accel.x))", "calls itself", "inlined"),
    ("if s.accel.x > 0:\n    dmp.emit(x=s.accel.x)", "dmp.emit under a condition", "fixed length"),
    ("x = s.accel.x << 2\ndmp.emit(x=x)", "<< and >> work on dmp.bits", "x * 4"),
    ("k: dmp.param = 0.5\nk = 0.25\ndmp.emit(k=k)", "is a dmp.param", "copy it"),
    ("n: dmp.state = 0\nn = n + s.accel.x\ndmp.emit(n=n)", "mixes an Int and a Q30", "int literal"),
    ("x = s.accel.x * 2.5\ndmp.emit(x=x)", "outside the Q30 range", "integer 2"),
    ("dmp.emit(x=nope)", "unknown name 'nope'", None),
    ("try:\n    x = 1.0\nexcept Exception:\n    x = 0.0\ndmp.emit(x=x)", "try/except is not supported",
     None),
    ("for i in range(2):\n    continue\ndmp.emit(x=s.accel.x)", "continue is not supported", "if/else"),
    ("f = s.accel.x > 0\ndmp.emit(f=f)", "a comparison is not a value", "1 if"),
    ("print(s.accel.x)", "print is not available", None),
    ("x = 'a'\ndmp.emit(x=x)", "is not a DMP value", None),
    ("x = True\ndmp.emit(x=x)", "is not a DMP value", "1 and 0"),
    ("if s.accel.x > 0:\n    y = 1.0\ndmp.emit(y=y)", "may be used before it is assigned", None),
    ("dmp.emit(x=s.foo)", "the sample has no 'foo'", "accel, gyro"),
    ("for x in [1, 2]:\n    pass\ndmp.emit(a=s.accel.x)", "a DMP loop is", "range(len"),
    ("q: dmp.state = [1.0, 0.0]\ndmp.emit(q=q)", "is a list", "q0="),
    ("dmp.emit(x=s.accel.x)\ndmp.emit(x=s.accel.y)", "already has a field named 'x'", None),
    ("x: dmp.state = 0.0\nx = 1\ndmp.emit(x=x)", "holds a Q30 value, but this is an Int", "int literal"),
    ("dmp.emit(x=g(s.accel.x))", "plain positional parameters", None),
    ("def h(v):\n    return v\ndmp.emit(x=s.accel.x)", "a nested def is not supported", "module level"),
    ("return 1.0", "returns nothing", "dmp.emit"),
    ("x = sb([s.accel.x, s.accel.y])\ndmp.emit(x=x)", "takes values, not lists", None),
    ("dmp.emit(x=dmp.clamp(s.accel.x, s.accel.y, 1.0))", "must be a constant", None),
    ("assert s.accel.x > 0\ndmp.emit(x=s.accel.x)", "assert is not supported", "dmp.assume"),
    ("x: dmp.state\ndmp.emit(x=s.accel.x)", "needs an initial value", None),
    ("dmp.emit()", "nothing to send", None),
    ("n: dmp.state = 3\ndmp.emit(x=n * s.accel.x)", "Int values only scale by a constant integer", None),
    ("dmp.emit(x=s.accel.x / 0)", "division by 0", None),
    ("f = lambda v: v\ndmp.emit(x=s.accel.x)", "lambda is not supported", None),
    ("h: dmp.state = [0.0, 0.0]\ndmp.emit(x=h[0:1])", "a slice is not supported", "constant"),
    ("x: float = 0.0\ndmp.emit(x=x)", "annotate a declaration with dmp.state", "no annotation"),
    ("x = s.accel.x\nif s.accel.y > 0:\n    y: dmp.state = 0.0\ndmp.emit(x=x)",
     "declare dmp.state and dmp.param at the top", None),
    ("x = s.accel.x * dmp.bits(3)\ndmp.emit(x=x)", "dmp.bits", None),
    ("x = s.accel.x & s.accel.y\ndmp.emit(x=x)", "& ^ | work on dmp.bits", "dmp.bits(0x"),
]
REJECT_HEAD = """
def f(v):
    return f(v)

def g(v, w=1.0):
    return v

@dmp.sub
def sb(v):
    return v
"""


@pytest.mark.parametrize("body,msg,hint", REJECTS, ids=[r[1][:40] for r in REJECTS])
def test_rejected(mk, body, msg, hint):
    p = mk(body, head=REJECT_HEAD)
    with pytest.raises(DmpError) as e:
        p.ir
    assert msg in e.value.msg, str(e.value)
    if hint:
        assert e.value.hint and hint in e.value.hint, str(e.value)
    assert e.value.loc is not None and "^" in str(e.value)


def test_program_signature(mk):
    p = mk("dmp.emit(x=s.accel.x)", args="s, t")
    with pytest.raises(DmpError, match="takes one argument"):
        p.ir


def test_error_points_at_the_construct(mk):
    p = mk("x = s.accel.x + s.accel.y / s.accel.z\ndmp.emit(x=x)")
    with pytest.raises(DmpError) as e:
        p.ir
    loc = e.value.loc
    assert (loc.line, loc.col) == (7, 20)              # the division, not the whole line
    assert str(e.value).splitlines()[2].strip() == "^" * len("s.accel.y / s.accel.z")


def test_program_object():
    import opendmp as dmp

    @dmp.program(rate=50)
    def p(s):
        dmp.emit(x=s.accel.x)
    with pytest.raises(TypeError, match="not called"):
        p(None)
    with pytest.raises(ValueError, match="gyro_fs"):
        dmp.program(rate=50, gyro_fs=300)(lambda s: None)
    with pytest.raises(RuntimeError, match="compiled, not run"):
        dmp.emit(1.0)
    assert p.ir.packet[0].name == "x" and p.ir.packet_size == 2
    assert p.simulate([Sample(ax=-1)]).packets == [b"\xff\xff"]


def test_cli(tmp_path, capsys):
    from opendmp.cli import main
    ex = pathlib.Path(__file__).resolve().parent.parent / "examples"
    assert main(["check", str(ex / "tilt.py")]) == 0
    out = capsys.readouterr().out
    assert "packet 6 B: angle q30, accel_z i16" in out and "nothing can wrap" in out
    csv_out = tmp_path / "p.csv"
    assert main(["sim", str(ex / "pedometer.py"), "--gen", "walk", "--seconds", "2", "-o", str(csv_out)]) == 0
    rows = csv_out.read_text().splitlines()
    assert rows[0] == "pass,irq,steps,m" and len(rows) == 201
    assert main(["ir", str(ex / "taps.py")]) == 0
    assert "emit event=event" in capsys.readouterr().out
    bad = tmp_path / "bad.py"
    bad.write_text("import opendmp as dmp\n\n@dmp.program(rate=10)\ndef p(s):\n    dmp.emit(x=s.accel.x / s.gyro.x)\n")
    assert main(["check", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "bad.py:5:16: error: division by a variable" in err and "hint:" in err
