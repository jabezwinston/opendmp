"""The backends: every example builds for both chips, and its bytes, run in the emulator, give
the reference semantics' packets and interrupts bit for bit."""
import pathlib
import random
import runpy

import pytest

from opendmp.emu import Sample
from opendmp.emu.gen import generate
from opendmp.isa import assemble

EXAMPLES = sorted((pathlib.Path(__file__).resolve().parent.parent / "examples").glob("*.py"))
MOTION = ["rest", "tilt", "rotate", "walk", "taps", "vibration", "freefall", "shake"]
CHIPS = ("mpu6050", "icm20948")


def program(path):
    ns = runpy.run_path(str(path))
    (p,) = [v for v in ns.values() if type(v).__name__ == "Program"]
    return p


def check(p, chip, samples):
    ref = p.simulate(samples)
    emu = p.build(chip).emulate(samples)
    for i, (a, b) in enumerate(zip(ref.packets, emu.packets)):
        assert a == b, f"pass {i}: reference {ref.fields()[i]} != emulator {emu.fields()[i]}"
    assert ref.irqs == emu.irqs
    return emu


@pytest.mark.parametrize("chip", CHIPS)
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_bit_exact(path, chip):
    p = program(path)
    emu = check(p, chip, generate(MOTION, rate=p.opts["rate"], seconds=0.25))
    b = p.build(chip)
    assert b.cycles_max() is not None and b.cycles_max() >= max(emu.cycles)   # the bound holds
    if b.bounded and not any(e.insn is None and e.loop_head for e in b.out):
        assert b.cycles_max() <= b.cycles_straight()


@pytest.mark.parametrize("chip", CHIPS)
def test_tilt_10k_random_samples(chip):
    """M1's acceptance: tilt on 10,000 passes of random full-scale samples."""
    p = program(EXAMPLES[[x.stem for x in EXAMPLES].index("tilt")])
    rnd = random.Random(7)
    r = lambda: rnd.randint(-32768, 32767)
    samples = [Sample(r(), r(), r(), r(), r(), r(), temp=r()) for _ in range(10000)]
    check(p, chip, samples)


@pytest.mark.parametrize("chip", CHIPS)
@pytest.mark.parametrize("name", ["hello_fifo", "tilt", "taps", "pedometer"])
def test_asm_reassembles(name, chip):
    """The .asm output assembles back to the image's code and data."""
    b = program(next(x for x in EXAMPLES if x.stem == name)).build(chip)
    prog = assemble(b.asm())
    img = prog.image(len(b.image))
    assert bytes(img[b.entry:b.entry + b.size]) == b.code
    for w in b.words:
        if w.kind in ("state", "param", "const"):
            assert img[w.addr:w.addr + 4] == b.image[w.addr:w.addr + 4]


def test_listing_and_schema():
    b = program(next(x for x in EXAMPLES if x.stem == "tilt")).build("mpu6050")
    lst = b.listing()
    assert "tilt.py:33" in lst and "; packet 6 B" in lst and "fmode 0" in lst
    s = b.schema()
    assert s["packet"][1] == {"name": "accel_z", "type": "s16", "offset": 4, "width": 2}
    assert s["symbols"]["limit"]["kind"] == "param" and s["entry"] == 0x400


def test_params_reach_the_bytes():
    p = program(next(x for x in EXAMPLES if x.stem == "tilt"))
    s = generate(["tilt"], rate=200, seconds=1)
    for chip in CHIPS:
        a = p.simulate(s, params={"limit": 0.1})
        b = p.build(chip).emulate(s, params={"limit": 0.1})
        assert a.irqs == b.irqs and sum(a.irqs) > sum(p.simulate(s).irqs)


@pytest.mark.parametrize("chip", CHIPS)
def test_trace_build_carries_its_samples(chip):
    """A trace build's packets start with the pass's raw sample words; decoding them gives the
    samples back, so a run on silicon can be replayed (opendmp.host.verify)."""
    p = program(next(x for x in EXAMPLES if x.stem == "tilt"))
    b = p.build(chip, trace=True)
    s = generate(["shake", "walk"], rate=200, seconds=0.1)
    emu = b.emulate(s)
    assert [b.trace_sample(pk) for pk in emu.packets] == s
    ref = p.simulate(s)
    assert [pk[b.trace_bytes:] for pk in emu.packets] == ref.packets


WHILE_SRC = '''
import opendmp as dmp

@dmp.program(rate=100)
def w(s):
    n = 0
    x = s.accel.x * 0.5
    while x < 0.25:                  # data-dependent trips: a backward branch on both cores
        x = x + 0.0625
        n += 1
    k = 0
    while k < 40:                    # a body the MPU's 32-byte reach needs no hop for
        k += 1
        x = x * 0.99
    dmp.emit(n=n, x=x)
'''


@pytest.mark.parametrize("chip", CHIPS)
def test_while_on_both_cores(tmp_path, chip):
    f = tmp_path / "w.py"
    f.write_text(WHILE_SRC)
    p = runpy.run_path(str(f))["w"]
    check(p, chip, generate(["shake", "walk", "tilt"], rate=100, seconds=0.5))
    assert p.build(chip).cycles_max() is None and "no bound" in p.build(chip).listing()


def test_loop_bound_under_worst_case_inputs():
    """unit_vectors' trips depend on the data: 1-LSB vectors take all 15 doublings."""
    p = program(next(x for x in EXAMPLES if x.stem == "unit_vectors"))
    rnd = random.Random(5)
    ss = [Sample(1, 1, 1, 1, 1, 1), Sample(0, 0, 0, 0, 0, 0), Sample(-1, 0, 0, 0, 0, -1),
          Sample(32767, 32767, 32767, -32768, -32768, -32768)]
    ss += [Sample(*(rnd.randint(-m, m) for _ in range(6))) for m in (1, 2, 3, 5, 8, 16, 64) for _ in range(20)]
    for chip in CHIPS:
        emu = check(p, chip, ss)
        assert max(emu.cycles) <= p.build(chip).cycles_max() <= 1.15 * max(emu.cycles)


FSYNC_SRC = '''
import opendmp as dmp

@dmp.program(rate=50)
def sync(s):
    edges: dmp.state = 0
    last: dmp.state = 0
    f = s.fsync
    if f == 1:
        if last == 0:
            edges += 1
    last = f
    dmp.emit(fsync=f, edges=edges, temp=s.temp)
'''


@pytest.mark.parametrize("chip", CHIPS)
def test_fsync(tmp_path, chip):
    """s.fsync is the temperature's bit 0 with EXT_SYNC_SET = 1, which then also carries it in
    s.temp; a trace build's replay recovers it from there."""
    f = tmp_path / "sync.py"
    f.write_text(FSYNC_SRC)
    p = runpy.run_path(str(f))["sync"]
    rnd = random.Random(3)
    ss = [Sample(temp=rnd.randint(-32768, 32767), fsync=rnd.randint(0, 1)) for _ in range(200)]
    check(p, chip, ss)
    assert p.simulate(ss, "ir").packets == p.simulate(ss).packets
    b = p.build(chip, trace=True)
    assert b.fsync and [s.fsync for s in map(b.trace_sample, b.emulate(ss).packets)] == [s.fsync for s in ss]


def test_div_is_exact_everywhere():
    """dmp.div: truncated toward zero, exact to the last bit, saturating at |q| >= 2 and y = 0,
    for every input including -2.0 -- on the reference, and on both chips' bytes."""
    p = program(next(x for x in EXAMPLES if x.stem == "slope"))
    edge = [-32768, -32767, -16384, -1, 0, 1, 16383, 16384, 32767]
    rnd = random.Random(11)
    ss = [Sample(ax=a, az=z) for a in edge for z in edge]
    ss += [Sample(ax=rnd.randint(-32768, 32767), az=rnd.randint(-32768, 32767)) for _ in range(150)]
    ref = p.simulate(ss)
    assert not ref.assume_fails
    for s, pk in zip(ss, ref.packets):
        x, y = s.ax << 16, s.az << 16
        neg = (x < 0) != (y < 0)
        if y == 0 or abs(x) >= 2 * abs(y):
            want = (1 << 31) - 1
        else:
            want = (abs(x) << 30) // abs(y)
        assert int.from_bytes(pk[:4], "big", signed=True) == (-want if neg else want), (s.ax, s.az)
    for chip in CHIPS:
        check(p, chip, ss)


def test_method_notes():
    """`opendmp build` says, per construct and line, how it was compiled."""
    tilt = program(next(x for x in EXAMPLES if x.stem == "tilt")).build("mpu6050")
    lines = tilt.explain()
    assert any(l.startswith("tilt.py:33") and "one CLAMP instruction" in l for l in lines)
    assert any(l.startswith("tilt.py:35") and "RSUB" in l for l in lines)
    uv = program(next(x for x in EXAMPLES if x.stem == "unit_vectors"))
    assert any("unit(): one routine, called here" in l for l in uv.build("icm20948").explain())
    assert any("unit(): inlined here (the MPU has no call)" in l for l in uv.build("mpu6050").explain())
    assert any("at most 15 trips" in l for l in uv.build("mpu6050").explain())
    sl = program(next(x for x in EXAMPLES if x.stem == "slope")).build("mpu6050")
    assert any("x / y (dmp.div): exact restoring division" in l for l in sl.explain())
    assert any("dmp.div" in l and "costs up to" in l and " us each" in l for l in sl.explain())
    assert not any("lib.py" in l for l in sl.explain())            # the library's insides stay hidden
    bq = program(next(x for x in EXAMPLES if x.stem == "biquad")).build("icm20948")
    assert sum("biquad(): a helper, inlined here" in l for l in bq.explain()) == 1   # merged
    assert "  ;   -> `" in tilt.listing()


FLOORDIV_SRC = '''
import opendmp as dmp

@dmp.program(rate=50)
def fd(s):
    n: dmp.param = 0
    m: dmp.param = 1
    dmp.emit(q=n // m)
'''


def test_floor_division_matches_python(tmp_path):
    """n // m on Int values: Python's floor division for every 32-bit pair; // 0 and the one
    overflow (-2^31 // -1) saturate."""
    f = tmp_path / "fd.py"
    f.write_text(FLOORDIV_SRC)
    p = runpy.run_path(str(f))["fd"]
    L = 1 << 31
    edge = [-L, -L + 1, -7, -2, -1, 0, 1, 2, 3, 7, L - 1]
    builds = [p.build(c) for c in CHIPS]
    for a in edge:
        for b in edge:
            want = (L - 1 if a >= 0 else -L) if b == 0 else min(a // b, L - 1)
            ref = p.simulate([Sample()], params={"n": a, "m": b})
            assert int.from_bytes(ref.packets[0], "big", signed=True) == want, (a, b)
            for bd in builds:
                assert bd.emulate([Sample()], params={"n": a, "m": b}).packets == ref.packets, (a, b)
    assert any("n // m (dmp.idiv): exact floor division" in l for l in builds[0].explain())


IMUL_SRC = '''
import opendmp as dmp

@dmp.program(rate=50)
def im(s):
    n: dmp.param = 0
    m: dmp.param = 0
    dmp.emit(p=n * m, q=-n * m)
'''


def test_int_product_matches_python(tmp_path):
    """n * m on two Int values (dmp.imul, shift-and-add): Python's product modulo 2^32."""
    f = tmp_path / "im.py"
    f.write_text(IMUL_SRC)
    p = runpy.run_path(str(f))["im"]
    L = 1 << 31
    s32 = lambda v: (v + L) % (1 << 32) - L      # noqa: E731
    edge = [-L, -L + 1, -65536, -7, -1, 0, 1, 3, 46341, L - 1]
    builds = [p.build(c) for c in CHIPS]
    for a in edge:
        for b in edge:
            ref = p.simulate([Sample()], params={"n": a, "m": b})
            pk = ref.packets[0]
            got = [int.from_bytes(pk[i:i + 4], "big", signed=True) for i in (0, 4)]
            assert got == [s32(a * b), s32(-a * b)], (a, b)
            for bd in builds:
                assert bd.emulate([Sample()], params={"n": a, "m": b}).packets == ref.packets, (a, b)
    assert any("(dmp.imul): the exact Int product" in l for l in builds[0].explain())


SCALE_SRC = '''
import opendmp as dmp

@dmp.program(rate=50)
def sc(s):
    n: dmp.state = 3
    y: dmp.state = 0.0
    n = n * 100 + 1
    a = dmp.raw(s.accel.x)
    up = y + dmp.q30(3)
    y = s.accel.y * 7 + y * 3 - s.gyro.x * s.gyro.y + s.accel.z * 9
    dmp.emit(n=n, m=7 * a - 3 * n, w=-a * 5, z=s.gyro.z * -3, k=s.gyro.z * 1000, up=up, y=y,
             back=dmp.q30(a))
'''


def test_integer_scaling_and_casts(tmp_path):
    """Constant integer coefficients by doubling (Horner), plain terms into a MAC run once,
    and dmp.raw / dmp.q30 as free reinterpretations: bit-exact on both chips."""
    f = tmp_path / "sc.py"
    f.write_text(SCALE_SRC)
    p = runpy.run_path(str(f))["sc"]
    rnd = random.Random(5)
    r16 = lambda: rnd.randint(-32768, 32767)     # noqa: E731
    ss = [Sample(ax=r16(), ay=r16(), az=r16(), gx=r16(), gy=r16(), gz=r16()) for _ in range(60)]
    ss += [Sample(ax=-32768, ay=-32768, az=-32768, gx=-32768, gy=-32768, gz=-32768)]
    ref = p.simulate(ss)
    for s, fl in zip(ss, ref.fields()):
        assert fl["w"] == ((-(s.ax << 16) * 5 + (1 << 31)) % (1 << 32)) - (1 << 31)
        assert fl["back"] == pytest.approx(s.ax / 16384)
    for chip in CHIPS:
        check(p, chip, ss)
    notes = p.build("mpu6050").explain()
    assert any("sc.py:8" in l and "ADD/RSUB by doubling" in l for l in notes)
    assert any("sc.py:9" in l and "no instruction" in l for l in notes)
    assert any("sc.py:11" in l and "entering the multiply-accumulate run once" in l for l in notes)
    assert not any("multiply-accumulate run of" in l and "sc.py:12" in l for l in notes)


INCMAC_SRC = '''
import opendmp as dmp

@dmp.program(rate=50)
def incmac(s):
    x: dmp.state = 0.25
    g: dmp.param = 0.5
    x = x + dmp.q30(1)
    x = x * g + s.accel.x * g
    dmp.emit(x=x)
'''


@pytest.mark.parametrize("chip", CHIPS)
def test_inc_then_mac_into_the_same_word(tmp_path, chip):
    """An INC (or DEC) writes R, and on the MPU a MAC after a P2 move continues from R
    (InvensenseDMP Finding 54): a MAC run into the word an INC just changed must still start a
    fresh sum.  emulate == reference on both chips."""
    f = tmp_path / "incmac.py"
    f.write_text(INCMAC_SRC)
    p = runpy.run_path(str(f))["incmac"]
    check(p, chip, generate(["walk", "tilt"], rate=50, seconds=0.2))


SHIFT_KS = (0, 1, 2, 15, 16, 17, 29, 30, 31)
SHIFT_SRC = '''
import opendmp as dmp

@dmp.program(rate=50)
def sh(s):
    b: dmp.param = dmp.bits(0)
    dmp.emit(''' + ", ".join(f"r{k}=b >> {k}, l{k}=b << {k}" for k in SHIFT_KS) + ''')
'''


@pytest.mark.parametrize("chip", CHIPS)
def test_shifts_every_count(tmp_path, chip):
    """>> k (k = 31: two MULs) and << k (in place; MPU k >= 16: a lane MOVX first; DMP3 k >= 2:
    MUL then SHL 30) against Python's arithmetic shifts on a 32-bit word."""
    f = tmp_path / "sh.py"
    f.write_text(SHIFT_SRC)
    p = runpy.run_path(str(f))["sh"]
    bd = p.build(chip)
    L = 1 << 31
    s32 = lambda v: (v + L) % (1 << 32) - L      # noqa: E731
    for v in (0, 1, -1, 3, -7, 0x12345678, s32(0x80000001), L - 1, -L):
        ref = p.simulate([Sample()], params={"b": v})
        pk = ref.packets[0]
        got = [int.from_bytes(pk[i:i + 4], "big", signed=True) for i in range(0, len(pk), 4)]
        want = [x for k in SHIFT_KS for x in (v >> k, s32(v << k))]
        assert got == want, v
        assert bd.emulate([Sample()], params={"b": v}).packets == ref.packets, v


@pytest.mark.parametrize("chip", CHIPS)
def test_shift_left_uses_no_temp_per_doubling(tmp_path, chip):
    """<< 31 used to allocate a word per doubling (30 extra data words)."""
    f = tmp_path / "sh2.py"
    f.write_text(SHIFT_SRC.replace(", ".join(f"r{k}=b >> {k}, l{k}=b << {k}" for k in SHIFT_KS),
                                   "x=b << 31, y=b << 15"))
    bd = runpy.run_path(str(f))["sh"].build(chip)
    assert len([w for w in bd.words if w.kind == "temp"]) <= 3


def test_dmp3_int_product_is_two_instructions(tmp_path):
    """DMP3 computes n * m as MUL then SHL 30 (Finding 46): random pairs, Python's product."""
    f = tmp_path / "im.py"
    f.write_text(IMUL_SRC)
    p = runpy.run_path(str(f))["im"]
    bd = p.build("icm20948")
    assert any("MUL then SHL 30" in l for l in bd.explain())
    assert "shl 30" in bd.asm() and bd.cycles_max() < 400
    assert assemble(bd.asm()).code() == bytes(bd.image[bd.entry:bd.entry + len(assemble(bd.asm()).code())])
    rnd = random.Random(46)
    L = 1 << 31
    s32 = lambda v: (v + L) % (1 << 32) - L      # noqa: E731
    for _ in range(300):
        a, b = s32(rnd.getrandbits(32)), s32(rnd.getrandbits(32))
        got = bd.emulate([Sample()], params={"n": a, "m": b}).packets[0]
        assert [int.from_bytes(got[i:i + 4], "big", signed=True) for i in (0, 4)] == [s32(a * b), s32(-a * b)]
