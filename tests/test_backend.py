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
    assert any("dmp.div: exact restoring division" in l for l in sl.explain())
    assert not any("lib.py" in l for l in sl.explain())            # the library's insides stay hidden
    bq = program(next(x for x in EXAMPLES if x.stem == "biquad")).build("icm20948")
    assert sum("biquad(): a helper, inlined here" in l for l in bq.explain()) == 1   # merged
    assert "  ;   -> `" in tilt.listing()
