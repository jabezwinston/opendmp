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
    check(p, chip, generate(MOTION, rate=p.opts["rate"], seconds=0.25))


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
