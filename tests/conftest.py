import pathlib
import pytest

from opendmp.emu import Machine, Model, Sample

RE_REPO = pathlib.Path(__file__).resolve().parents[2] / "InvensenseDMP"


def words(b: bytes):
    return [int.from_bytes(b[i:i + 4], "big") for i in range(0, len(b) - 3, 4)]


def run1(code, chip="mpu6050", setup=None, entry=None, sample=None, model=None, passes=1):
    m = Machine(chip, model or Model())
    if entry is not None:
        m.entry = entry
    if setup:
        setup(m)
    m.load(bytes(code) + b"\xff", m.entry)
    rs = [m.run_pass(sample or Sample()) for _ in range(passes)]
    return m, rs[-1] if passes == 1 else rs


@pytest.fixture
def corpus():
    d = RE_REPO / "corpus/images"
    if not d.is_dir():
        pytest.skip("InvensenseDMP corpus not present")
    return d
