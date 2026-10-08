"""The host runtime's state and param access, against a stand-in for the bench bridge."""
import pathlib
import runpy

import pytest

from opendmp.host.rig import Rig

EX = pathlib.Path(__file__).resolve().parent.parent / "examples"


class FakeBridge:
    def __init__(self, image):
        self.mem = bytearray(image)

    def mem_read(self, addr, n):
        return bytes(self.mem[addr:addr + n])

    def mem_write(self, addr, data):
        self.mem[addr:addr + len(data)] = data


def rig_for(name, chip="mpu6050"):
    ns = runpy.run_path(str(EX / f"{name}.py"))
    p = [v for v in ns.values() if type(v).__name__ == "Program"][0]
    b = p.build(chip)
    rig = Rig.__new__(Rig)
    rig.b, rig.chip, rig.build, rig.watched = FakeBridge(b.image), chip, b, []
    return rig


def test_read_write_by_name():
    rig = rig_for("tilt")
    assert rig.read("limit") == 0.25 and rig.read("angle") == 0.0
    rig.write("limit", -0.5)
    assert rig.read("limit") == -0.5
    addr = next(w.addr for w in rig.build.words if w.name == "limit")
    assert rig.b.mem[addr:addr + 4] == bytes.fromhex("e0000000")
    with pytest.raises(KeyError, match="not a state or param"):
        rig.read("K")
    with pytest.raises(ValueError, match=r"\[-2, 2\)"):
        rig.write("limit", 2.0)


def test_declared_param_range_is_enforced():
    rig = rig_for("pid_level", "icm20948")
    rig.write("setpoint", 0.3)
    assert rig.read("setpoint") == pytest.approx(0.3, abs=1e-9)
    with pytest.raises(ValueError, match="declared range"):
        rig.write("setpoint", 0.31)


def test_int_state():
    rig = rig_for("pedometer")
    ints = [w.name for w in rig.build.words if w.kind == "state" and w.var.type == "int"]
    assert ints
    rig.write(ints[0], 7)
    assert rig.read(ints[0]) == 7


def test_run_actions_and_changes_are_validated_first():
    rig = rig_for("pid_level")
    with pytest.raises(ValueError, match="declared range"):
        rig.run(1.0, changes=[(0.5, "setpoint", 0.9)])       # refused before anything starts


class RecordingBridge:
    """Records register writes (and ICM bank selects) as (bank, reg, value)."""
    def __init__(self):
        self.log, self._bank = [], 0

    def bank(self, n):
        self._bank = n

    def wr(self, reg, data):
        self.log.append((self._bank, reg, data))


@pytest.mark.parametrize("chip,want", [
    ("mpu6050", [(0, 0x6A, 0x48), (0, 0x6A, 0x40)]),
    ("icm20948", [(0, 0x03, 0x48), (0, 0x03, 0x40)]),
])
def test_stop_halts_with_dmp_reset(chip, want):
    """Clearing DMP_EN does not stop a running pass; DMP_RST does (InvensenseDMP Finding 47)."""
    rig = Rig.__new__(Rig)
    rig.b, rig.chip = RecordingBridge(), chip
    rig.stop()
    assert rig.b.log == want
