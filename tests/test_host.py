"""The host runtime's state and param access, against a stand-in for the bench bridge."""
import pathlib
import runpy

import pytest

from opendmp.emu import Machine, Sample
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


class EmuBridge:
    """A bridge whose part is the emulator: DMP memory, the entry registers, DMP_EN, the FIFO, and
    T (one pass).  While the DMP is enabled, each FIFO_COUNT read runs one more pass."""
    def __init__(self, chip, live=Sample()):
        self.chip, self.live = chip, live
        self.m = Machine(chip)
        self.fifo, self.n_ints, self.running, self._bank, self.closed = bytearray(), 0, False, 0, False

    def part(self):
        return self.chip

    def bank(self, n):
        self._bank = n

    def pause(self, seconds):
        pass

    def wr(self, reg, data):
        v = bytes([data]) if isinstance(data, int) else bytes(data)
        icm = self.chip == "icm20948"
        if (reg, self._bank) == ((0x50, 2) if icm else (0x70, 0)):
            self.m.entry = int.from_bytes(v, "big")
        elif (reg, self._bank) == ((0x03, 0) if icm else (0x6A, 0)):
            self.running = bool(v[0] & 0x80)
            if not icm and v[0] & 0x04:
                self.fifo.clear()
        elif icm and (reg, self._bank) == (0x68, 0) and v[0] & 1:
            self.fifo.clear()

    def mem_write(self, addr, data):
        self.m.mem[addr:addr + len(data)] = data

    def mem_read(self, addr, n):
        return bytes(self.m.mem[addr:addr + n])

    def _pass(self):
        r = self.m.run_pass(self.live)
        self.fifo += r.fifo
        self.n_ints += r.ints
        return r

    def cmd(self, line):
        assert line.startswith("T ")
        if self.chip == "mpu6050":
            self.fifo.clear()                      # T's USER_CTRL 0x4C resets the MPU's FIFO
        n0 = self.n_ints
        r = self._pass()
        return f"{self.n_ints - n0} {round(r.duration_ns / 1000)}"

    def fifo_count(self):
        if self.running:
            self._pass()
        return len(self.fifo)

    def fifo_read(self, n):
        out = bytes(self.fifo[:n])
        del self.fifo[:n]
        return out

    def zero_ints(self):
        self.n_ints = 0

    def ints(self):
        return self.n_ints

    def close(self):
        self.closed = True


COUNT_ASM = """\
        fmode   1
        always
        store
        p2bank  2
        p2page  0
        inc                     ; passes, counted at 0x200
        fifo
        p0bank  1
        p0page  0
        mov     0               ; emit the word at 0x100
        int
        end
"""


def _cli_on(monkeypatch, bridge, argv, capsys):
    from opendmp import cli
    import opendmp.host.rig as rig_mod
    monkeypatch.setattr(rig_mod, "Bridge", lambda port: bridge)
    status = cli.main(argv)
    cap = capsys.readouterr()
    return status, cap.out + cap.err


@pytest.mark.parametrize("chip", ["mpu6050", "icm20948"])
@pytest.mark.parametrize("passes", [1, 3])
def test_run_asm_on_a_rig(tmp_path, monkeypatch, capsys, chip, passes):
    """`opendmp run prog.asm`: no .chip (the bridge's part), loaded alone, run, dumped, verified."""
    f = tmp_path / "count.asm"
    f.write_text(COUNT_ASM)
    br = EmuBridge(chip)
    status, out = _cli_on(monkeypatch, br, ["run", str(f), "--port", "x", "--passes", str(passes),
                                            "--poke", "0x100=0x0f0f0f0f", "--dump", "0x200:4",
                                            "--verify", "-o", str(tmp_path / "out.fifo")], capsys)
    assert status == 0, out
    assert f"count.asm on the {chip}: {passes} interrupt" in out
    assert f"0x0200  {passes:08x}" in out and "BIT-EXACT" in out
    assert (tmp_path / "out.fifo").read_bytes() == bytes.fromhex("0f0f0f0f") * passes
    assert br.m.entry == (0x1000 if chip == "icm20948" else 0x400) and br.closed


def test_run_asm_verify_needs_the_live_sample(tmp_path, monkeypatch, capsys):
    """The emulator gives every pass --sample (zeros by default); silicon reads live data."""
    f = tmp_path / "s.asm"
    f.write_text(".chip mpu6050\nfmode 1\nalways\nfifo\np0bank 0\np0page 0\nmov 0\nint\nend\n")
    live = Sample(ax=5, ay=-1)
    status, out = _cli_on(monkeypatch, EmuBridge("mpu6050", live), ["run", str(f), "--port", "x", "--verify"], capsys)
    assert status == 1 and "DIFFERS" in out and "silicon 0005ffff, emulator 00000000" in out
    status, out = _cli_on(monkeypatch, EmuBridge("mpu6050", live),
                          ["run", str(f), "--port", "x", "--verify", "--sample", "ax=5,ay=-1"], capsys)
    assert status == 0 and "BIT-EXACT" in out


def test_run_asm_refuses_the_other_part(tmp_path, monkeypatch, capsys):
    f = tmp_path / "i.asm"
    f.write_text(".chip icm20948\nint\nend\n")
    status, out = _cli_on(monkeypatch, EmuBridge("mpu6050"), ["run", str(f), "--port", "x"], capsys)
    assert status == 1 and "is for the icm20948" in out


def test_asm_options_need_a_asm(monkeypatch, capsys):
    from opendmp import cli
    with pytest.raises(SystemExit, match="--passes, --dump: for a .asm program only"):
        cli.main(["run", str(EX / "tilt.py"), "--port", "x", "--passes", "2", "--dump", "0x100:4"])


@pytest.mark.parametrize("chip", ["mpu6050", "icm20948"])
def test_load_writes_the_image_and_entry(chip):
    """load(build) goes through load_image: all of DMP memory below the ICM's FIFO, read back,
    and the entry register."""
    ns = runpy.run_path(str(EX / "tilt.py"))
    b = [v for v in ns.values() if type(v).__name__ == "Program"][0].build(chip)
    rig = Rig.__new__(Rig)
    rig.b, rig.chip, rig.watched = EmuBridge(chip), chip, []
    rig.load(b)
    end = 0x3000 if chip == "icm20948" else 0xC00
    assert bytes(rig.b.m.mem[:end]) == bytes(b.image[:end]) and rig.b.m.entry == b.entry
    assert rig.build is b
