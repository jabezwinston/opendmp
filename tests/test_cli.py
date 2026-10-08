"""The command line, fed entirely from files."""
import csv

import pytest

from opendmp.emu.__main__ import main
from opendmp.emu.files import read_sensor_csv


def test_run_from_files(tmp_path, capsys):
    # emit accel_x as a 16-bit word each pass, and raise DMP_INT when it is positive
    (tmp_path / "prog.hex").write_text("b0 80 dd f2 c0  f1 b4 90 dc 00 da fe d8 ff  # comment\n")
    (tmp_path / "s.csv").write_text("t,ax,ay,az\n0,100,0,16384\n1,-50,0,16384\n2,7,0,16384\n")
    (tmp_path / "run.toml").write_text(
        'chip = "mpu6050"\nrate = 100\nsensors = "s.csv"\nload = ["prog.hex@0x400"]\n'
        'log = "log.csv"\nfifo = "out.bin"\ntrace = "trace.txt"\ntrace_passes = 2\nwatch = ["0x240"]\n')
    assert main([str(tmp_path / "run.toml"), "--poke", "0x240=0.5q"]) == 0
    rows = list(csv.DictReader(open(tmp_path / "log.csv")))
    assert [r["fifo_hex"] for r in rows] == ["0064", "ffce", "0007"]
    assert (tmp_path / "out.bin").read_bytes() == bytes.fromhex("0064ffce0007")
    assert len((tmp_path / "trace.txt").read_text().splitlines()) > 10
    assert "3 passes over 3 samples" in capsys.readouterr().out


def test_si_units(tmp_path):
    (tmp_path / "s.csv").write_text("ax ay az gx gy gz temp\n1.0 -0.5 0 250 0 -2000 36.53\n")
    s = read_sensor_csv(tmp_path / "s.csv", units="si", accel_fs=2, gyro_fs=2000)[0]
    assert (s.ax, s.ay, s.gx, s.gz, s.temp) == (16384, -8192, 4096, -32768, 0)


ASM = ".chip mpu6050\nfmode 1\nalways\nfifo\np0bank 0\np0page 0\nmov 0\nint\nend\n"   # emit word 0


def test_asm_with_a_fixed_sample(tmp_path, capsys):
    """No sensor file: every pass sees --sample (zeros if omitted)."""
    (tmp_path / "p.asm").write_text(ASM)
    for extra, want in (([], "00000000"), (["--sample", "ax=100,ay=-2"], "0064fffe")):
        assert main(["--load", str(tmp_path / "p.asm"), "--passes", "2", "--fifo", str(tmp_path / "o.bin")]
                    + extra) == 0
        assert (tmp_path / "o.bin").read_bytes() == bytes.fromhex(want) * 2
    assert "2 passes over 2 samples" in capsys.readouterr().out


def test_sample_and_sensors_conflict(tmp_path):
    (tmp_path / "p.asm").write_text(ASM)
    (tmp_path / "s.csv").write_text("ax\n1\n")
    with pytest.raises(SystemExit, match="give one"):
        main(["--load", str(tmp_path / "p.asm"), "--sensors", str(tmp_path / "s.csv"), "--sample", "ax=1"])
    with pytest.raises(SystemExit, match="'bx=1'"):
        main(["--load", str(tmp_path / "p.asm"), "--sample", "bx=1"])


def test_model_values_are_checked(tmp_path):
    (tmp_path / "p.asm").write_text(ASM)
    with pytest.raises(SystemExit, match="mac_restart = 'advance': one of chip, move, op, none"):
        main(["--load", str(tmp_path / "p.asm"), "--model", "mac_restart=advance"])
    with pytest.raises(SystemExit, match="unknown xchgs"):
        main(["--load", str(tmp_path / "p.asm"), "--model", "carry=pointers,xchgs"])
    assert main(["--load", str(tmp_path / "p.asm"), "--model", "carry=none", "--model", "mac_precision=q30"]) == 0


def test_asm_chip_picks_the_chip(tmp_path, capsys):
    """An ICM .asm runs on the ICM without --chip (its .chip used to be overridden by the default)."""
    (tmp_path / "i.asm").write_text(".chip icm\nfmode 1\nalways\nfifo\np0bank 1\np0page 0\nmov 0\nint\nend\n")
    assert main(["--load", str(tmp_path / "i.asm"), "--poke", "0x100=0x01020304i", "--fifo", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o").read_bytes() == bytes.fromhex("01020304")
    assert capsys.readouterr().out.startswith("icm20948: 1 passes")
