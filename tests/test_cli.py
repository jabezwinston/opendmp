"""The command line, fed entirely from files."""
import csv

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
