"""Everything the emulator reads from or writes to disk.

Sensor data (CSV)
    One row per sample tick.  A header names the columns; any of
        ax ay az gx gy gz temp fsync      (also accel_x.., gyro_x.., t / time, which is ignored)
    Missing columns read 0.  Delimiter: comma, semicolon, tab or spaces.  '#' starts a comment.
    units="raw": int16 counts as the sensor produces them (the default).
    units="si":  accel in g, gyro in deg/s, temp in deg C, converted with the full-scale ranges.

Programs and memory
    FILE@ADDR     a binary loaded at ADDR (hex or decimal).  FILE.hex holds hex text instead,
                  and FILE.s / FILE.asm is assembled with InvensenseDMP's tools/dmpasm.py
                  (found through $OPENDMP_DMPASM or the sibling checkout).
    ADDR=W,W,...  words poked at ADDR: hex ('40000000') or Q30 values with a q ('0.98q').
"""
from __future__ import annotations

import csv
import importlib.util
import io
import os
import pathlib
import re

from ..q30 import parse_word
from .machine import Sample

ALIASES = {
    "ax": "ax", "ay": "ay", "az": "az", "gx": "gx", "gy": "gy", "gz": "gz",
    "accel_x": "ax", "accel_y": "ay", "accel_z": "az",
    "gyro_x": "gx", "gyro_y": "gy", "gyro_z": "gz",
    "temp": "temp", "temperature": "temp", "fsync": "fsync",
}
IGNORED = {"t", "time", "t_s", "t_us", "index", "n"}
TEMP_SI = {"mpu6050": (340.0, 36.53), "icm20948": (333.87, 21.0)}   # LSB/degC, offset


def _num(text: str) -> float:
    t = text.strip()
    return float(int(t, 0)) if re.fullmatch(r"[-+]?0[xX][0-9a-fA-F]+", t) else float(t)


def read_sensor_csv(path, units: str = "raw", accel_fs: float = 2, gyro_fs: float = 2000,
                    chip: str = "mpu6050") -> list[Sample]:
    text = pathlib.Path(path).read_text()
    lines = [ln.split("#", 1)[0].strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError(f"{path}: no data")
    sample_line = lines[0]
    delim = next((d for d in (",", ";", "\t") if d in sample_line), None)
    rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delim)) if delim else \
        [ln.split() for ln in lines]
    header = [h.strip().lower() for h in rows[0]]
    cols = {}
    for i, h in enumerate(header):
        if h in ALIASES:
            cols[ALIASES[h]] = i
        elif h not in IGNORED:
            raise ValueError(f"{path}: unknown column {h!r} (known: {', '.join(sorted(ALIASES))})")
    if not cols:
        raise ValueError(f"{path}: the first row must be a header naming the columns")
    if units not in ("raw", "si"):
        raise ValueError("units is 'raw' or 'si'")
    a_lsb, g_lsb = 32768 / accel_fs, 32768 / gyro_fs
    t_lsb, t_off = TEMP_SI["icm20948" if chip.startswith("icm") else "mpu6050"]
    out = []
    for n, row in enumerate(rows[1:], 2):
        try:
            v = {k: _num(row[i]) for k, i in cols.items() if i < len(row) and row[i].strip()}
        except ValueError as e:
            raise ValueError(f"{path}:{n}: {e}") from None
        if units == "si":
            for k in ("ax", "ay", "az"):
                if k in v: v[k] *= a_lsb
            for k in ("gx", "gy", "gz"):
                if k in v: v[k] *= g_lsb
            if "temp" in v: v["temp"] = (v["temp"] - t_off) * t_lsb
        out.append(Sample(**{k: int(round(x)) for k, x in v.items()}))
    return out


def write_sensor_csv(path, samples, header=("ax", "ay", "az", "gx", "gy", "gz", "temp")):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in samples:
            w.writerow([getattr(s, h) for h in header])


def parse_addr(text: str) -> int:
    return int(str(text).strip(), 0)


def _dmpasm():
    cands = [os.environ.get("OPENDMP_DMPASM", ""),
             pathlib.Path(__file__).resolve().parents[3] / "InvensenseDMP/tools/dmpasm.py"]
    for c in cands:
        if c and pathlib.Path(c).is_file():
            spec = importlib.util.spec_from_file_location("dmpasm", c)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("assembling needs InvensenseDMP/tools/dmpasm.py; set OPENDMP_DMPASM")


def read_program(path, chip: str = "mpu6050") -> bytes:
    """Bytes to load at an address: binary, `.hex` text, or `.s` assembled by InvensenseDMP's
    dmpasm.  (opendmp's own `.asm` carries its addresses: see read_asm.)"""
    p = pathlib.Path(path)
    if p.suffix.lower() == ".hex":
        return bytes.fromhex(re.sub(r"(#|;).*", "", p.read_text()).replace("\n", " ").replace(" ", ""))
    if p.suffix.lower() == ".s":
        asm = _dmpasm()
        return asm.assemble(p.read_text(), "icm" if chip.startswith("icm") else "mantis")
    if p.suffix.lower() == ".asm":
        raise ValueError(f"{p.name}: an opendmp .asm places itself (.org/.entry); load it without @ADDR")
    return p.read_bytes()


def read_asm(path, chip: str | None = None):
    """An opendmp `.asm` file, assembled: an opendmp.isa.Program (segments, entry, symbols)."""
    from ..isa import assemble
    return assemble(pathlib.Path(path).read_text(), chip)


def parse_load(spec: str) -> tuple[str, int | None]:
    """'prog.bin@0x400' -> ('prog.bin', 0x400); 'prog.asm' -> ('prog.asm', None)"""
    if "@" not in spec:
        if spec.lower().endswith(".asm"):
            return spec, None
        raise ValueError(f"--load needs FILE@ADDR, got {spec!r}")
    f, a = spec.rsplit("@", 1)
    return f, parse_addr(a)


def parse_poke(spec: str) -> tuple[int, list[int]]:
    """'0x100=40000000,0.98q' -> (0x100, [0x40000000, 0x3eb851ec])"""
    if "=" not in spec:
        raise ValueError(f"--poke needs ADDR=WORD[,WORD...], got {spec!r}")
    a, ws = spec.split("=", 1)
    return parse_addr(a), [parse_word(w) for w in ws.split(",") if w.strip()]
