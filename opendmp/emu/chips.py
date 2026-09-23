"""Chip profiles: memory geometry, sample layout, pass scheduling and the cycle model.

Timing is counted in DMP clock cycles.  Every instruction class has an integer cycle count,
measured on the bench rigs by timing single passes between two interrupts
(InvensenseDMP/bench/probe90.py, the bridge's `W` command): every measured cost on both parts
came out an integer multiple of one period, to within 0.1 %.

    MPU-6050   121.96 ns (8.199 MHz)     ICM-20948   47.675 ns (20.975 MHz)

The clock is this bench's parts, relative to the ESP32's crystal; another part will differ by
its oscillator's tolerance (Machine(..., clock_hz=...) overrides it).  A class with no
measurement falls back along FALLBACK to one that has, and `cost_report()` says which is which.
"""
from dataclasses import dataclass, field

# Every instruction is charged under one of these keys (".fifo" is appended when the result went
# to the FIFO; on the MPU that costs the same as RAM [HW, probe90]).  A skipped instruction costs
# "skip" per BYTE [HW both].  Unmeasured keys fall back along this chain.
FALLBACK = {
    "page": "cheap", "skip": "cheap", "int": "cheap", "halt": "cheap", "int2": "cheap",
    "call": "branch", "ret": "cheap", "branch": "cheap", "branch1": "branch",
    "movx": "mov", "clr": "mov", "sload": "mov", "lo": "clr",
    "add16": "add", "mul16": "mul", "mac": "mul", "clamp": "add", "logic": "add", "cmp": "add",
    "inc": "add", "shift": "add", "fccmp": "add",
}


@dataclass(frozen=True)
class Chip:
    name: str
    icm: bool                  # DMP3 (ICM-20948) semantics
    ram_size: int              # bytes the emulator models (host-addressable DMP memory)
    data_limit: int            # bytes a pointer can reach
    block_mod: int             # bc/bd/be wrap: 3 on the MPU, 4 on DMP3
    fifo_size: int
    fifo_base: int | None      # where the FIFO's bytes live in DMP memory, if the host can see it
    code_limit: int            # code must end below this (the FIFO, or the end of memory)
    entry: int                 # default PRGM_START
    rate_policy: str           # "tick": a pass starts only on a sample tick (MPU)
                               # "free": a pending tick starts the next pass at once (DMP3)
    clock_hz: float
    cycles: dict = field(default_factory=dict)       # key -> cycles
    source: dict = field(default_factory=dict)       # key -> where the number comes from
    start_cycles: int = 0      # sample tick -> first instruction
    tick_stall_cycles: int = 0 # a sample write landing inside a running pass

    def cyc(self, key: str) -> int:
        k = key
        while k not in self.cycles:
            if k.endswith(".fifo"):
                k = k[:-5]
            elif k in FALLBACK:
                k = FALLBACK[k]
            else:
                raise KeyError(f"no cycle count for {key!r} on {self.name}")
        return self.cycles[k]

    @property
    def cycle_ns(self) -> float:
        return 1e9 / self.clock_hz

    def cost(self, key: str) -> float:
        """ns for one instruction of this class (one byte, for "skip")."""
        return self.cyc(key) * self.cycle_ns

    def cost_report(self) -> list[tuple[str, int, float, str]]:
        out = []
        for k in sorted(set(self.cycles) | set(FALLBACK)):
            if k in self.cycles:
                out.append((k, self.cycles[k], self.cost(k), self.source.get(k, "")))
            else:
                out.append((k, self.cyc(k), self.cost(k), f"unmeasured: charged as {FALLBACK[k]}"))
        return out


# MPU-6050: probe90 at a 100 Hz sample rate (no sample write inside the pass), 1000 fillers,
# spread 0-2 us.  us = cycles x 0.12196.
_MPU = {
    "cheap": 5,     # 0.610: d8 b0 f1 dc d0 d1, fc; f4 with bit 7 set
    "page": 5,      # 0.610: 87
    "skip": 5,      # 0.610 per byte (40 and c0 under a false d9; f4 00 skipped = 10)
    "branch": 15,   # 1.830: f4 00 and f4 01, taken; the bytes jumped over cost nothing
    "branch1": 10,  # 1.220: a one-byte branch in mode 4, taken (probe104; skipped: 5)
    "clr": 12,      # 1.464: de, df
    "cmp": 17,      # 2.073: f5 01
    "mov": 18, "movx": 18,   # 2.195: c0 c1 c8, to RAM or the FIFO
    "add16": 22,    # 2.683: 40 under f2
    "logic": 23,    # 2.805: f5 02
    "inc": 23,      # 2.805: f8
    "add": 28,      # 3.415: 40 41 44 (RAM or FIFO)
    "clamp": 28,    # 3.415: f5 00
    "mul16": 52,    # 6.342: 42 under f3
    "mul": 58,      # 7.073: 42 43
    "mac": 60,      # 7.317: 00 under f0
}
# ICM-20948: probe90 (no stall at any rate), 3000 fillers, spread 0-2 us.  us = cycles x 0.047675.
_ICM = {
    "cheap": 4,     # 0.1907: d8 8f f1
    "page": 4,
    "skip": 4,      # per byte: 40 skipped; a skipped f4 hi lo = 12
    "branch": 13,   # jump f4 hi lo, taken (jump + f1 = 17)
    "call": 13,     # call + ret = 18 (call S; f1 with S = d8 f1 e0: 30); the split is assumed
    "ret": 5,
    "clr": 11,      # 0.525: de
    "lo": 11,       # 0.525: d2
    "sload": 15,    # 0.716: d4
    "shift": 15,    # 0.716: fd 3f
    "mov": 17, "movx": 17,   # 0.811
    "fccmp": 19,    # 0.907: fc c0 00
    "add16": 20,    # 0.954: 40 under f3
    "inc": 21,      # 1.002: f8
    "mul": 25,      # 1.193: 42
    "add": 26,      # 1.241: 40 41
    "clamp": 26,    # 1.241: f5 00
    "mac": 27,      # 1.288: f0 00
}

MPU6050 = Chip(
    name="mpu6050", icm=False,
    ram_size=0xC00, data_limit=0xC00, block_mod=3, fifo_size=1024, entry=0x0400,
    fifo_base=None,            # not in the 3 KB the host reaches; 0xc00-0xfff reads 0 (probe91)
    code_limit=0xC00,
    rate_policy="tick", clock_hz=8.199e6,
    cycles=_MPU, source={k: "probe90" for k in _MPU},
    start_cycles=110,          # 13.4 us: data-ready edge to the program's first byte (probe90)
    tick_stall_cycles=20,      # 14-28 by mix: d8 3.4 us, MOV 2.5, ADD 1.75 per tick (probe90)
)

ICM20948 = Chip(
    name="icm20948", icm=True,
    ram_size=0x4000, data_limit=0x1000, block_mod=4, fifo_size=4096, entry=0x1000,
    fifo_base=0x3000,          # the FIFO is DMP RAM 0x3000-0x3fff, filled from 0x3000 (probe91)
    code_limit=0x3000,
    rate_policy="free", clock_hz=20.975e6,
    cycles=_ICM, source={k: "probe90" for k in _ICM},
    start_cycles=130,          # ~6.2 us (probe90)
    tick_stall_cycles=0,       # the same body at 9 kHz, 1 kHz and 100 Hz: 1145/1146/1146 us
)

CHIPS = {"mpu6050": MPU6050, "mpu": MPU6050, "icm20948": ICM20948, "icm": ICM20948}


def get_chip(name, clock_hz: float | None = None) -> Chip:
    if isinstance(name, Chip):
        chip = name
    else:
        try:
            chip = CHIPS[name.lower()]
        except KeyError:
            raise ValueError(f"unknown chip {name!r} (known: mpu6050, icm20948)") from None
    if clock_hz:
        from dataclasses import replace
        chip = replace(chip, clock_hz=clock_hz)
    return chip
