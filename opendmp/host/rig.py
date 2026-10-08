"""A DMP on the bench: whole-program load and run, with no vendor firmware (ISA2.md Finding 39).

    rig = Rig("/dev/ttyUSB0")                  # the part comes from the bridge
    rig.load(prog.build(rig.chip))             # reset, minimal init, zero memory, image, entry
    rig.write("limit", 0.1)                    # a param (or a state), by name, before or while running
    packets, ints = rig.run(seconds=5, watch=["angle"], changes=[(2.0, "limit", 0.3)])
    rig.read("angle"), rig.watched             # a state now; the watch log [(t, {name: value})]

The init is the minimal one measured in probe91: no image, no FIFO-rate patch, no I2C-master
setup.  A device reset keeps DMP memory, so the loader clears it.
"""
import time

from ..ir import nodes as N
from ..q30 import MASK, ONE, s32
from .bridge import Bridge

GYRO_FS = {250: 0, 500: 1, 1000: 2, 2000: 3}
ACCEL_FS = {2: 0, 4: 1, 8: 2, 16: 3}
CLEAR = {"mpu6050": 0x0C00, "icm20948": 0x3000}     # all DMP memory (ICM: up to the FIFO)
FIFO = {"mpu6050": 1024, "icm20948": 4096}


class Rig:
    def __init__(self, port: str):
        self.b = Bridge(port)
        part = self.b.part()
        if part.startswith("mpu"):
            self.chip = "mpu6050"
        elif part.startswith("icm"):
            self.chip = "icm20948"
        else:
            raise IOError(f"{port}: the bridge reports {part!r}")
        self.build = None
        self.rate = None
        self.watched: list = []

    def close(self):
        self.b.close()

    # ------------------------------------------------------------------ registers
    def reg(self, bank, reg, data):
        if self.chip == "icm20948":
            self.b.bank(bank)
        self.b.wr(reg, data)

    def init(self, rate: float, gyro_fs: int, accel_fs: int, fsync: bool = False):
        """Device reset and the minimal register set for a DMP program at `rate`.  fsync: put
        the FSYNC level in bit 0 of the temperature sample (EXT_SYNC_SET = 1)."""
        b = self.b
        if self.chip == "mpu6050":
            b.wr(0x6B, 0x80)                            # PWR_MGMT_1: device reset
            time.sleep(0.1)
            b.wr(0x6B, 0x01)                            # wake; PLL on the X gyro
            time.sleep(0.02)
            b.wr(0x6A, 0x00)                            # USER_CTRL
            b.wr(0x19, max(0, min(255, round(1000 / rate) - 1)))   # SMPLRT_DIV: 1 kHz / (1 + div)
            b.wr(0x1A, 0x03 | (0x08 if fsync else 0))  # CONFIG: DLPF 42 Hz; EXT_SYNC_SET
            b.wr(0x1B, GYRO_FS[gyro_fs] << 3)
            b.wr(0x1C, ACCEL_FS[accel_fs] << 3)
            b.wr(0x38, 0x02)                            # INT_ENABLE: DMP_INT
            self.rate = 1000 / (1 + max(0, min(255, round(1000 / rate) - 1)))
        else:
            b._bank = None
            self.reg(0, 0x06, 0x80)                     # PWR_MGMT_1: device reset
            time.sleep(0.1)
            b._bank = None
            self.reg(0, 0x06, 0x01)                     # wake, auto clock
            time.sleep(0.02)
            self.reg(0, 0x07, 0x00)                     # PWR_MGMT_2: accel and gyro on
            self.reg(0, 0x05, 0x00)                     # LP_CONFIG: no duty cycling
            self.reg(0, 0x03, 0x00)                     # USER_CTRL
            div = max(0, min(255, round(1100 / rate) - 1))
            self.reg(2, 0x00, div)                      # GYRO_SMPLRT_DIV: 1.1 kHz / (1 + div)
            self.reg(2, 0x01, (GYRO_FS[gyro_fs] << 1) | 1)
            self.reg(2, 0x10, bytes([div >> 8, div & 0xFF]))   # ACCEL_SMPLRT_DIV
            self.reg(2, 0x14, (ACCEL_FS[accel_fs] << 1) | 1)
            self.reg(2, 0x52, 0x01 if fsync else 0x00)  # FSYNC_CONFIG: EXT_SYNC_SET [probe83]
            self.reg(0, 0x0F, 0x00)                     # INT_PIN_CFG
            self.reg(0, 0x10, 0x02)                     # INT_ENABLE: DMP_INT1
            self.reg(0, 0x11, 0x00)
            self.reg(0, 0x12, 0x00)
            self.reg(0, 0x66, 0x00)                     # FIFO_EN_1/2: the DMP owns the FIFO
            self.reg(0, 0x67, 0x00)
            self.rate = 1100 / (1 + div)

    def load(self, build, rate: float | None = None, verify: bool = True):
        """Reset the part, clear DMP memory, write the program's image, set the entry."""
        if build.chip != self.chip:
            raise ValueError(f"a {build.chip} build on a {self.chip}")
        o = build.ir.opts
        self.init(rate or o["rate"], o["gyro_fs"], o["accel_fs"], fsync=build.fsync)
        end = CLEAR[self.chip]
        img = bytes(build.image[:end])
        self.b.mem_write(0, img)
        if verify:
            back = self.b.mem_read(0, end)
            if back != img:
                bad = next(i for i in range(end) if back[i] != img[i])
                raise IOError(f"load verify failed at {bad:#06x}")
        e = build.entry
        if self.chip == "mpu6050":
            self.b.wr(0x70, bytes([e >> 8, e & 0xFF]))  # DMP_CFG_1/2
        else:
            self.reg(2, 0x50, bytes([e >> 8, e & 0xFF]))   # PRGM_START_ADDRH/L
        self.build = build

    # ------------------------------------------------------------------ state and params
    def _word(self, name):
        if self.build is None:
            raise ValueError("load a build first")
        w = next((w for w in self.build.words if w.name == name and w.kind in ("state", "param")), None)
        if w is None:
            names = ", ".join(w.name for w in self.build.words if w.kind in ("state", "param"))
            raise KeyError(f"{name!r} is not a state or param of {self.build.ir.name} ({names})")
        return w

    def read(self, name: str):
        """A state's or param's current value, read from DMP memory (while running, too)."""
        w = self._word(name)
        raw = s32(int.from_bytes(self.b.mem_read(w.addr, 4), "big"))
        return raw / ONE if (w.var is None or w.var.type == N.Q30) else raw

    def write(self, name: str, value):
        """Set a param (or a state) by name.  While the program runs, the word lands between two
        passes, or -- rarely, for one pass -- is read half-written: the bridge writes its four
        bytes in one I2C burst, and a pass may read it mid-burst."""
        w, raw = self._encode(name, value)
        self.b.mem_write(w.addr, (raw & MASK).to_bytes(4, "big"))

    def _encode(self, name, value):
        w = self._word(name)
        rng = w.var.range if w.var is not None else None
        if rng is not None and not rng[0] <= value <= rng[1]:
            # the program was range-checked assuming this interval (dmp.param(lo, hi))
            raise ValueError(f"{name} = {value}: outside its declared range [{rng[0]:g}, {rng[1]:g}]")
        if w.var is None or w.var.type == N.Q30:
            if not -2.0 <= value < 2.0:
                raise ValueError(f"{name} = {value}: a q30 value is in [-2, 2)")
            raw = round(value * ONE)
        else:
            raw = int(value)
        return w, raw

    # ------------------------------------------------------------------ running
    def start(self):
        self.b.zero_ints()
        if self.chip == "mpu6050":
            self.b.wr(0x6A, 0x4C)                       # FIFO_EN | DMP_RESET | FIFO_RESET
            self.b.wr(0x6A, 0xC0)                       # DMP_EN | FIFO_EN
        else:
            self.reg(0, 0x68, 0x1F)                     # FIFO_RST
            self.reg(0, 0x68, 0x1E)
            self.reg(0, 0x03, 0x08)                     # DMP_RST
            self.reg(0, 0x03, 0xC0)                     # DMP_EN | FIFO_EN

    def stop(self):
        """Halt the DMP.  DMP_RST stops a pass that is running (clearing DMP_EN alone lets it run to
        its end, or forever: InvensenseDMP Finding 47); FIFO_EN stays set so the FIFO stays readable."""
        if self.chip == "mpu6050":
            self.b.wr(0x6A, 0x48)                       # FIFO_EN | DMP_RESET
            self.b.wr(0x6A, 0x40)
        else:
            self.reg(0, 0x03, 0x48)                     # FIFO_EN | DMP_RST
            self.reg(0, 0x03, 0x40)

    def run(self, seconds: float, watch=(), period: float = 0.25, changes=()):
        """Run the loaded program, reading the FIFO as it fills; (packets, interrupt count).
        A partial packet at the end (the stop can land mid-pass) is dropped.

        watch:   state or param names read every `period` seconds into self.watched, as
                 [(seconds since the start, {name: value})]
        changes: [(at seconds, name, value)]: param writes made while the program runs, or
                 [(at seconds, fn)]: fn(rig) called then (e.g. drive the bridge's FSYNC line),
                 in this loop, so it never talks to the bridge at the same time as the reads"""
        size = self.build.packet_size
        cap = FIFO[self.chip]
        data = bytearray()
        todo = sorted(changes, key=lambda c: c[0])
        for name in watch:
            self._word(name)
        for c in todo:
            if len(c) == 3:
                self._encode(c[1], c[2])            # refuse a bad change before starting
        self.watched = []
        next_watch = 0.0
        self.start()
        t0 = time.time()
        try:
            while time.time() - t0 < seconds:
                t = time.time() - t0
                while todo and todo[0][0] <= t:
                    c = todo.pop(0)
                    if len(c) == 3:
                        self.write(c[1], c[2])
                    else:
                        c[1](self)
                if watch and t >= next_watch:
                    self.watched.append((round(t, 3), {n: self.read(n) for n in watch}))
                    next_watch = t + period
                n = self.b.fifo_count()
                if n >= cap - size:
                    raise IOError(f"the FIFO filled ({n} B) faster than the bridge reads it: "
                                  "run at a lower rate")
                while n > 0:
                    chunk = self.b.fifo_read(min(n, 128))
                    data += chunk
                    n -= len(chunk)
        finally:
            self.stop()
        ints = self.b.ints()
        n = self.b.fifo_count()
        while n > 0:
            chunk = self.b.fifo_read(min(n, 128))
            data += chunk
            n -= len(chunk)
        full = len(data) // size
        return [bytes(data[i * size:(i + 1) * size]) for i in range(full)], ints
