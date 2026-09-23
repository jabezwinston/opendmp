"""A DMP on the bench: whole-program load and run, with no vendor firmware (ISA2.md Finding 39).

    rig = Rig("/dev/ttyUSB0")                  # the part comes from the bridge
    rig.load(prog.build(rig.chip))             # reset, minimal init, zero memory, image, entry
    packets = rig.run(seconds=5)               # the FIFO, cut into packets

The init is the minimal one measured in probe91: no image, no FIFO-rate patch, no I2C-master
setup.  A device reset keeps DMP memory, so the loader clears it.
"""
import time

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

    def close(self):
        self.b.close()

    # ------------------------------------------------------------------ registers
    def reg(self, bank, reg, data):
        if self.chip == "icm20948":
            self.b.bank(bank)
        self.b.wr(reg, data)

    def init(self, rate: float, gyro_fs: int, accel_fs: int):
        """Device reset and the minimal register set for a DMP program at `rate`."""
        b = self.b
        if self.chip == "mpu6050":
            b.wr(0x6B, 0x80)                            # PWR_MGMT_1: device reset
            time.sleep(0.1)
            b.wr(0x6B, 0x01)                            # wake; PLL on the X gyro
            time.sleep(0.02)
            b.wr(0x6A, 0x00)                            # USER_CTRL
            b.wr(0x19, max(0, min(255, round(1000 / rate) - 1)))   # SMPLRT_DIV: 1 kHz / (1 + div)
            b.wr(0x1A, 0x03)                            # CONFIG: DLPF 42 Hz
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
        self.init(rate or o["rate"], o["gyro_fs"], o["accel_fs"])
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
        if self.chip == "mpu6050":
            self.b.wr(0x6A, 0x40)
        else:
            self.reg(0, 0x03, 0x40)                     # keep FIFO_EN: the FIFO stays readable

    def run(self, seconds: float):
        """Run the loaded program, reading the FIFO as it fills; (packets, interrupt count).
        A partial packet at the end (the stop can land mid-pass) is dropped."""
        size = self.build.packet_size
        cap = FIFO[self.chip]
        data = bytearray()
        self.start()
        t0 = time.time()
        try:
            while time.time() - t0 < seconds:
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
