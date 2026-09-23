"""The serial protocol of the ESP32 bench bridge (InvensenseDMP/bench/esp32_dmp_bridge).

One command per line, one `OK ...` or `ERR ...` answer:

    K                 the part: "mpu addr 68" / "icm addr 68"
    G reg n           read n registers          S reg hex        write registers
    M bank off n      read DMP memory (<= 128)  N bank off hex   write DMP memory
    F                 FIFO count                R n              read n FIFO bytes (<= 128)
    I                 INT level and edge count  Z                zero the edge count
    T ms / W ms       run until the first / second INT edge (timing)

On the ICM-20948 the bridge selects register bank 0 itself for M, N, F and R; `bank()` tracks
the bank for G and S.
"""
import time

try:
    import serial
except ImportError:                             # pragma: no cover
    serial = None


class BridgeError(IOError):
    pass


class Bridge:
    def __init__(self, port: str, baud: int = 115200):
        if serial is None:
            raise BridgeError("pyserial is needed for the bench: pip install opendmp[bench]")
        self.port = port
        self.s = serial.serial_for_url(port, baudrate=baud, timeout=1)
        self._bank = None
        self._sync()

    def close(self):
        self.s.close()

    def _sync(self, tries=3):
        """Reset the ESP32 (its ROM prints binary with no newlines) and wait for the banner."""
        for _ in range(tries):
            self.s.setDTR(False)
            self.s.setRTS(True)
            time.sleep(0.15)
            self.s.setRTS(False)
            time.sleep(0.05)
            self.s.setDTR(True)
            t0, buf = time.time(), b""
            while time.time() - t0 < 4.0:
                buf += self.s.read(512)
                if b"esp32_dmp_bridge ready" in buf:
                    time.sleep(0.1)
                    self.s.reset_input_buffer()
                    return
            self.s.reset_input_buffer()
            self.s.write(b"P\n")
            time.sleep(0.4)
            if b"OK" in self.s.read(64):
                self.s.reset_input_buffer()
                return
        raise BridgeError(f"no bridge on {self.port}: is esp32_dmp_bridge flashed?")

    def cmd(self, line: str, retries: int = 2) -> str:
        for attempt in range(retries + 1):
            self.s.reset_input_buffer()
            self.s.write((line + "\n").encode())
            t0 = time.time()
            while time.time() - t0 < 2.5:
                r = self.s.readline().decode(errors="replace").strip()
                if not r or r.startswith("#"):
                    continue
                if r.startswith("ERR"):
                    raise BridgeError(f"{r} (after: {line[:40]})")
                if r.startswith("OK"):
                    return r[3:].strip()
        raise BridgeError(f"timeout on: {line[:40]}")

    # ------------------------------------------------------------------ registers
    def bank(self, n: int):
        """ICM-20948 register bank (REG_BANK_SEL = 0x7f)."""
        if self._bank != n:
            self.cmd(f"S 7f {n << 4:02x}")
            self._bank = n

    def rd(self, reg: int, n: int = 1) -> bytes:
        return bytes.fromhex(self.cmd(f"G {reg:02x} {n}"))

    def wr(self, reg: int, data):
        if isinstance(data, int):
            data = bytes([data])
        self.cmd(f"S {reg:02x} {bytes(data).hex()}")

    # ------------------------------------------------------------------ DMP memory, FIFO
    def mem_read(self, addr: int, n: int) -> bytes:
        out = bytearray()
        while n:
            bank, off = divmod(addr, 256)
            k = min(n, 256 - off, 128)
            out += bytes.fromhex(self.cmd(f"M {bank} {off:02x} {k}"))
            addr, n = addr + k, n - k
        self._bank = 0 if self._bank is not None else None
        return bytes(out)

    def mem_write(self, addr: int, data: bytes):
        i = 0
        while i < len(data):
            bank, off = divmod(addr + i, 256)
            k = min(len(data) - i, 256 - off, 64)
            self.cmd(f"N {bank} {off:02x} {bytes(data[i:i + k]).hex()}")
            i += k
        self._bank = 0 if self._bank is not None else None

    def fifo_count(self) -> int:
        r = int(self.cmd("F"))
        self._bank = 0 if self._bank is not None else None
        return r

    def fifo_read(self, n: int) -> bytes:
        r = bytes.fromhex(self.cmd(f"R {min(n, 128)}"))
        self._bank = 0 if self._bank is not None else None
        return r

    def ints(self) -> int:
        return int(self.cmd("I").split()[1])

    def zero_ints(self):
        self.cmd("Z")

    def part(self) -> str:
        return self.cmd("K")
