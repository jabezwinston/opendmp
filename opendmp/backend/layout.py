"""Words -> RAM addresses.

v1 packs words into pages in order: states, params, constants, temps.  On the ICM a word that
receives a MAC run goes at an odd word of its page: the run's first op must pre-advance to
restart the accumulator (ISA2 Finding 37), and from a freshly loaded P2 only indices 0 and 2
are reachable without an op, so the destination is index 1 or 3.
"""
from ..lang.errors import DmpError

DATA = {"mpu6050": (0x010, 0x400),       # page 0 is the sample page; pointers reach 1 KB (no blocks)
        "icm20948": (0x020, 0x400)}      # 0x0000-0x0013 samples; block 0 only in v1

ORDER = {"state": 0, "param": 1, "const": 2, "temp": 3}


def layout(words, chip: str):
    lo, hi = DATA[chip]
    free = list(range(lo, hi, 4))
    icm = chip == "icm20948"
    placed = [w for w in words if w.kind != "sample" and w.addr is None]
    placed.sort(key=lambda w: ORDER[w.kind])
    for w in placed:
        if icm and w.mac_dst:
            slot = next((a for a in free if (a >> 2) & 1), None)
        elif icm:                               # leave the odd words for MAC destinations
            slot = next((a for a in free if not (a >> 2) & 1), free[0] if free else None)
        else:
            slot = free[0] if free else None
        if slot is None:
            raise DmpError(f"the program needs more than {(hi - lo) // 4} words of data memory")
        free.remove(slot)
        w.addr = slot
    return words
