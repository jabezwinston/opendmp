"""Words -> RAM addresses.

Every ALU op needs its A operand on P0's page, its B operand on P1's page and its destination
at P2's walked word index; a pointer that must move costs a page load (and a bank load when the
bank changes).  Pointer loads were over a third of all code bytes when words were simply packed
in declaration order.  So the layout is chosen for the code:

  1. a first placement packs words by affinity: two words that one pointer touches one after the
     other want the same page (each such pair sharing a page saves a page load);
  2. a local search then moves words between slots and keeps a move when the scheduler's actual
     output (sched.py, the same code the build emits) gets smaller, or equal in size and faster.

The search is seeded and bounded, so a build is reproducible.  On the ICM a word that receives a
MAC run must sit at an odd word of its page: the run's first op pre-advances to restart the
accumulator (ISA2 Finding 37), and from a freshly loaded P2 only indices 0 and 2 are reachable
without an op.
"""
import random
from collections import defaultdict

from ..emu.chips import get_chip
from ..isa.core import cost_key
from ..lang.errors import DmpError
from . import tac as T

DATA = {"mpu6050": (0x010, 0x400),       # page 0 is the sample page; pointers reach 1 KB (no blocks)
        "icm20948": (0x020, 0x400)}      # 0x0000-0x0013 samples; block 0 only in v1

ORDER = {"state": 0, "param": 1, "const": 2, "temp": 3}


def ok_slot(w, addr, icm):
    return not (icm and w.mac_dst) or (addr >> 2) & 1


def layout(words, chip: str):
    """The plain placement: states, params, constants, temps, in order."""
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


# ------------------------------------------------------------------ affinity
def streams(ops):
    """The words each pointer touches, in program order (P0 = A and MOV source, P1 = B, P2 =
    destination)."""
    s = ([], [], [])
    for op in ops:
        if isinstance(op, T.Alu):
            s[0].append(op.a)
            s[1].append(op.b)
            if op.dst is not T.FIFO:
                s[2].append(op.dst)
        elif isinstance(op, T.MacRun):
            s[2].append(op.dst)
            for _, a, b in op.terms:
                s[0].append(a)
                s[1].append(b)
        elif isinstance(op, T.Mov):
            s[0].append(op.src)
            if op.dst is not T.FIFO:
                s[2].append(op.dst)
        elif isinstance(op, (T.Clr, T.IncDec)):
            s[2].append(op.dst)
        elif isinstance(op, T.Cmp):
            s[0].append(op.b)
            s[1].append(op.a)
            s[2].append(op.scratch)
        elif isinstance(op, T.EqTest):
            s[0].append(op.a)
            s[1].append(op.b)
            s[2].append(op.scratch)
    return s


def greedy(words, ops, chip):
    """Pack words into pages by affinity: seed a page with the most-connected unplaced word, then
    add the word most attached to the page, while it has room."""
    lo, hi = DATA[chip]
    icm = chip == "icm20948"
    nbr = defaultdict(lambda: defaultdict(int))
    for seq in streams(ops):
        for x, y in zip(seq, seq[1:]):
            if x is not y:
                nbr[x][y] += 1
                nbr[y][x] += 1
    left = [w for w in words if w.kind != "sample"]
    deg = {w: sum(nbr[w].values()) for w in left}
    left.sort(key=lambda w: (-deg[w], ORDER[w.kind]))
    pages = []
    while left:
        page = [left.pop(0)]
        while len(page) < 4 and left:
            cand = left
            if icm and sum(p.mac_dst for p in page) >= 2:
                cand = [w for w in left if not w.mac_dst]
                if not cand:
                    break
            best = max(cand, key=lambda w: (sum(nbr[w][p] for p in page), deg[w]))
            page.append(best)
            left.remove(best)
        pages.append(page)
    if lo + 16 * len(pages) > hi:
        raise DmpError(f"the program needs more than {(hi - lo) // 4} words of data memory")
    for k, page in enumerate(pages):
        free = list(range(lo + 16 * k, lo + 16 * k + 16, 4))
        for w in sorted(page, key=lambda w: not w.mac_dst) if icm else page:
            s = next(a for a in free if ok_slot(w, a, icm))
            free.remove(s)
            w.addr = s
    return words


# ------------------------------------------------------------------ search
def cost(ops, chip):
    """(bytes, cycles) of the scheduler's output for this placement, branches counted at their
    short size (relaxation hops come later)."""
    from .sched import schedule
    c = get_chip(chip)
    size = cyc = 0
    for e in schedule(ops, chip):
        if e.insn is None:
            continue
        op = e.insn.op
        size += (1 if e.mode == 4 else 2) if op == "branch" else 3 if op == "jump" else \
            2 if op in ("shl", "sar") else 1
        cyc += c.cyc(cost_key(e.insn, e.mode if e.mode is not None else 1))
    return size, cyc


def optimise(words, ops, chip, evals: int | None = None, seed: int = 1):
    """Place `words` for `ops`: the better of the plain and the affinity placement, then a
    bounded local search over moves and swaps, scored by the scheduler's output."""
    lo, hi = DATA[chip]
    icm = chip == "icm20948"
    mine = [w for w in words if w.kind != "sample"]
    if not mine:
        return words
    layout(mine, chip)
    plain = {w: w.addr for w in mine}
    c_plain = cost(ops, chip)
    for w in mine:
        w.addr = None
    greedy(mine, ops, chip)
    best = cost(ops, chip)
    if c_plain < best:
        for w in mine:
            w.addr = plain[w]
        best = c_plain
    rnd = random.Random(seed)
    evals = evals if evals is not None else min(1200, 25 * len(mine))
    top = max(w.addr for w in mine) + 16
    slots = list(range(lo, min(hi, (top | 15) + 1), 4))
    counts = defaultdict(int)
    for seq in streams(ops):
        for w in seq:
            counts[w] += 1
    hot = [w for w in mine for _ in range(1 + counts[w])]
    at = {w.addr: w for w in mine}
    for _ in range(evals):
        w = rnd.choice(hot)
        dest = rnd.choice(slots)
        src = w.addr
        if dest == src:
            continue
        other = at.get(dest)
        if not ok_slot(w, dest, icm) or (other is not None and not ok_slot(other, src, icm)):
            continue
        w.addr = dest
        if other is not None:
            other.addr = src
        c = cost(ops, chip)
        if c <= best:
            best = c
            at[dest] = w
            if other is not None:
                at[src] = other
            else:
                del at[src]
        else:
            w.addr = src
            if other is not None:
                other.addr = dest
    return words
