"""Every machine behaviour opendmp relies on, with its evidence.

Status:
  HW          measured on silicon (the finding or probe named in `source`)
  VENDOR      stated or implied by InvenSense code or datasheets, not re-measured here
  INFER       deduced from measurements, not measured directly
  UNMEASURED  assumed; the tools that depend on it say so

Sources are InvensenseDMP's `ISA2.md` Findings and `bench/probeNN.py`.  The backend cites
fact ids where it relies on them, and a test checks that every cited id exists.
"""
from dataclasses import dataclass

HW, VENDOR, INFER, UNMEASURED = "HW", "VENDOR", "INFER", "UNMEASURED"
BOTH, MPU, ICM = ("mpu6050", "icm20948"), ("mpu6050",), ("icm20948",)


@dataclass(frozen=True)
class Fact:
    id: str
    chips: tuple
    text: str
    status: str
    source: str


FACTS = [
    # --- boot, memory, the hardware's writes ---------------------------------------------
    Fact("boot.vendor_free", BOTH, "A program runs with no vendor image: device reset, a minimal "
         "register init, all DMP memory zeroed, only our code and data", HW, "Finding 39, probe91 B1"),
    Fact("boot.reset_keeps_ram", BOTH, "A device reset (PWR_MGMT_1 = 0x80) does not clear DMP "
         "memory; the loader must", HW, "Finding 39, probe91"),
    Fact("boot.entry", BOTH, "The program starts at PRGM_START (MPU DMP_CFG_1/2 = 0x70/0x71; ICM "
         "bank 2 0x50/0x51), any address", HW, "Findings 23, 24, 39"),
    Fact("ram.sample.mpu", MPU, "Before every pass the hardware writes ax ay az temp gx gy gz, "
         "int16 BE, at 0x000-0x00d, and nothing else", HW, "Findings 35, 39, probe84/91"),
    Fact("ram.sample.icm", ICM, "Before every pass the hardware clears 0x0000-0x0003 and writes ax ay "
         "az gx gy gz temp, int16 BE, at 0x0004-0x0011", HW, "Findings 34, 39, probe82b/91"),
    Fact("ram.fsync.icm", ICM, "With EXT_SYNC_SET = 1, bit 0 of the temperature sample (RAM 0x0011) "
         "is the FSYNC level", HW, "Finding 35, probe83"),
    Fact("ram.fifo.icm", ICM, "The FIFO is DMP RAM 0x3000-0x3fff: every FIFO byte lands there, "
         "filled from 0x3000 after a FIFO reset; code and data must stay below 0x3000", HW,
         "Finding 39, probe91 FIFO"),
    Fact("ram.fifo.mpu", MPU, "The FIFO is not in the host-visible 3 KB; emits write no DMP RAM",
         HW, "Finding 39, probe91 FIFO"),
    Fact("ram.dmp_off", BOTH, "With DMP_EN clear the hardware writes no DMP memory", HW,
         "Finding 39, probe91 B2"),
    Fact("ram.data_reach", BOTH, "A pointer reaches the data space only: MPU 0x000-0xbff (3 KB, "
         "blocks mod 3), ICM 0x0000-0x0fff", HW, "Findings 24, 35"),
    # --- the flag and predicates --------------------------------------------------------
    Fact("flag.sign", BOTH, "One flag F; no Z.  ADD: F = A + B < 0 exactly; RSUB: F = B + (-A) < 0 "
         "exactly with -A wrapped, so the exact sign of B - A unless A = -2.0; MUL: F = A * B < 0",
         HW, "Finding 37, probe86 E-H, probe88, probe99/100"),
    Fact("flag.test", BOTH, "d1 makes F = (last result == 0)", HW, "Finding 37"),
    Fact("flag.logic", BOTH, "f5 op 1 (\"CMP\") stores nothing and tests signs only: F := A < 0 <= B "
         "(so the compiler compares with RSUB); AND/XOR leave F alone", HW,
         "probe98/99 (56 cells, both parts)"),
    Fact("alu.clamp", BOTH, "f5 op 0 (the vendor's \"MIN\") clamps A to [-B, B]: r = A >= 0 ? min(A, B) "
         ": max(A, -B), -B exact, wrapped; F := A was clamped; on a lane, the same on the lane's "
         "values", HW, "Finding 41, probe103 (1920 cells per part)"),
    Fact("mode4.mpu", MPU, "Every f4 (branch, f4 40, the one-byte form; taken, not taken or "
         "skipped) leaves mode 4.  There a byte below 0x40 is a ONE-byte branch to pc + 1 + "
         "sext6(b) (10 cycles taken, 5 skipped, predicable), 0x40-0x7f neither branch nor "
         "interrupt; MOV, CLR, INC and page loads work", HW, "probe95-97, probe104 (Finding 42)"),
    Fact("flag.frozen", BOTH, "F does not change while a predicate is armed (except d1, fc cK)",
         HW, "probe67/68, Finding 35"),
    Fact("pred.sticky", BOTH, "d9/da/db stay in force until the next d8-db; d9 = db = IF F, "
         "da = IF not F", HW, "Findings 27, 37"),
    Fact("pred.mode_bytes", BOTH, "Mode bytes f0-f7 act even under a false predicate; DMP3 f4 "
         "enters mode 4 even when skipped", HW, "probe64, probe70"),
    # --- ALU ------------------------------------------------------------------------------
    Fact("alu.wrap", BOTH, "ADD/RSUB wrap mod 2^32 (Q30 range [-2, 2))", HW, "probe9"),
    Fact("alu.mul_floor", BOTH, "MUL/NMUL floor the Q60 product to Q30", HW, "Finding 37, probe86 B"),
    Fact("alu.lanes", BOTH, "f2/f6: the ALU computes on the high half-words in place (low halves "
         "cleared); f3/f7: on the sign-extended low halves as integers; INC/DEC/CLR honour the "
         "lane", HW, "Finding 37, probe86 A/E/F, probe88"),
    Fact("alu.preadvance", BOTH, "op&4 advances the P2 word index before the store", HW,
         "Finding 17, probe32"),
    Fact("mac.mode", BOTH, "Under f0 an ALU op is acc +/- A*B (op&1 = sign; op 2/3 = op 0/1), "
         "storing the accumulator; a non-f0 op replaces the accumulator; reset every pass",
         HW, "Finding 30, probe74, probe88 (DMP3 f0 = MAC)"),
    Fact("mac.precision", BOTH, "Products are summed exactly (Q60) and floored once", HW,
         "Finding 37, probe86 B"),
    Fact("mac.restart.mpu", MPU, "Any move of the destination restarts the sum: an op&4, a P2 "
         "page load, d0, df, an advancing MOV", HW, "Finding 37, probe86 B"),
    Fact("mac.restart.icm", ICM, "Only an ALU op's own op&4 restarts the sum", HW,
         "Finding 37, probe88"),
    # --- moves, routing, pointers ----------------------------------------------------------
    Fact("route.sticky", BOTH, "dc (RAM) / dd (FIFO) routing is sticky; FIFO is the state after "
         "DMP_RST", HW, "probe16"),
    Fact("mov.width", BOTH, "MOV and MOVX honour the width mode and the routing; MOVX swaps the "
         "half-words", HW, "Finding 29"),
    Fact("ptr.p2_reset", BOTH, "Loading P2's page resets its word index to 0", HW, "probe61/69"),
    Fact("ptr.xchg", MPU, "fc / fd are toggles (A and MOV via P1 / via P2, store to P0's page); "
         "reset every pass", HW, "probe67/68, Finding 37"),
    Fact("ptr.blocks.mpu", MPU, "bc-be add 1 KB mod 3 and are kept across passes: never emit",
         HW, "Finding 35, probe84"),
    Fact("ptr.blocks.icm", ICM, "bc-be add 1 KB mod 4 and reset every pass", HW, "probe47"),
    Fact("pass.carry.mpu", MPU, "Across passes: pointers, P2 index, route, mode, F survive; the "
         "fc/fd exchanges do not", HW, "Finding 37, probe86 C"),
    Fact("pass.carry.icm", ICM, "Mode 4 survives across passes: start every pass with f1, then d8",
         HW, "probe54b"),
    # --- control flow -----------------------------------------------------------------------
    Fact("branch.mpu", MPU, "f4 d, d < 0x40: a branch by d as a signed 6-bit field (0x00-0x1f "
         "forward 0..31, 0x20-0x3f back 32..1) after the instruction [HW, probe101]; every f4 "
         "enters mode 4, even when skipped [HW, probe95-97]; "
         "f4 40 does nothing; f4 41-7f (low five bits set) raise DMP_INT; f4 >= 0x80 is one byte",
         HW, "probe26/27/69/70"),
    Fact("jump.icm", ICM, "f4 hi lo: jump (hi bit 6 clear) or call to (hi << 8 | lo) & 0x3fff; "
         "every f4 enters mode 4, where a low byte is a two-byte target until a width byte",
         HW, "Findings 24-26, probe46/54b/64"),
    Fact("call.depth", ICM, "The return stack is 4 deep and circular; ret with no call stops "
         "the DMP", HW, "Finding 33, probe80"),
    Fact("fc.compare", ICM, "fc cK Y: F := (P1 word K & 0x3f) ==/>/</!= Y >> 2 (cond Y & 3), "
         "even under a predicate; stores nothing", HW, "Finding 35, probe83b"),
    Fact("int.mpu", MPU, "fe raises DMP_INT (honours a predicate)", HW, "probe69/70"),
    Fact("int.icm", ICM, "e2 raises DMP_INT1", HW, "probe53"),
    Fact("int.icm_fe", ICM, "fe on DMP3 also raises the interrupt", INFER, "emulator default"),
    Fact("end.pass", BOTH, "ff ends the pass", HW, "Finding 23"),
    # --- DMP3 extras -------------------------------------------------------------------------
    Fact("shift.icm", ICM, "d4-d7 load the shifter from P0's page; fd n shifts it (left s pulling "
         "the next word's fraction, or arithmetic right) and stores the selected word; every ALU "
         "op reloads it", HW, "Findings 25, 64, probe51/64"),
    Fact("lo.icm", ICM, "d2/d3 store the low 30 bits of the last product", HW, "probe51"),
    Fact("movz.icm", ICM, "bf toggles MOV-stores-zero", HW, "Finding 35, probe83d/e"),
    Fact("io.icm", ICM, "A page load under mode 4 addresses the host register file (register "
         "k = a/4, bank k >> 7); blocks 2-3 reach nothing", HW, "Finding 34, probe82"),
    # --- timing -------------------------------------------------------------------------------
    Fact("time.cycles", BOTH, "Every instruction class costs an integer number of DMP cycles "
         "(MPU 8.199 MHz, ICM 20.975 MHz on this bench); the table is opendmp.emu.chips", HW,
         "Finding 38, probe90"),
    Fact("time.start", BOTH, "Tick to first instruction: 110 cycles (MPU), 130 (ICM)", HW,
         "Finding 38, probe90"),
    Fact("time.stall.mpu", MPU, "A sample write inside a running pass stalls it 14-28 cycles",
         HW, "Finding 38, probe89/90"),
    Fact("sched.mpu", MPU, "A pass starts only on a tick: an overrun drops the rate to an integer "
         "submultiple", HW, "Finding 23, probe41"),
    Fact("sched.icm", ICM, "A pending tick starts the next pass at once: the rate is 1/pass time",
         HW, "probe90"),
    # --- undefined, never emitted ---------------------------------------------------------
    Fact("undef.bf.mpu", MPU, "bf corrupts stores unpredictably", HW, "probe84b"),
    Fact("undef.fc_low.icm", ICM, "fc X Y with X < 0xc0: not decoded", UNMEASURED, "UNKNOWNS.md"),
    Fact("undef.xchg_both", MPU, "fc and fd exchanges in force together", UNMEASURED, "-"),
]

BY_ID = {f.id: f for f in FACTS}
assert len(BY_ID) == len(FACTS), "duplicate fact id"


def fact(fid: str) -> Fact:
    return BY_ID[fid]


def table(chip: str | None = None) -> str:
    """The facts as a plain-text table (for `opendmp facts`)."""
    rows = [f for f in FACTS if chip is None or chip in f.chips]
    w = max(len(f.id) for f in rows)
    return "\n".join(f"{f.id:{w}s}  {f.status:10s}  {'both' if len(f.chips) == 2 else f.chips[0]:9s}  "
                     f"{f.text}  [{f.source}]" for f in rows)
