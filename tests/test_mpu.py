"""MPU-6050 semantics against recorded hardware results (InvensenseDMP bench probes)."""
import pytest

from opendmp.emu import Machine, Model, Sample, Unmodelled
from conftest import run1, words

CONST = [0x20000000, 0x00000000, 0x40000000, 0x60000000]      # {0.5, 0.0, 1.0, 1.5} at 0x70


def const_page(m):
    m.poke_words(0x70, CONST)


def alu(op, A, B):
    return (A << 5) | (B << 3) | {"add": 0, "rsub": 1, "mul": 2, "nmul": 3}[op]


@pytest.mark.parametrize("op,A,B,want", [
    ("add", 2, 0, 0x60000000), ("rsub", 2, 0, 0xE0000000), ("mul", 3, 0, 0x30000000),
    ("nmul", 3, 0, 0xD0000000), ("add", 2, 3, 0xA0000000), ("mul", 2, 3, 0x60000000),
    ("rsub", 0, 2, 0x20000000), ("add", 3, 3, 0xC0000000),
])
def test_probe9_alu_wraps(op, A, B, want):
    # P0 = P1 = constant page, P2 = 0x240, RAM routing: `97 a4 ba dc <v>` (probe9, 8/8 on silicon)
    m, _ = run1([0xB0, 0x87, 0xB4, 0x97, 0xBA, 0xA4, 0xDC, alu(op, A, B)], setup=const_page)
    assert m.peek_word(0x240) & 0xFFFFFFFF == want


PTR74 = [0xF1, 0xB0, 0x87, 0xB4, 0x96, 0xBA, 0xA4, 0xDD]      # probe74: A at 0x70, B at 0x60, FIFO


def cell74(body, A=0x20000000, B=(0x20000000,) * 4, passes=1):
    def setup(m):
        m.poke_words(0x70, [A] * 4)
        m.poke_words(0x60, list(B))
    m, r = run1(PTR74 + body, setup=setup, passes=passes)
    return [words(x.fifo) for x in r] if passes > 1 else words(r.fifo)


def test_probe74_mac():
    assert cell74([0x00]) == [0x40000000]
    assert cell74([0x02]) == [0x10000000]
    assert cell74([0xF0, 0x00]) == [0x10000000]
    assert cell74([0xF0, 0x00, 0x00, 0x00, 0x00]) == [0x10000000, 0x20000000, 0x30000000, 0x40000000]
    assert cell74([0x00, 0x00, 0x00]) == [0x40000000] * 3
    assert cell74([0xF0, 0x00, 0xF1, 0x00, 0xF0, 0x00]) == [0x10000000, 0x40000000, 0x50000000]
    # the dot product of A = 0.5 with B = [1, 1/2, 1/4, 1/8]
    assert cell74([0xF0, 0x00, 0x08, 0x10, 0x18], B=(0x40000000, 0x20000000, 0x10000000, 0x08000000)) \
        == [0x20000000, 0x30000000, 0x38000000, 0x3C000000]
    # R carries across passes, but each pass's P2 page load zeroes it, so every pass sums afresh
    assert cell74([0xF0, 0x00, 0x00], passes=3) == [[0x10000000, 0x20000000]] * 3


def test_mac_tiny_products_floor():
    # 7 * -5 * 2^-30 truncates toward -inf to -1 (Finding 30)
    assert cell74([0xF0, 0x00], A=7, B=(-5 & 0xFFFFFFFF,) * 4) == [0xFFFFFFFF]


def f5cell(body, seed=0x11110000):
    def setup(m):
        m.poke_words(0x70, [5])
        m.poke_words(0x60, [-5 & 0xFFFFFFFF])
        m.poke_words(0x240, [seed])
    m, r = run1([0xF1, 0xB0, 0x87, 0xB4, 0x96, 0xBA, 0xA4, 0xDC] + body, setup=setup)
    return m.peek_word(0x240) & 0xFFFFFFFF, r


def test_probe70_logic_family():
    assert f5cell([0xF5, 0x00])[0] == 0xFFFFFFFB           # min(5, -5)
    assert f5cell([0xF5, 0x01])[0] == 0x11110000           # compare: nothing stored
    assert f5cell([0xF5, 0x02])[0] == 0x00000001           # 5 AND -5
    assert f5cell([0xF5, 0x03])[0] == 0xFFFFFFFE           # 5 XOR -5
    assert f5cell([0xF7, 0x03])[0] == 0x1111FFFE           # lower half-word store
    assert f5cell([0xF7, 0x02])[0] == 0x11110001
    assert f5cell([0xF6, 0x02])[0] == 0x00000000           # upper half-word store
    # after the compare of (5, -5) `da` fires and `d9` does not
    assert f5cell([0xF5, 0x01, 0xDA, 0xFE, 0xD8])[1].ints == 1
    assert f5cell([0xF5, 0x01, 0xD9, 0xFE, 0xD8])[1].ints == 0


def test_movx_and_preadvance():
    def setup(m):
        m.poke_words(0x70, [5])
    m, _ = run1([0xB0, 0x87, 0xBA, 0xA4, 0xDC, 0xC8], setup=setup)
    assert m.peek_word(0x240) == 0x00050000                 # probe67 D
    m, _ = run1([0xB0, 0x87, 0xB4, 0x97, 0xBA, 0xA4, 0xDC, 0x44, 0x44, 0x44], setup=const_page)
    assert [m.peek_word(0x240 + 4 * i) for i in range(4)] == [0, 0x60000000, 0x60000000, 0x60000000]
    m, _ = run1([0xB0, 0x87, 0xB4, 0x97, 0xBA, 0xA4, 0xDC, 0x44, 0xA4, 0x40], setup=const_page)
    assert [m.peek_word(0x240 + 4 * i) for i in range(2)] == [0x60000000, 0x60000000]


def test_sticky_predicates_and_int():
    # 0.5 + 0.0 > 0: d9 (N or Z) is false and skips both the INT and the MOV; da fires
    base = [0xB0, 0x87, 0xB4, 0x97, 0xDD, 0x08]
    _, r = run1(base + [0xD9, 0xFE, 0xC0, 0xD8], setup=const_page)
    assert (r.ints, len(r.fifo)) == (0, 4)
    _, r = run1(base + [0xDA, 0xFE, 0xC0, 0xD8], setup=const_page)
    assert (r.ints, len(r.fifo)) == (1, 8)
    # flags freeze while armed: a negative result under an armed da does not turn it off
    _, r = run1(base + [0xDA, 0x09, 0xFE, 0xD8], setup=const_page)
    assert r.ints == 1


def test_branch_forms():
    m, r = run1([0xB0, 0x87, 0xDD, 0xF4, 0x02, 0xC0, 0xC0, 0xC4], setup=const_page)
    assert words(r.fifo) == [0x40000000]                    # f4 02 skips two bytes
    m, r = run1([0xB0, 0x87, 0xDD, 0xF4, 0xC0], setup=const_page)
    assert words(r.fifo) == [0x20000000]                    # f4 c0 behaves as c0
    _, r = run1([0xF4, 0x44])
    assert r.ints == 1                                      # f4 4x raises DMP_INT, no branch
    with pytest.raises(Unmodelled):
        run1([0xF4, 0x60])


def test_clear_emits_zero_under_fifo():
    _, r = run1([0xDD, 0xF1, 0xE0])
    assert r.fifo == bytes(4)                               # probe69: e0 emits a zero word


def test_cycle_classes():
    """Cycle counts measured on the MPU-6050 (probe90), summed exactly."""
    _, r = run1([0xD8] * 10 + [0xC0, 0x40, 0x42, 0x87, 0xF2, 0x40, 0x08, 0xD9, 0x40, 0xD8],
                setup=const_page)
    # 110 start; d8 x10 = 50; mov 18, add 28, mul 58 (FIFO costs the same); page 5; f2 5;
    # 40 and 08 under the sticky f2 are lane adds, 22 each; d9 5; skipped 40 = 5; d8 5; ff 5
    assert r.cycles == 110 + 50 + 18 + 28 + 58 + 5 + 5 + 22 + 22 + 5 + 5 + 5 + 5
    assert r.duration_ns == pytest.approx(r.cycles * 1e9 / 8.199e6)


def test_skip_costs_per_byte():
    # a skipped `f4 00` is two bytes: 10 cycles (probe90 branch)
    _, a = run1([0xB0, 0x87, 0xB4, 0x97, 0xDC, 0x08, 0xD9, 0xF4, 0x00, 0xD8], setup=const_page)
    _, b = run1([0xB0, 0x87, 0xB4, 0x97, 0xDC, 0x08, 0xD9, 0xD8], setup=const_page)
    assert a.cycles - b.cycles == 10


def test_probe91_vendor_free_packets():
    """probe91 B1 on silicon, from zeroed memory: the hardware writes 0x000-0x00d only, so 0x00e
    stays 0 (probe84's "constant 12" was its tag), and a RAM count runs 1, 2, ..."""
    prog = bytes.fromhex("d8f1dcb9a0f8ddb080c0c2c4c6b180c0feff")
    m = Machine("mpu6050")
    m.load(prog, 0x400)
    hw1 = [0xC9EC1EF6, 0xFED2FCAB, 0xFFD90023, 0x00180000, 0x00000001]     # packet #1 as read
    r = m.run_pass(Sample(ax=-13844, ay=7926, az=-302, temp=-853, gx=-39, gy=35, gz=24))
    assert words(r.fifo) == hw1 and r.ints == 1
    r = m.run_pass(Sample(ax=1))
    assert words(r.fifo)[3] & 0xFFFF == 0 and words(r.fifo)[4] == 2


# ---- probes 95-101 (ISA2 Finding 40): what opendmp's first silicon runs uncovered ----
ONE = 1 << 30
A99 = [-ONE // 2, ONE // 4, 3 * ONE // 4, -ONE]
B99 = [ONE // 2, -ONE // 4, -3 * ONE // 4, ONE]


def flag_after(chip, op_byte, mode_byte, a_words, b_words):
    """Run `mode op` with P0 = 0x100, P1 = 0x110, P2 -> 0x120 (RAM); report F via markers."""
    icm = chip == "icm20948"
    code = ([0xF1, 0xD8] if icm else [0xD8, 0xF1]) + [0xDC, 0xB9, 0xA2, mode_byte, 0xB1, 0x80, 0xB5, 0x91,
                                                      op_byte, 0xDD, 0xB3, 0x8F, 0xD9, 0xC0, 0xDA, 0xC2, 0xD8]
    m, r = run1(code, chip=chip, entry=0x1000 if icm else 0x400,
                setup=lambda m: (m.poke_words(0x100, [w & 0xFFFFFFFF for w in a_words]),
                                 m.poke_words(0x110, [w & 0xFFFFFFFF for w in b_words]),
                                 m.poke_words(0x3F0, [0x0F0F0F0F, 0x0E0E0E0E])))
    return words(r.fifo)[0] == 0x0F0F0F0F


@pytest.mark.parametrize("chip", ["mpu6050", "icm20948"])
def test_probe99_flags(chip):
    """CMP (f5 op 1) tests signs only; RSUB, ADD, MUL give the exact sign (both parts)."""
    for i in range(4):
        for j in range(4):
            a, b = A99[i], B99[j]
            byte = (i << 5) | (j << 3)
            assert flag_after(chip, byte | 1, 0xF5, A99, B99) == (a < 0 <= b)      # "cmp"
            assert flag_after(chip, byte | 1, 0xF1, A99, B99) == (b - a < 0)      # rsub
            assert flag_after(chip, byte | 0, 0xF1, A99, B99) == (a + b < 0)      # add
            assert flag_after(chip, byte | 2, 0xF1, A99, B99) == (a * b < 0)      # mul


@pytest.mark.parametrize("chip", ["mpu6050", "icm20948"])
def test_probe100_rsub_cannot_subtract_minus_two(chip):
    for a, b, f in [(-3 * ONE // 2, 3 * ONE // 2, False), (3 * ONE // 2, -3 * ONE // 2, True),
                    (-2 * ONE, 0, True), (0, -2 * ONE, True), (-2 * ONE, 2 * ONE - 1, True),
                    (2 * ONE - 1, -2 * ONE, True), (ONE, ONE, False), (-1, 0, False), (0, -1, True)]:
        assert flag_after(chip, 0x01, 0xF1, [a, 0, 0, 0], [b, 0, 0, 0]) == f, (a, b)


@pytest.mark.parametrize("chip", ["mpu6050", "icm20948"])
@pytest.mark.parametrize("op,want", [
    (0x00, [0x90000000, 0x20000000, 0xB0000000, 0x40000000, 0x0F0F0F0F]),    # mac: +9.0, F set
    (0x01, [0x70000000, 0xE0000000, 0x50000000, 0xC0000000, 0x0E0E0E0E]),    # macn: -9.0, F clear
])
def test_probe106_mac_register_is_64_bit(chip, op, want):
    """Four MACs of 1.5 * 1.5: the stored words wrap mod 2^32, and F is the sign of a 64-bit
    accumulator (Q60 in [-8, 8)), so +9.0 sets it and -9.0 clears it (silicon, both parts)."""
    icm = chip == "icm20948"
    code = [0xF0, 0xD8, 0xDD, 0xB1, 0x80, 0xB5, 0x91] + [op] * 4
    code += [0xF1, 0xB3, 0x8F, 0xD9, 0xC0, 0xDA, 0xC2, 0xD8]
    m, r = run1(code, chip=chip, entry=0x1000 if icm else 0x400,
                setup=lambda m: (m.poke_words(0x100, [0x60000000]), m.poke_words(0x110, [0x60000000]),
                                 m.poke_words(0x3F0, [0x0F0F0F0F, 0x0E0E0E0E])))
    assert [w & 0xFFFFFFFF for w in words(r.fifo)] == want


# (A, B, the value silicon stored, F): cells from probe103, identical on both parts (Finding 41)
CLAMP_CELLS = [
    (0x80000001, 0xC0000000, 0x40000000, True),     # A < 0: max(A, -B) = -B
    (0x80000001, 0x7FFFFFFF, 0x80000001, False),    # A = -B: unchanged
    (0x80000001, 0x80000001, 0x7FFFFFFF, True),     # B < 0 too: -B
    (0x954FED61, 0xF91291C3, 0x06ED6E3D, True),
    (0x954FED61, 0x39A59521, 0xC65A6ADF, True),     # the old "-1 when B - A overflows" case
    (0x80000000, 0x00000001, 0xFFFFFFFF, True),     # probe86 F's "anomaly": max(-2.0, -1 lsb)
    (0x80000000, 0x80000000, 0x80000000, True),     # -B = +2.0 wraps back to A, yet F is set
    (0x00000005, 0xFFFFFFFB, 0xFFFFFFFB, True),     # A >= 0: min(A, B)
    (0x00000000, 0xC0000000, 0xC0000000, True),     # A = 0 takes the min side
    (0x30000000, 0x40000000, 0x30000000, False),    # inside the limits: unchanged
    (0xD0000000, 0x40000000, 0xD0000000, False),
    (0x60000000, 0x40000000, 0x40000000, True),     # clamped to +B
    (0xA0000000, 0x40000000, 0xC0000000, True),     # clamped to -B
]


@pytest.mark.parametrize("chip", ["mpu6050", "icm20948"])
def test_probe103_min_is_a_symmetric_clamp(chip):
    for a, b, r, f in CLAMP_CELLS:
        icm = chip == "icm20948"
        code = ([0xF5, 0xD8] if icm else [0xD8, 0xF5]) + [0xDC, 0xB1, 0x80, 0xB5, 0x91, 0xB9, 0xA2, 0x00]
        m, _ = run1(code, chip=chip, entry=0x1000 if icm else 0x400,
                    setup=lambda m: (m.poke_words(0x100, [a]), m.poke_words(0x110, [b])))
        assert m.peek_word(0x120) & 0xFFFFFFFF == r, (hex(a), hex(b))
        assert flag_after(chip, 0x00, 0xF5, [a, 0, 0, 0], [b, 0, 0, 0]) == f, (hex(a), hex(b))
    # on the lanes the rule runs on the half-words; only that half is written (probe103 C)
    code = [0xD8, 0xF7, 0xDC, 0xB1, 0x80, 0xB5, 0x91, 0xB9, 0xA2, 0x00]
    m, _ = run1(code, setup=lambda m: (m.poke_words(0x100, [0x1234_8000]), m.poke_words(0x110, [0x0000_0001]),
                                       m.poke_words(0x120, [0x5A5A_5A5A])))
    assert m.peek_word(0x120) & 0xFFFFFFFF == 0x5A5A_FFFF


def test_probe101_branch_is_signed_6_bit():
    head = [0xD8, 0xF1, 0xDC, 0xB9, 0xA0, 0xDE]            # P2 -> 0x100, cleared
    for d in range(0x20):                                   # forward: 64 incs, landing +d
        m, r = run1(head + [0xF4, d] + [0xF8] * 64 + [0xD8, 0xFE])
        assert m.peek_word(0x100) == 64 - d
    for d in range(0x22, 0x3A):                             # backward into a 24-inc sled
        m, r = run1(head + [0xF4, 0x1C] + [0xF8] * 24 + [0xD8, 0xFE, 0xFF, 0xFF] + [0xF4, d, 0xD8, 0xFE])
        assert m.peek_word(0x100) == 0x420 - (0x426 + d - 64)


def test_probe96_f4_leaves_mode_4():
    ptrs = [0xDC, 0xB1, 0x80, 0xB5, 0x91, 0xB9, 0xA2]                   # RAM; P0, P1, P2
    setup = lambda m: m.poke_words(0x100, [0x20000000, 0xF0000000, 0, 0, 0x10000000, 0, 0, 0,
                                           0xEEEEEEEE])
    m, r = run1([0xF1] + ptrs + [0xF4, 0x00, 0xC2], setup=setup)       # MOV works in mode 4
    assert m.peek_word(0x120) & 0xFFFFFFFF == 0xF0000000
    for pre in ([0xF4, 0x00], [0xF4, 0x40], [0xF4, 0xB0], [0x08, 0xDA, 0xF4, 0x00, 0xD8]):
        m0, _ = run1([0xF1] + ptrs + pre, setup=setup)
        m, r = run1([0xF1] + ptrs + pre + [0x00], setup=setup)          # 00 is a branch there, not ADD
        assert m.peek_word(0x120) == m0.peek_word(0x120)
        with pytest.raises(Unmodelled, match="mode 4"):                # 40-7f: no measured meaning
            run1([0xF1] + ptrs + pre + [0x40], setup=setup)
    m, r = run1([0xF1] + ptrs + [0xF4, 0x00, 0xF1, 0x00], setup=setup)   # a mode byte ends it
    assert m.peek_word(0x120) == 0x30000000


def test_probe104_mode4_one_byte_branch():
    """In mode 4 an MPU byte below 0x40 is a one-byte branch to pc + 1 + sext6(b) (Finding 42)."""
    head = [0xD8, 0xF1, 0xDC, 0xB9, 0xA0, 0xDE]            # P2 -> 0x100, cleared
    for b in (0x00, 0x05, 0x1F):                           # forward over a 40-inc sled
        m, r = run1(head + [0xF4, 0x00, b] + [0xF8] * 40 + [0xD8, 0xFE])
        assert m.peek_word(0x100) == 40 - b
    # back: `f4 01` over the hop byte into a sled, then 0x3c (-4) from after the sled's end
    m, r = run1(head + [0xF4, 0x05, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xD8, 0xFE, 0xFF, 0xFF])
    assert m.peek_word(0x100) == 0
    # predicated: a false d9 skips it (F clear after an ALU op leaving 0)
    zero = [0xB1, 0x8F, 0xB5, 0x9F, 0xDD, 0x01, 0xDC]         # P0, P1 -> zeros; FIFO a 0 - 0
    m, r = run1(head + zero + [0xF4, 0x00, 0xD9, 0x02, 0xD8] + [0xF8] * 4 + [0xD8, 0xFE])
    assert m.peek_word(0x100) == 4
    m, r = run1(head + zero + [0xF4, 0x00, 0xDA, 0x02, 0xD8] + [0xF8] * 4 + [0xD8, 0xFE])
    assert m.peek_word(0x100) == 3                          # taken: over d8 and one of the incs


def _r_cell(chip, x, y, body, tail):
    """probe107's cell: P0 -> [x, 0.5, 0, 0] at 0x100, P1 -> [y, 0.5, 0, 0] at 0x110, P2 -> 0x120,
    R := 1.0 (add 1,1), then body, then tail."""
    from opendmp.isa import assemble
    icm = chip == "icm20948"
    e = 0x1000 if icm else 0x400
    src = (f".chip {chip}\n.entry {e:#x}\n.org {e:#x}\nfmode 1\nalways\nstore\np0bank 1\np0page 0\n"
           f"p1bank 1\np1page 1\np2bank 1\np2page 2\nadd 1,1\n{body}{tail}")
    code = assemble(src, chip).code()
    return run1(code[:-1] if code.endswith(b"\xff") else code, chip=chip, entry=e,
                setup=lambda m: (m.poke_words(0x100, [x & 0xFFFFFFFF, 0x20000000, 0, 0]),
                                 m.poke_words(0x110, [y & 0xFFFFFFFF, 0x20000000, 0, 0])))[0]


@pytest.mark.parametrize("chip", ["mpu6050", "icm20948"])
def test_probe107_test_reads_the_result_register(chip):
    """d1 tests R's word (wrapped) at the lane width in force; CLAMP leaves its compare in R;
    AND/XOR/MOV leave R; on the MPU a P2 move zeroes R, on DMP3 it does not (silicon, both)."""
    def f(x, y, body):
        return _r_cell(chip, x, y, body, "test\nend\n").f
    assert f(-1 << 31, -1 << 31, "add 0,0\n")                 # -2 + -2 wraps to 0: F set
    assert f(-1 << 31, -1 << 31, "mul 0,0\n")
    assert f(0x8000, 0x8000, "fmode 3\nadd 0,0\n")            # the low lane wraps to 0
    assert not f(0x8000, 0x8000, "fmode 3\nadd 0,0\nfmode 1\n")   # ... the word does not
    assert not f(0, 0, "fmode 5\nand 0,0\n")                  # AND leaves R = 1.0
    assert f(5, 5, "fmode 5\nclamp 0,0\n") and not f(-5, -5, "fmode 5\nclamp 0,0\n")
    assert f(3, 3, "p2page 2\n") == (chip == "mpu6050")
    assert f(3, 3, "p2bank 1\n") == (chip == "mpu6050")
    assert not f(3, 3, "p0page 5\n")


@pytest.mark.parametrize("chip", ["mpu6050", "icm20948"])
@pytest.mark.parametrize("x,y,body,want", [
    (5, 7, "fmode 5\nclamp 0,0\n", 2), (-5, 7, "fmode 5\nclamp 0,0\n", 2),
    (9, 7, "fmode 5\nclamp 0,0\n", -2), (5, 6, "fmode 5\nxor 0,0\n", 0x40000000),
    (3, 3, "fmode 1\ninc\n", 0x40000001), (3, 3, "fmode 1\nmov 0\n", 0x40000000),
])
def test_probe107_mac_continues_from_r(chip, x, y, body, want):
    m = _r_cell(chip, x, y, body, "fmode 0\nmac 2,2\nend\n")
    assert m.peek_word(0x120) == want


@pytest.mark.parametrize("chip", ["mpu6050", "icm20948"])
def test_probe113_pass_carry(chip):
    """R carries from pass to pass (a lone MAC run accumulates across passes); the route after
    DMP_RST is RAM; DMP3 starts every pass with its pointers at 0, the MPU keeps them (silicon)."""
    from opendmp.isa import assemble
    e = 0x1000 if chip == "icm20948" else 0x400
    src = (f".chip {chip}\n.entry {e:#x}\n.org {e:#x}\nfmode 0\nalways\nfifo\np0bank 1\np0page 0\n"
           "p1bank 1\np1page 1\nmac 0,0\nend\n")
    m = Machine(chip)
    for a, d in assemble(src, chip).segments():
        m.load(d, a)
    m.poke_words(0x100, [0x20000000])
    m.poke_words(0x110, [0x20000000])
    got = [words(m.run_pass(Sample()).fifo)[0] for _ in range(4)]
    assert got == [0x10000000, 0x20000000, 0x30000000, 0x40000000]
    m = Machine(chip)
    assert m.route_ram
    m.load(bytes([0xF1, 0xD8, 0xB2, 0x85, 0xFF]), e)    # P0 := 0x250, end
    m.run_pass(Sample())
    m.begin_pass(Sample())
    assert m.p[0].addr() == (0 if chip == "icm20948" else 0x250)


@pytest.mark.parametrize("chip", ["mpu6050", "icm20948"])
@pytest.mark.parametrize("body,want", [
    ("fmode 1\ninc\n", 0x40000001), ("fmode 1\ndec\n", 0x3FFFFFFF), ("fmode 1\ninc\ninc\n", 0x40000002),
    ("fmode 5\nclamp 0,0\n", 2),
])
def test_probe116_mac_continues_after_a_move(chip, body, want):
    """After a P2 move, a MAC continues from what INC, DEC or CLAMP wrote into R: on the MPU the
    move zeroes R and that is all the restart there is (silicon, both parts, probe116)."""
    x, y = (5, 7) if "clamp" in body else (3, 3)
    m = _r_cell(chip, x, y, "p2page 2\n" + body, "fmode 0\nmac 2,2\nend\n")
    assert m.peek_word(0x120) & 0xFFFFFFFF == want
