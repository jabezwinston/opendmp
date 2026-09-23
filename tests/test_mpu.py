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
    # reset at the start of every pass
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
