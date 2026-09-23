"""ICM-20948 (DMP3) semantics against recorded hardware results."""
import pytest

from opendmp.emu import DmpFault, Machine, Model, Sample, Unmodelled
from conftest import RE_REPO, run1, words

ICM = "icm20948"


def test_probe47_blocks():
    def setup(m):
        for a, v in ((0x3F0, 0x03F0), (0x7F0, 0x07F0), (0xBF0, 0x0BF0), (0xFF0, 0x0FF0)):
            m.poke_words(a, [v])
    for pre, want in (([], 0x3F0), ([0xBC], 0x7F0), ([0xBC] * 2, 0xBF0), ([0xBC] * 3, 0xFF0),
                      ([0xBC] * 4, 0x3F0)):
        _, r = run1([0xF1, 0xDD] + pre + [0xB3, 0x8F, 0xC0], chip=ICM, setup=setup)
        assert words(r.fifo) == [want]


P51 = [0x01234567, 0x89ABCDEF, 0x12345678, 0x9ABCDEF0]


@pytest.mark.parametrize("body,want", [
    ([0xD4, 0xFD, 0x08], [0x23456726]),
    ([0xD5, 0xFD, 0x28], [0xFFFFFF89]),
    ([0xD7, 0xFD, 0x08], [0xBCDEF004]),
    ([0xD5, 0xFD, 0x08, 0xFD, 0x48], [0xABCDEF48, 0xCDEF48D1]),
])
def test_probe51_shifter(body, want):
    _, r = run1([0xF1, 0xDD, 0xB3, 0x8F] + body, chip=ICM, setup=lambda m: m.poke_words(0x3F0, P51))
    assert words(r.fifo) == want


def test_bare_shift_stores_zero():
    _, r = run1([0xF1, 0xDD, 0xFD, 0x08], chip=ICM)
    assert words(r.fifo) == [0]


def test_probe51_product_low_word():
    def setup(m):
        m.poke_words(0x3F0, [0x01234567])
        m.poke_words(0x3E0, [0x89ABCDEF])
    _, r = run1([0xF1, 0xDD, 0xB3, 0x8F, 0xB7, 0x9E, 0x02, 0xD3], chip=ICM, setup=setup)
    assert words(r.fifo) == [0xFDE578DB, 0x094E4629]


def test_probe83b_immediate_compare():
    def setup(m):
        m.poke_words(0x3E0, [0x12345682])                  # low 6 bits = 2
    for y, fires in ((0x08, True), (0x0C, False), (0x05, True), (0x0A, False), (0x0B, False), (0x0F, True)):
        _, r = run1([0xF1, 0xB7, 0x9E, 0xD8, 0xFC, 0xC0, y, 0xD9, 0xE2, 0xD8], chip=ICM, setup=setup)
        assert r.ints == (1 if fires else 0), hex(y)
    with pytest.raises(Unmodelled):
        run1([0xFC, 0x51, 0x00], chip=ICM)


def test_call_ret_and_mode4_hygiene():
    m = Machine(ICM)
    m.load(bytes([0xF1, 0xF4, 0x50, 0x10, 0xF1, 0xFF]), 0x1000)     # call 0x1010; f1; halt
    m.load(bytes([0xD8, 0xF1, 0xE2, 0xE0]), 0x1010)                 # int; ret
    r = m.run_pass(Sample())
    assert r.ints == 1 and m.halted


def test_ret_without_call_kills_the_dmp():
    with pytest.raises(DmpFault):
        run1([0xF1, 0xE0], chip=ICM)


def test_fifth_call_cycles_forever():
    m = Machine(ICM, Model(max_steps=5000))
    m.load(bytes([0xF4, 0x50, 0x10, 0xFF]), 0x1000)
    for a in (0x1010, 0x1020, 0x1030, 0x1040):                       # five nested calls
        m.load(bytes([0xD8, 0xF1, 0xF4, 0x50 | ((a + 0x10) >> 8 & 0x3F), (a + 0x10) & 0xFF, 0xF1, 0xE0]), a)
    m.load(bytes([0xD8, 0xF1, 0xE0]), 0x1050)
    with pytest.raises(DmpFault):
        m.run_pass(Sample())


def test_skipped_jump_still_enters_mode4():
    # `d9 f4 hi lo  d8 hi' lo'`: the if-jump is skipped, the bare target after d8 jumps
    m = Machine(ICM)
    m.poke_words(0x3F0, [0x20000000])
    m.load(bytes([0xF1, 0xB3, 0x8F, 0xB7, 0x9F, 0xDD, 0x00,           # emit 0.5+0.5 > 0: F clear
                  0xD9, 0xF4, 0x10, 0x40, 0xD8, 0x10, 0x30]), 0x1000)
    m.load(bytes([0xD8, 0xF1, 0xE2, 0xFF]), 0x1030)                   # else-target: int
    m.load(bytes([0xD8, 0xF1, 0xFF]), 0x1040)
    r = m.run_pass(Sample())
    assert r.ints == 1


def test_loop_icm_counts_n_plus_one(tmp_path):
    src = RE_REPO / "programs/loop_icm.s"
    if not src.is_file():
        pytest.skip("InvensenseDMP not present")
    from opendmp.emu.files import read_program
    code = read_program(src, ICM)
    for n in (0, 1, 5):
        m = Machine(ICM)
        m.poke_words(0x3F0, [0x11223344, n])
        m.load(code, 0x1000)
        r = m.run_pass(Sample())
        assert words(r.fifo) == [0x11223344] * (n + 1)


def test_samples_and_io_space():
    m = Machine(ICM)
    m.write_sample(Sample(ax=1, ay=2, az=3, gx=4, gy=5, gz=6, temp=7))
    assert m.mem[0:18].hex() == "00000000" + "0001000200030004000500060007"
    # a page load under mode 4 addresses the register file: word 0 = WHO_AM_I
    _, r = run1([0xF1, 0xDD, 0xF4, 0xB0, 0x80, 0xC0, 0xF1], chip=ICM)
    assert words(r.fifo) == [0xEA]


def test_icm_cycles():
    # call S; f1 with S = d8 f1 e0 costs 30 cycles; jump + f1 costs 17 (probe90 branch)
    m = Machine(ICM)
    m.load(bytes([0xF1, 0xF4, 0x50, 0x10, 0xF1, 0xFF]), 0x1000)
    m.load(bytes([0xD8, 0xF1, 0xE0]), 0x1010)
    r = m.run_pass(Sample())
    assert r.cycles == 130 + 4 + 30 + 4                  # start, f1, the call unit, halt
    m = Machine(ICM)
    m.load(bytes([0xF1, 0xF4, 0x10, 0x04, 0xF1, 0xFF]), 0x1000)
    r = m.run_pass(Sample())
    assert r.cycles == 130 + 4 + 17 + 4


def test_probe91_fifo_is_ram_at_0x3000():
    """probe91: every FIFO byte also lands in DMP RAM from 0x3000 (and 0x0000-0x0003 read 0)."""
    prog = bytes.fromhex("f1d8dcb9a0f8ddb080c0c2c4c6b081c0b180c0e2ff")
    m = Machine(ICM)
    m.load(prog, 0x1000)
    r1 = m.run_pass(Sample(ax=-168, ay=-1804, az=-8020, gx=1, gy=3, gz=6, temp=4432))
    assert words(r1.fifo) == [0, 0xFF58F8F4, 0xE0AC0001, 0x00030006, 0x11500000, 1]
    r2 = m.run_pass(Sample(ax=7))
    assert bytes(m.mem[0x3000:0x3000 + 48]) == r1.fifo + r2.fifo
    assert m.chip.code_limit == 0x3000
