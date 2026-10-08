import pytest

from opendmp.emu import Machine, Sample, run

SAMPLES = [Sample(ax=i) for i in range(30)]


def filler_program(m, n):
    m.load(bytes([0xD8] * n + [0xFF]), m.entry)


def test_mpu_drops_to_integer_submultiples():
    m = Machine("mpu6050")
    m.load(bytes([0xDC] + [0x40] * 600 + [0xFF]), m.entry)   # 56 + 1.2 + 600*3.9 = 2397 us
    rs = run(m, SAMPLES, rate_hz=1000)
    assert [r.sample for r in rs[:4]] == [0, 3, 6, 9]


def test_icm_runs_at_one_over_pass_time():
    m = Machine("icm20948")
    filler_program(m, 9000)                         # 16 + 9001*0.19 = 1726.2 us, period 880
    rs = run(m, SAMPLES, rate_hz=1136)
    gaps = [b.start_ns - a.start_ns for a, b in zip(rs, rs[1:])]
    assert all(g == pytest.approx(rs[0].duration_ns) for g in gaps)


def test_short_passes_see_every_sample():
    m = Machine("mpu6050")
    filler_program(m, 10)
    rs = run(m, SAMPLES, rate_hz=200)
    assert [r.sample for r in rs] == list(range(30))


def _copy_twice(m, gap):
    """store; P2 -> 0x200; copy sample word 0; `gap` cheap bytes; copy it again into 0x204."""
    m.load(bytes([0xF1, 0xD8, 0xDC, 0xBA, 0xA0, 0xC0] + [0xD8] * gap + [0xC1, 0xFF]), m.entry)


@pytest.mark.parametrize("midpass", [True, False])
def test_long_pass_sees_the_next_sample(midpass):
    """The sample page is rewritten at every tick, even under a running pass (InvensenseDMP
    Finding 47): a 1.2 ms pass at 1 kHz reads sample 1 at its end; the write stalls it 20 cycles."""
    m = Machine("mpu6050")
    _copy_twice(m, 2000)
    samples = [Sample(ax=2 * i + 1, ay=2 * i + 2) for i in range(4)]
    r = run(m, samples, rate_hz=1000, max_passes=1, midpass=midpass)[0]
    assert m.peek_word(0x200) == 0x00010002
    assert m.peek_word(0x204) == (0x00030004 if midpass else 0x00010002)
    if midpass:
        assert (r.ticks, r.stall_cycles) == (1, 20)
        assert r.duration_ns == pytest.approx((r.cycles + 20) * m.chip.cycle_ns)


def test_short_pass_sees_one_sample():
    m = Machine("mpu6050")
    _copy_twice(m, 10)
    r = run(m, [Sample(ax=1, ay=2), Sample(ax=3, ay=4)], rate_hz=1000, max_passes=1)[0]
    assert m.peek_word(0x204) == m.peek_word(0x200) and r.ticks == 0


@pytest.mark.parametrize("chip,n", [("mpu6050", 300), ("icm20948", 1100)])
def test_fifo_keeps_the_newest(chip, n):
    """One pass emits more than the FIFO holds (MPU 1 KB, ICM 4 KB): the oldest bytes are lost
    and, on the ICM, the ring at 0x3000 holds the newest (InvensenseDMP Finding 52, probe114)."""
    m = Machine(chip)
    head = [0xF1, 0xD8, 0xB2, 0x80, 0xBA, 0xA0]               # P0 = P2 = 0x200
    m.load(bytes(head + [0xDC, 0xF8, 0xDD, 0xC0] * n + [0xFF]), m.entry)   # count += 1; emit it
    r = m.run_pass(Sample())
    cap = m.chip.fifo_size
    words = [int.from_bytes(r.fifo[i:i + 4], "big") for i in range(0, len(r.fifo), 4)]
    assert words == list(range(n - cap // 4 + 1, n + 1))
    assert m.fifo_lost == 4 * n - cap
    if chip == "icm20948":
        ring = [int.from_bytes(m.mem[0x3000 + i:0x3004 + i], "big") for i in range(0, cap, 4)]
        k = n - cap // 4                                        # the ring's newest wrapped words
        assert ring[:k] == list(range(cap // 4 + 1, n + 1)) and ring[k:] == list(range(k + 1, cap // 4 + 1))
