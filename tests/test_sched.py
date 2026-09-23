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
