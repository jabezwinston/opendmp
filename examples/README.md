# Examples

Twelve programs, written in the opendmp language (PLAN.md §2). Each one builds for both chips
(`python -m opendmp build examples/tilt.py`). Its bytes run in the emulator bit-exact against
the reference semantics (`tests/test_backend.py`), and it runs bit-exact on both bench rigs
(`python -m opendmp run examples/tilt.py --port /dev/ttyUSB0 --verify`). The main README has
the code size and cycles of each, and [docs/tutorial.md](../docs/tutorial.md) walks through `tilt.py`.

All twelve pass `python -m opendmp check` with no range warning: interval analysis proves that
no value can wrap. Where it cannot see a fact, the program states it with `dmp.assume` and a
comment giving the reason:
- `biquad`: the filter's gain, the L1 norm of its impulse response, 1.091;
- `goertzel`: the resonator's energy;
- `mahony`: |a × v| ≤ √3;
- `gyro_quat`, `dmp.quat`: unit quaternion components;
- `unit_vectors`: the Newton and range-reduction invariants.

`python -m opendmp sim` runs each on synthetic motion (rest, rotation, walking, taps, 10 Hz
vibration, free fall and a saturating shake). No value wraps, and every assumption holds on
every pass.

Two examples changed when the checker first ran:
- `goertzel` now scales its input by 1/256: at 1/128, a full-scale resonant input would push
  the resonator to 1.28, past its declared range.
- `unit_vectors` now recomputes |v|² on every doubling trip, and allows 15 trips. Keeping
  `m * 4` let the floor of a tiny vector's x² grow into a 12 % error in |u|, and 14 trips were
  one too few for a 1-LSB sample.

| file | algorithm | exercises |
|---|---|---|
| `hello_fifo.py` | stream the raw accelerometer | sensor extraction, 16-bit emits (milestone M1) |
| `tilt.py` | complementary filter, one axis | ranged state + clamp, a param, a MAC run, a compare, `irq` |
| `biquad.py` | 2nd-order Butterworth low-pass on 3 axes, designed at compile time | a five-term MAC run, a helper updating state lists, `for` over axes |
| `despike.py` | median-of-3 spike filter + low-pass | `min`/`max`, a delay line in state lists |
| `vibration.py` | running mean / mean-square, alarm with hysteresis | EMA statistics, compile-time thresholds, an edge-triggered interrupt |
| `pedometer.py` | peak counting on \|a\|², with hysteresis and a minimum gap | sum of squares, `Int` counters and timers, `and` conditions |
| `taps.py` | single / double tap from jerk | a state machine with two countdown timers, `abs`, nested `if` |
| `gyro_quat.py` | strapdown quaternion integration + Newton renormalisation | quaternion in a state list, helpers returning lists, `dmp.assume` |
| `mahony.py` | 6-axis AHRS (Mahony PI filter) | `dmp.quat`, cross products, clamps, gains folded at compile time |
| `pid_level.py` | PID levelling with anti-windup | ranged params, saturation, derivative on the measurement |
| `goertzel.py` | single-bin DFT tone detector (10 Hz) | a resonator with a coefficient near 2, a block counter, DC removal |
| `unit_vectors.py` | exact normalisation via range reduction + Newton | `@dmp.sub` (a real call on the ICM), `dmp.loop` + `break`, early `return`, several return values |

`emu/` is different: it is an **emulator demo** in assembly. `tilt.asm` is PLAN.md §3's
hand-compiled tilt filter in opendmp assembly, with its data. `tilt.s` is the same program in
dmpasm syntax. `tilt.toml` is the run configuration, and `tilt_walk.csv` is generated motion:

```bash
cd examples/emu
python -m opendmp.emu.gen tilt,walk,rotate --rate 200 -o tilt_walk.csv
python -m opendmp.emu tilt.toml          # writes out/tilt_passes.csv, out/tilt.fifo, out/tilt_trace.txt
python -m opendmp asm tilt.asm           # the listing, with cycles per instruction
```
