# Examples

Twelve programs, written in the opendmp language (PLAN.md §2). The compiler that builds them
does not exist yet, so for now they serve as the language's specification and as the
compiler's acceptance suite. Each must build for both chips, simulate bit-exact against the
emulator, and run on both rigs.

All twelve pass a range check on synthetic motion (rest, rotation, walking, taps, 10 Hz
vibration, free fall and a saturating shake): no value leaves [-2, 2), and every declared
range and `dmp.assume` holds.

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
| `unit_vectors.py` | exact normalisation via range reduction + Newton | `@dmp.sub`, `dmp.loop` + `break`, early `return`, several return values |

`emu/` is different: it is an **emulator demo** you can run today. `tilt.asm` is PLAN.md §3's
hand-compiled tilt filter in opendmp assembly, with its data. `tilt.s` is the same program in
dmpasm syntax. `tilt.toml` is the run configuration, and `tilt_walk.csv` is generated motion:

```bash
cd examples/emu
python -m opendmp.emu.gen tilt,walk,rotate --rate 200 -o tilt_walk.csv
python -m opendmp.emu tilt.toml          # writes out/tilt_passes.csv, out/tilt.fifo, out/tilt_trace.txt
python -m opendmp asm tilt.asm           # the listing, with cycles per instruction
```
