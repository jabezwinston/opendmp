# opendmp tutorial

This walks one program from source to silicon, then through what larger programs need. It
assumes Python 3.11+. A bench rig (an ESP32 bridge wired to an MPU-6050 or ICM-20948, see
InvensenseDMP `bench/`) is needed only for §6. The language itself is in
[language.md](language.md).

## 1. A first program

`examples/tilt.py` estimates one tilt angle with a complementary filter. The gyro gives the
short-term change, and the accelerometer pulls the long-term value back to gravity:

```python
import math

import opendmp as dmp

RATE = 200                              # Hz
K = 0.98                                # gyro weight
GYRO_RAD = math.radians(1000)           # 1.0 gyro unit = 1000 dps at +/-2000 dps full scale
GYRO_DT = GYRO_RAD / RATE               # rad per gyro unit per pass: 0.0873


@dmp.program(rate=RATE, gyro_fs=2000, accel_fs=2)
def tilt(s):
    angle: dmp.state(-1.5, 1.5) = 0.0   # radians, kept across passes
    limit: dmp.param = 0.25             # radians, host-adjustable

    angle = dmp.clamp(K * (angle + s.gyro.y * GYRO_DT) + (1 - K) * s.accel.x, -1.5, 1.5)
    dmp.emit(angle, s.accel.z)
    if angle > limit:
        dmp.irq()
```

What each part means:
- **`@dmp.program(rate=200)`**: the function runs on the DMP once per sensor sample, 200 times a
  second. Python never calls it; the compiler reads its source.
- **`s`**: this pass's sample. `s.accel.x` is in g and `s.gyro.y` in units of 1000 deg/s, both
  as Q30 fixed-point values in [−2, 2).
- **`angle: dmp.state(...) = 0.0`**: a word that keeps its value from pass to pass. The range is
  an *assumption* for the checker. `dmp.clamp` is what makes it true.
- **`limit: dmp.param`**: a word the host can change while the program runs.
- **The update line**: exact sums of products. The DMP computes each in one multiply-accumulate
  run, in full precision, rounded once. The factor `(angle + s.gyro.y * GYRO_DT)` is a sum
  itself, so it is computed first, then `K` times it plus `(1 - K)` times the accelerometer.
- **`dmp.emit`**: builds this pass's packet: `angle` (4 bytes, Q30) and `accel_z` (2 bytes, the
  raw int16).
- **`dmp.irq()`**: raises the DMP interrupt on the passes where the angle exceeds the limit.

Module-level names (`K`, `GYRO_DT`) are ordinary Python evaluated at compile time. So `math`
works, and only the value that reaches the DMP must fit in [−2, 2).

## 2. Check it

```
$ python -m opendmp check examples/tilt.py
tilt: 200 Hz, gyro +/-2000 dps, accel +/-2 g; 1 state word, 1 param
  packet 6 B: angle q30, accel_z i16
  nothing can wrap
```

`check` runs interval analysis over the whole pass, including states fed back from earlier
passes. DMP arithmetic wraps rather than saturating, so the checker reports every place a value
could leave [−2, 2). Delete the `dmp.clamp` and see:

```
tilt.py:33:5: warning: angle may leave its declared range [-1.5, 1.5]: this can store [-1.681, 1.681]
        angle = K * (angle + s.gyro.y * GYRO_DT) + (1 - K) * s.accel.x
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
hint: clamp the value (dmp.clamp) where it is computed, widen the range, or dmp.assume(...) what interval arithmetic cannot see
```

A fast enough spin really would carry the integral past ±1.5, which is why the clamp is there.
A syntax the DMP cannot do is an error with a hint. `x / y` says the DMP has no divide, and
points at a constant reciprocal or at the Newton iteration in `examples/unit_vectors.py`.

## 3. Simulate it

```
$ python -m opendmp sim examples/tilt.py --gen tilt,walk --seconds 1
tilt: 400 passes (reference), 179 interrupts, packet 6 B
  angle        q30   min   -0.0203809  max      0.50468  last    -0.019779
  accel_z      s16   min      0.59613  max      1.40546  last     0.971252
```

`--gen` makes synthetic motion (rest, tilt, rotate, walk, taps, vibration, freefall, shake), and
`--sensors data.csv` reads recorded samples. `-o out.csv` writes the decoded packets, and
`--param limit=0.4` sets a param first.

There are three engines, and they must agree bit for bit:
- `--engine reference` (the default) evaluates the source directly;
- `--engine ir` runs the compiler's intermediate form;
- `--engine emu:mpu6050` / `emu:icm20948` compiles the program and runs **its bytes** in a
  cycle-accurate emulator. That engine also prints the pass time:

```
$ python -m opendmp sim examples/tilt.py --gen tilt --seconds 1 --engine emu:mpu6050
tilt: 200 passes (emu:mpu6050), 136 interrupts, packet 6 B
  pass time   662-667 cycles = 80.7-81.4 us
```

From Python: `tilt.simulate(samples, engine="emu:icm20948")`, with samples from
`opendmp.emu.gen.generate(...)` or `opendmp.emu.read_sensor_csv(...)`.

## 4. Build it and read the listing

```
$ python -m opendmp build examples/tilt.py --list
```

This writes `build/tilt.{mpu6050,icm20948}.{lst,asm,json,bin}`. For each chip it says how every
construct was compiled, with its line:

```
tilt for the mpu6050: 45 B of code, packet 6 B, at most 82.0 us per pass (1.6 % of the period)
  tilt.py:33  `dmp.clamp(K * (angle + s.gyro.y * GYRO_DT) + ...`: one CLAMP instruction (f5 op 0): x clamped to +-1.5, exact for every value (Finding 41)
  tilt.py:33  `angle + s.gyro.y * GYRO_DT`: one multiply-accumulate run of 2 terms (...), summed exactly (Q60) and floored once
  tilt.py:35  `angle > limit`: RSUB into a scratch word, then a branch on its sign (the f5 compare only tests signs)
```

With `--list` it also prints the listing, where the same notes sit under each source line:

```
; addr  bytes     asm                    effect                                   cycles
        ; tilt.py:33  angle = dmp.clamp(K * (angle + s.gyro.y * GYRO_DT) + (1 - K) * s.accel.x, -1.5, 1.5)
040f  f0        fmode 0                multiply-accumulate                          5
0410  83        p0page 3               P0 page 3                                    5
0411  b4        p1bank 0               P1 bank 0                                    5
0412  91        p1page 1               P1 page 1                                    5
0413  14        mac 0,2,inc            P2 word += 1; acc + P0.0 * P1.2             60
0414  94        p1page 4               P1 page 4                                    5
0415  68        mac 3,1                acc + P0.3 * P1.1                           60
...
041b  f5        fmode 5                clamp/cmp/and/xor, 32-bit                    5
041c  14        clamp 0,2,inc          P2 word += 1; P0.0 clamped to +-P1.2; F := it was    28
        ; tilt.py:34  dmp.emit(angle, s.accel.z)
041d  dd        fifo                   route -> FIFO                                5
041e  c2        mov 1                  P0.1                                        18
```

Each source line heads the instructions it became. The columns are the address, the bytes, the
assembly, what the instruction does, and its cost in DMP clock cycles, as measured on the bench.
Some patterns in this one:
- **The filter is two MAC runs.** The first sums `angle * 1.0 + gyro * GYRO_DT` into a temporary.
  The second sums `K * that + (1-K) * accel`. Each run adds its products exactly and floors once.
  On the MPU a run restarts when its destination moves, here the `inc` and the `p2page`.
- **The clamp is one instruction.** The DMP's `f5` op 0 clamps A to ±B (ISA2.md Finding 41).
- **The pointer loads** (`p0page`, `p1bank` …) are the machine's addressing: every operand is
  named by a pointer's page. The compiler places words to need as few as it can.

The listing ends with the data words and the packet layout, and the pass's cost:

```
; pass   45 B of code; at most 672 cycles = 82.0 us after the tick (1.6 % of 5000 us)
```

The `.asm` is the same program as opendmp assembly: it reassembles to the same bytes
(`python -m opendmp asm build/tilt.mpu6050.asm`). The `.json` is the packet schema and symbol
table the host needs.

## 5. The budget

A pass must finish inside the sample period, 5 ms at 200 Hz. The listing's footer gives an
upper bound: the longest path through the code, with loops at their trip counts and calls at
their routines' bounds. For unit_vectors on the MPU it reads

```
; pass   359 B of code; at most 23071 cycles = 2813.9 us after the tick (28.1 % of 10000 us); loops at their trip counts; 10 branch hops
```

A `while` loop has no trip count, so there the footer says the bound is open; the emulator
(`sim --engine emu:...`) still counts real passes. A bound over the period is flagged. As a rule
of thumb on the MPU-6050 (8.2 MHz):
- 5 cycles for a pointer, mode or predicate byte;
- 18 for a MOV, 28 for an add or compare;
- 58 for a multiply, 60 for a MAC step.

The ICM-20948 runs at 21 MHz, where a multiply costs the same as an add.

## 6. Run it on silicon

With a bench rig on `/dev/ttyUSB0` (the chip is detected):

```
$ python -m opendmp run examples/tilt.py --port /dev/ttyUSB0 --seconds 5 -o tilt.csv
```

The runtime resets the part and zeroes DMP memory. It then writes a minimal register set and the
program, and starts it. **No vendor firmware is involved**: the program is the DMP's whole
program. It streams and decodes the packets.

**Check the compiled code against the models:**

```
$ python -m opendmp run examples/tilt.py --port /dev/ttyUSB0 --verify --seconds 2
tilt on the mpu6050: 403 passes at 200.0 Hz; emulator 0 and reference 0 packets differ; interrupts 0 on silicon, 0 emulated; 749-749 cycles a pass: BIT-EXACT
```

A verify run uses a *trace build*: every packet starts with that pass's raw sensor sample. The
samples are replayed through the emulator and the reference, and every packet and interrupt must
match.

**Params and state, live:**

```
$ python -m opendmp run examples/pid_level.py --port /dev/ttyUSB0 --seconds 4 \
      --param setpoint=0.1 --set 2:setpoint=-0.1 --watch integ,pitch
```

- `--param` sets a param before the start.
- `--set T:NAME=VALUE` changes it T seconds in.
- `--watch` reads states while the program runs, and `--watch-output watch.csv` saves them.

A param with a declared range (`dmp.param(-0.3, 0.3)`) refuses values outside it. From Python:

```python
from opendmp.host import Rig
rig = Rig("/dev/ttyUSB0")
rig.load(prog.build(rig.chip))
packets, ints = rig.run(5, watch=["angle"], changes=[(2.0, "limit", 0.3)])
rig.read("angle"); rig.write("limit", 0.1)
```

## 7. Bigger programs

The twelve programs in `examples/` each exercise something:

| to learn | read |
|---|---|
| lists of states, a helper that updates them, `for` over axes | `biquad.py` |
| `min`/`max`, a delay line | `despike.py` |
| Int counters and timers, `and` conditions | `pedometer.py`, `taps.py` |
| quaternions, `dmp.quat`, cross products | `mahony.py`, `gyro_quat.py` |
| params with ranges, saturation | `pid_level.py` |
| `@dmp.sub`, `dmp.loop` with `break`, early `return` | `unit_vectors.py` |
| division: `x / y` (Q30, exact), `n // m` (Int, floor) | `slope.py`; `docs/language.md` §5 |
| the FSYNC input: `s.fsync` counts edges on the sync line | `language.md` §3 |

**Loops.**
- `for i in range(N)` unrolls, so `i` is a constant and can index lists.
- `for _ in dmp.loop(N):` with `break` is a real loop on the DMP, bounded by N trips.
- `while cond:` is a real loop with no bound: make sure it ends.
- Every loop's trips count against the pass budget.

**Subs.** `@dmp.sub` marks a helper that the ICM-20948 compiles once and calls (it has call and
return); the MPU inlines it. Plain helper `def`s are always inlined.

## 8. Writing good DMP code

- **Keep signals within ±1.** That leaves headroom for sums, and the checker stays quiet.
- **Write sums of products as sums of products.** `K*a + (1-K)*b` is one exact MAC run.
- **Prefer weighted forms for filters.** Write `(1-a)*y + a*x`, not `y + a*(x - y)`, whose
  difference can reach 4 and wrap before it is scaled.
- **Clamp symmetrically when you can.** `dmp.clamp(x, -c, c)` is a single instruction.
- **Divide by constants when you can.** `x / C` is one multiply; `x / y` is exact but costs
  about 1.3 ms on the MPU.
- **Declare a range on every state that feeds back into itself**, and clamp it where a real input
  can push it out.
- **Emit at the end of the pass, unconditionally**: packets have a fixed length.
- **Use Int only for counters and flags, and Bits only for masks.**
- **Read the listing.** Pointer loads and branches show where the bytes go, and the footer and
  the emulator say whether the pass fits.

When the front end refuses something, the error points at it and suggests the DMP way
([language.md §10](language.md#10-refused-with-the-hint-the-compiler-gives)). `python -m opendmp
ir file.py` prints what the front end made of a program, which helps when a result surprises you.
