# The opendmp language

A DMP program is a Python function that runs once per sensor sample on the motion processor
inside an MPU-6050 or ICM-20948. It is written in a subset of Python, with one import. The
compiler reads its source (the function is never run by Python) and checks it. It then builds it
for either chip, and simulates it three ways that must agree bit for bit.

```python
import opendmp as dmp

@dmp.program(rate=200)
def tilt(s):
    angle: dmp.state(-1.5, 1.5) = 0.0
    limit: dmp.param = 0.25
    angle = dmp.clamp(0.98 * (angle + s.gyro.y * 0.0873) + 0.02 * s.accel.x, -1.5, 1.5)
    dmp.emit(angle, s.accel.z)
    if angle > limit:
        dmp.irq()
```

This page is the reference. [tutorial.md](tutorial.md) is the walk-through; `PLAN.md` §4 and §6
record the reasons behind the rules.

## 1. Programs

```python
@dmp.program(rate=200, gyro_fs=2000, accel_fs=2, name=None)
def name(s): ...
```

| argument | meaning |
|---|---|
| `rate` | samples per second. The function runs once per sample; the host runtime sets the sample rate (MPU: 1 kHz / (1 + div); ICM: 1.1 kHz / (1 + div)) |
| `gyro_fs` | gyro full scale, deg/s: 250, 500, 1000, 2000 |
| `accel_fs` | accelerometer full scale, g: 2, 4, 8, 16 |
| `name` | the program's name (default: the function's) |

The function takes one argument, the sample. Its body runs from the top on every pass, and plain
locals are per-pass temporaries. From Python, a program has:
- `.check()`: range warnings;
- `.simulate(samples, engine=..., params=...)`;
- `.build(chip, trace=False)`;
- `.ir`: the lowered form.

## 2. Values and types

Every value is one 32-bit word.

| type | literal | meaning |
|---|---|---|
| **Q30** | `0.0`, `0.25`, `-1.5` | fixed point, value = word / 2^30, range **[−2, 2)**, resolution 2^−30 |
| **Int** | `0`, `3`, `-1` | a raw 32-bit integer: counters and flags |
| **Bits** | `dmp.bits(0xffff0000)` | a 32-bit mask: `&`, `^`, `\|`, `~`, `<<`, `>>` |

**The literal picks the type, and a variable keeps it.** `x = 0.0` makes a Q30 variable, and
`x = 1` into it later is an error (write `1.0`). Int arithmetic wraps at 32 bits.

**Why two types, when a Q30 is just an integer scaled by 2⁻³⁰?** For most instructions they are
the same thing: ADD, RSUB, INC, compares, CLAMP, AND/XOR and moves give identical bits either
way, and so does scaling by a whole number (`7 * x` is a sum of copies). The multiply is the
exception. The DMP's MUL and MAC keep the top of the 64-bit product, (a·b) >> 30, which is the
Q30 product, and they throw an integer product away (3 × 5 → 0). So `a * b` needs the type to
mean one thing: on Q30 values it is one MUL, and on two Ints it is the integer product: on the
ICM a MUL then the shifter's SHL 30, which recovers the product's low word; on the MPU
`dmp.imul`, built from adds.
The type also fixes what a literal is: `3` is an Int, while `3.0` does not fit in Q30.

**Reinterpreting a word.** `dmp.raw(x)` reads a Q30 value's word as an Int, and `dmp.q30(n)` reads
an Int's word as Q30. The bits do not change and no instruction runs. Use them for bit-level work:
`up + dmp.q30(1)` adds exactly one LSB (2⁻³⁰), and `dmp.raw(s.accel.x)` is the sample as a raw
count (the 16-bit reading << 16). On constants they convert: `dmp.raw(0.5)` is 2²⁹, and
`dmp.q30(1)` is 2⁻³⁰.

**Constants.** These are folded in Python, exactly as written, then rounded to the nearest Q30
when they meet a DMP value:
- literals and module-level names (any Python, including `math`);
- `range()` indices;
- helper arguments that are constants;
- expressions of these.

`GYRO_DT = math.radians(1000) / RATE` is fine even though its factors lie outside [−2, 2); only
the value that reaches the DMP must fit. **Assigning to a local always makes a DMP value**: after
`h = 0.75`, `h` is a word in RAM.

## 3. The sample

| name | value | notes |
|---|---|---|
| `s.accel.x`, `.y`, `.z` | raw / 16384 | g at ±2 g |
| `s.gyro.x`, `.y`, `.z` | raw / 16384 | 1000 deg/s per unit at ±2000 deg/s |
| `s.temp` | raw / 16384 | the temperature sensor's raw count |
| `s.accel[i]`, `s.gyro[i]` | the same | the triples index like tuples, with constant indices |
| `s.fsync` | an Int, 0 or 1 | the FSYNC pin's level this pass |

Each sensor is the 16-bit sample moved into the high half of a word. So it is Q30, and emitting
it unchanged costs 2 bytes of packet.

**`s.fsync`.** When a program reads it, the runtime sets `EXT_SYNC_SET = 1`. The hardware then
puts the FSYNC level into bit 0 of the temperature sample, so `s.temp`'s lowest bit is the FSYNC
level in that program, on every engine. It is measured on both parts' sample pages. The ICM rig's
FSYNC is wired to the bridge (`Y 0|1`); the MPU rig's is not.

## 4. Declarations

```python
x: dmp.state = 0.0                 # kept across passes; the initialiser is the load-time value
x: dmp.state(-1.5, 1.5) = 0.0      # with an assumed range
k: dmp.param = 0.25                # a state the host may write while running
k: dmp.param(-0.3, 0.3) = 0.0      # the host runtime refuses writes outside this range
h: dmp.state = [0.0] * 4           # lists, nested lists: one word per element
```

- **Placement.** Declarations go at the top of the body, before other statements.
- **Assignment.** A param cannot be assigned by the program: copy it into a local.
- **Ranges are assumptions, not code.** They tell the range checker what holds, and the
  simulator verifies them on every pass. No instruction enforces one: `dmp.clamp` does that.

## 5. Expressions

| construct | notes |
|---|---|
| `a + b`, `a - b`, `-a` | see the exact-sum rule below |
| `a * b` | Q30 × Q30: floored. Int × Int: the exact product modulo 2³²: on the ICM two instructions (MUL, then SHL 30); on the MPU `dmp.imul(n, m)` (shift-and-add, 32 steps; up to about 1.0 ms). A constant integer factor scales either type exactly, by doublings |
| `dmp.raw(x)`, `dmp.q30(n)` | the same word read as an Int or as Q30: no instruction (above) |
| `x / y` | Q30 values: truncated toward zero, exact to the last bit, saturating at ±(2 − 2⁻³⁰) when the quotient reaches 2 or y is 0. It compiles to `dmp.div(x, y)` (below), which is also callable by that name |
| `x / C`, `x * (1 / C)` | C a constant: a multiply by the reciprocal, rounded to Q30 (one MUL, not exact division) |
| `n // m` | Int values: floor division exactly as Python's, for every 32-bit pair. `n // 0` and the one overflow, −2³¹ // −1, saturate at the Int limits. It compiles to `dmp.idiv(n, m)` (32 steps; about 2.7 ms on the MPU, 0.9 ms on the ICM) |
| `x * 2**k` | doublings |
| `x << k` on Bits, k 0–31 | the word shifted left, in place: doublings (on the MPU a lane move does the first 16; on the ICM one MUL and SHL 30 for k ≥ 2) |
| `x >> k` on Bits, k 0–31 | arithmetic shift right: a MUL by 2⁻ᵏ (k = 31: two MULs) |
| `abs(x)` | `abs(-2.0)` wraps to −2.0 |
| `min(a, b)`, `max(a, b)` | exact |
| `dmp.clamp(x, lo, hi)` | lo and hi constants; exact |
| `a if cond else b` | a select |
| `a & b`, `a ^ b`, `a \| b`, `~a` | on Bits |
| `dmp.assume(x, lo, hi)` | the value x, with a range the checker may use (verified in simulation) |

**The exact-sum rule.** A `+`/`-` expression is **one exact sum**. Its terms can be:
- values;
- integer multiples (`2*x`);
- two-factor products (`a*b`, `K*x`, `x/C`).

They are added in full precision (products in Q60), floored once and wrapped once. That is
exactly what one multiply-accumulate run computes on the DMP, so `K*a + (1-K)*b` has one meaning
on every engine. An intermediate that would pass ±2 does no harm if the result fits, and
parentheses inside a sum change nothing. A product's factors are values: an inner product or sum
is floored and wrapped first.

**Where values wrap.** A sum wraps at its end, and so does `abs(-2.0)`; nothing else can.
Compares, `min`, `max` and `clamp` are exact. `opendmp check` reports every place a value can
wrap, with the interval, and names the fix: a declared range, scaling down, or `dmp.assume`.

## 6. Conditions and control flow

| construct | notes |
|---|---|
| `<` `<=` `>` `>=` `==` `!=`, `and`, `or`, `not` | exact. `and`/`or` evaluate every operand unless a constant decides |
| `if` / `elif` / `else` | nested freely |
| `for i in range(N)` | N constant; unrolled, so `i` is a constant (list indices, constants) |
| `for _ in dmp.loop(N):` ... `break` | a bounded loop: a real backward branch on both chips. The loop variable has no value |
| `while cond:` | a real loop on both chips. Its pass time is unbounded: make sure it ends within the sample period |
| `break` | out of the innermost loop |
| `return` | in the program: ends the pass; in a helper: its value(s) |

A comparison is not a value: write `1 if a > b else 0`. On the DMP a comparison is a subtraction
whose sign is tested. Compares against −2.0 are handled correctly: range analysis inserts a guard
where a subtrahend can be exactly −2.0 (ISA2 Finding 40).

## 7. Output

```python
dmp.emit(angle, s.accel.z)        # fields named angle, accel_z
dmp.emit(x=a, y=b)                # named expressions
dmp.irq()                         # raise the DMP interrupt this pass
```

- **The packet.** The emits, in program order, make one packet per pass, and its layout is fixed
  at compile time: a Q30 or Int is 4 bytes, and a raw sensor emitted unchanged is its int16,
  2 bytes. A field name may appear once.
- **Placement.** Emit unconditionally: a packet has a fixed length. `dmp.irq()` may be conditional.
- **Decoding.** `build.schema()` (and the `.json` a build writes) describe the packet, and
  `opendmp.sim.decode` turns bytes into named floats.

## 8. Lists

Lists exist only at compile time. Every index must be a constant (a literal, a `range()` index,
or a constant expression), so each element is its own word. The DMP has no indexed addressing.
- **Sharing.** A named list (`row = z[1]`, or a helper argument) is shared: writing `row[0]`
  writes `z[1][0]`.
- **Copying.** A list display (`[a, b]`) or a helper's returned list is a copy.
- **Destructuring.** `w, x, y, z = q` unpacks.

## 9. Helpers and subs

```python
def biquad(x, z):                 # a helper: inlined at every call
    y = B0 * x + B1 * z[0] - A1 * z[2]
    z[1] = z[0]
    z[0] = x
    return y

@dmp.sub                          # ICM: one routine, called; MPU: inlined
def unit(x, y, z):
    ...
    return gx, gy, gz
```

- **Helpers.** A helper `def` at module level is inlined. It may take lists and update them, and
  return one or several values.
- **Subs.** A `@dmp.sub` takes values (not lists) and returns values. On the ICM-20948, a sub
  called more than once is compiled **once**, as a routine entered by `call` and left by `ret`.
  Its arguments and results live in its own fixed words; the DMP has no stack for data. Routines
  nest at most 3 deep below the program, because the hardware return stack is 4 entries and
  circular. Deeper calls are inlined. On the MPU, which has no call, subs are inlined.
- **No recursion**, and no nested `def`, lambdas or default arguments.

`dmp.quat` has quaternion helpers written in this language: `gravity(q)`, `rate_step(q, h)` and
`normalize(q)`.

`x / y`, that is `dmp.div(x, y)`, is written in this language too (`opendmp/lib.py`). It is restoring long
division: the integer bit, then 30 fraction bits, one compare and subtract each. It works on
−|x| and −|y|, which always fit (|−2.0| would not), so it is exact for every input. Because it is
compiled from its source, every engine, silicon included, computes the same bits. It is a
`@dmp.sub`: one routine on the ICM when a program calls it twice or more, and inlined otherwise.
It costs about 10,700 cycles on the MPU (1.3 ms) and 8,700 on the ICM (0.4 ms) at most.

## 10. Refused, with the hint the compiler gives

| construct | hint |
|---|---|
| `n / m` on Int values, `x // y` on Q30 values | `/` divides Q30 values, `//` divides Int values |
| an Int times a Q30 value | `n * m` multiplies two Ints and `x * y` two Q30 values; `dmp.q30(n)` reads an Int's word as Q30 |
| `%` | `n - (n // m) * m` |
| `math.sqrt(x)` of a DMP value, `x ** 2` | a Newton step; `x * x` |
| a list index that is not a constant | unroll with `range(N)`; there is no indirect addressing |
| a helper that calls itself | helpers are inlined |
| `dmp.emit` under a condition | packets have a fixed length |
| `<<` / `>>` / `&` on a Q30 | `x * 4`; masks are `dmp.bits(0x...)` |
| assigning a param | copy it into a local |
| mixing an Int and a Q30 | use a float literal for Q30 |
| a constant outside [−2, 2) | scale it (`x * 2.5` → `2 * x + 0.5 * x`) |
| `try`, `continue`, `lambda`, slices, `assert`, `print`, strings, `True`/`False` as values | `if`/`else`; `dmp.assume` for asserts; `1`/`0` for flags |
| a comparison used as a value | `1 if a > b else 0` |
| a declaration after other statements, or without an initial value | declare `dmp.state`/`dmp.param` at the top, with a value |

An error points at the construct, with the line, a caret under it, and the hint:

```
tilt.py:14:17: error: division by a variable: the DMP has no divide
        angle = x / y
                ^^^^^
hint: multiply by a constant reciprocal (x * (1 / C)), or use a Newton step (examples/unit_vectors.py)
```

## 11. What the compiler tells you

`opendmp build` lists, for every construct, the method it chose, with the line (`--quiet` turns
it off). The listing repeats each under its source line:

```
tilt for the mpu6050: 45 B of code, packet 6 B, at most 82.0 us per pass (1.6 % of the period)
  tilt.py:33  `dmp.clamp(K * (angle + s.gyro.y * GYRO_DT) + ...`: one CLAMP instruction (f5 op 0): x clamped to +-1.5, exact for every value (Finding 41)
  tilt.py:33  `K * (angle + s.gyro.y * GYRO_DT) + (1 - K) * ...`: one multiply-accumulate run of 2 terms (a plain value enters as x * 1.0), summed exactly (Q60) and floored once
  tilt.py:33  `angle + s.gyro.y * GYRO_DT`: one multiply-accumulate run of 2 terms (a plain value enters as x * 1.0), summed exactly (Q60) and floored once
  tilt.py:34  `s.accel.z`: the sample's int16, 2 B straight from the sample page (a MOV under a 16-bit lane mode)
  tilt.py:35  `angle > limit`: RSUB into a scratch word, then a branch on its sign (the f5 compare only tests signs)
```

These are the kinds of note:
- **Front end:** a multiply by a reciprocal, `range()` unrolled, a helper inlined.
- **Selection:** one ADD/RSUB/MUL/INC, an ADD chain, a MAC run, CLAMP forms and compare-and-skip,
  compares (and whether they need the −2.0 guard), loops with their trip counts, subs as routines
  or inlined, emits, FSYNC.
- **Assembly:** branches that need hops.

`Build.notes` has them as `(location, text)`, and `Build.explain()` as lines. A library function
is summarised at its call site, with the most cycles one call adds (`x / y (dmp.div): exact
restoring division ...; costs up to 10441 cycles = 1273 us each` on the MPU), not line by line.

## 12. How it compiles

| construct | both chips |
|---|---|
| a `+`/`-` sum with products or 3+ terms | one multiply-accumulate run: exact, floored once |
| `a + b`, `a - b`, `a * b` | one ADD, RSUB, MUL |
| `dmp.clamp(x, -c, c)` | **one** CLAMP instruction (`f5` op 0, ISA2 Finding 41) |
| `dmp.clamp(x, lo, hi)` | CLAMP about the midpoint, when `x - (lo+hi)/2` cannot wrap; else two compare-and-skips |
| `min(x, c)`, `max(x, c)` | one CLAMP where x's range allows, else a compare-and-skip |
| `a < b` | RSUB into a scratch word, then a sign test (with a −2.0 guard where needed) |
| `if` | a branch over the body; far targets through hops |
| `dmp.loop`, `while` | a counter or a condition, and a branch back |
| `n += 1` on an Int | one INC |
| a sum with no products (`a + b - c`, `r + r - b`) | a chain of ADD/RSUB: the same value modulo 2³² |
| constant integer coefficients (`n * 100`, `7 * a - 3 * k`) | doublings, by Horner's rule on the coefficients' binary or signed digits (`n * 100`, 100 = 64 + 32 + 4: 8 ops, not a 100-term run); with products in the sum, the scaled plain part enters the multiply-accumulate run as one term |
| `x / y` (`dmp.div`), `n // m` (`dmp.idiv`), `n * m` on Ints (`dmp.imul` on the MPU) | the library routines (above) |
| `dmp.raw(x)`, `dmp.q30(n)` | nothing: the same word |

**The pass-time bound.** A build computes an upper bound on a pass
(`build.cycles_max()`, the listing's last line and the `.json`):
- it is the longest path through the code as emitted, a far branch costing its hops;
- each loop counts at its trip count (`dmp.loop(N)`), and each call at the routine's bound;
- it is exact for straight-line code, and holds on every emulated pass;
- a `while` loop leaves it open, and the listing says so.

A bound over the sample period is flagged: the MPU would then drop to a lower rate.

The MPU's branch reaches 32 bytes each way. The assembler chains longer branches through
one-byte hops (ISA2 Finding 42), as the vendor's own compiler did. The layout of words in RAM is
searched to minimise pointer loads, scored by the scheduler's actual output. Both chips' output
runs bit-exact on silicon (`opendmp run --verify`).
