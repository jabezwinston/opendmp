# opendmp: plan of record

A Python-like language and compiler for the InvenSense DMP. You write one Python function, and
opendmp turns it into a DMP program for the **MPU-6050** and the **ICM-20948**. It also
produces an annotated listing, a cycle-accurate emulator run and a host runtime that loads the
program onto the bench rigs and decodes its FIFO packets by name.

The ISA knowledge comes from `~/Git_repos/InvensenseDMP`: the programmer's manual in
`docs/manual/`, the evidence log in `ISA2.md` (Findings 37–54 were measured for this
project) and the bench in `bench/`. This repo holds only original code. **No vendor firmware
is shipped or needed** (see B1 below).

**Status, 2026-10-08.**
- **Done:**
  - the plan and 12 example programs in `examples/`;
  - the **emulator**, `opendmp/emu/`: cycle-accurate, fed from files, and validated against
    both rigs cell by cell (§8);
  - **Phase 0**:
    - the ISA tables, encoder, decoder, assembler and `dis` for both chips (`opendmp/isa/`);
    - `facts.py`;
    - **B1 and B2 measured**. A program runs with **no vendor image** on both parts, and the
      clobber map is known. On the ICM the FIFO is DMP RAM 0x3000–0x3fff.
  - **Phase 2, the front end** (`opendmp/lang/`, `opendmp/ir/`):
    - the Python subset, lowered to a typed IR, with located errors that give a hint;
    - two executable semantics that agree bit for bit on all 12 examples: the reference
      (evaluating the source) and the IR interpreter;
    - range analysis. All 12 examples are proven free of wraps.
    - The CLI is `python -m opendmp check | sim | ir`.
  - **Phase 3, the backends** (`opendmp/backend/`): all 12 examples build for both chips, and
    their bytes in the emulator equal the reference on every pass, 10,000 random ones included.
  - **Phase 4, the runtime and silicon** (`opendmp/host/`): vendor-free load and run, and trace
    builds replayed in the emulator. **All 12 examples are bit-exact on both rigs.** The first
    silicon runs found four ISA behaviours that the manual and the emulator had wrong (ISA2.md
    Finding 40, §7); they are measured, modelled and compiled around.
  - **Phase 5, depth and polish**:
    - **smaller, faster code**, 3,134 B for the twelve against 5,099 before (−39 %), with 22 %
      fewer worst-case cycles, still bit-exact on both rigs. It comes from CLAMP selection,
      copy propagation and dead-store elimination, a layout search scored by the scheduler,
      loop-head state and one-byte branches;
    - **ICM calls** for `@dmp.sub`, and `while` on both chips;
    - **the runtime**: live param writes (declared ranges enforced) and state watching;
    - **docs**: [docs/tutorial.md](docs/tutorial.md) and [docs/language.md](docs/language.md).

    Two ISA findings came out of it, both measured on the rigs: `f5` op 0 is a symmetric clamp,
    not MIN (Finding 41), and the MPU's mode-4 low bytes are one-byte branches (Finding 42).
  - **Since:** `s.fsync` compiled on both chips (ICM: FSYNC toggled on the rig, 6 edges,
    bit-exact); a pass-time bound with loops at their trip counts and calls at their routines'
    bounds (`build.cycles_max()`: exact for straight-line code, within 10 % of unit_vectors'
    worst inputs, open for `while`); `run --watch-output` CSV; `dmp.div` (exact restoring
    division for every input, bit-exact on both rigs); a per-construct method report in
    `opendmp build` and the listing; ADD/RSUB chains for sums without products; hops at dead
    points.
  - **2026-10-08**, after InvensenseDMP Findings 44–54 (§13 item 8):
    - `n * m` on Ints is MUL then SHL 30 on the ICM (two instructions, not 32 steps);
      `x >> 31` compiles (it crashed), and `x << k` works in place;
    - `==` is RSUB then `test` for every pair (no XOR fallback, no −2.0 guard);
    - the MPU's INC/DEC write R, so a MAC run after one into the same word starts from a move;
    - the emulator rewrites the sample page under a long pass and caps the FIFO, keeping the
      newest bytes; `Model` checks its values;
    - `Rig.stop` pulses DMP_RST (clearing DMP_EN does not stop a pass);
    - `opendmp run prog.asm` runs hand-written assembly on a rig (`--verify` replays it), and
      `python -m opendmp.emu --sample` runs it without a sensor file.

## 1. Decisions (2026-09-23)

| question | decision |
|---|---|
| surface style | **A Python subset, parsed with `ast`.** One decorated function, one import (`import opendmp as dmp`) |
| targets | **MPU-6050 and ICM-20948 from day one**: one IR, two backends, and both rigs in every milestone |
| deployment | **Whole program only.** opendmp owns the entry point and the data region. Extending the stock firmware and patching vendor slots are deferred |
| numbers | **Q30 with range checks.** Every value is a Q30 word in [-2, 2) that wraps like the chip does. The compiler warns where a wrap reaches something that cannot tolerate it (§6.5). `Int` covers counters and `Bits` covers masks |
| outputs | Every build prints an **annotated listing** (`.lst`) and writes an **annotated assembly** file (`.asm`) next to the binary |
| emulator | **Cycle-accurate, fed from files** (sensor CSV, program/RAM images, a TOML run config), **built first**, because the compiler's tests need it. See `README.md` |
| unknowns | **Measured on the rigs, not guessed.** Every ISA behaviour the tools rely on is either hardware-confirmed or a named, switchable assumption |
| location | `~/Git_repos/opendmp`, separate from the RE repo |

## 2. What a program looks like

```python
import opendmp as dmp

K = 0.98                    # plain Python constants, folded at compile time
GYRO_DT = 0.0873            # gyro unit -> rad, times 5 ms

@dmp.program(rate=200)      # runs once per sample, 200 Hz
def tilt(s):                # s = this pass's sensor sample
    angle: dmp.state = 0.0  # kept across passes
    limit: dmp.param = 0.25 # the host may change it while running
    angle = K * (angle + s.gyro.y * GYRO_DT) + (1 - K) * s.accel.x
    dmp.emit(angle, s.accel.z)
    if angle > limit:
        dmp.irq()
```

```
$ opendmp build tilt.py --chip mpu6050
$ opendmp build tilt.py --chip icm20948
$ opendmp run   tilt.py --port /dev/ttyUSB0          # chip auto-detected from the bridge
$ opendmp sim   tilt.py --sensors walk.csv           # the cycle-accurate emulator
```

### Style rules (the language)

- **One import.** Everything is `dmp.<name>`: `dmp.program`, `dmp.state`, `dmp.param`,
  `dmp.emit`, `dmp.irq`, `dmp.clamp`, `dmp.assume`, `dmp.loop`, `dmp.sub`, `dmp.quat` and the
  small standard library. There are no star imports.
- **One function is one pass.** The body runs from the top on every sample interrupt. Plain
  locals are per-pass temporaries.
- **Declarations are annotated assignments.** `x: dmp.state = 0.0` persists across passes and
  its initialiser is the load-time value. `x: dmp.param = 0.25` is a state the host writes.
  Lists work too (`hist: dmp.state = [0.0] * 4`), indexed by compile-time constants.
- **Ranges are assumptions, not code.** `x: dmp.state(-1.5, 1.5) = 0.0` (or
  `dmp.assume(expr, lo, hi)`) tells the range checker what holds. The simulator verifies it;
  no instructions are emitted. `dmp.clamp` is what enforces a range. A param's range is also
  enforced by the host runtime, which refuses writes outside it.
- **The literal picks the type.** `0.0` is Q30 and `0` is `Int` (a raw 32-bit counter, so
  `n += 1` is one `INC`). `dmp.bits(0xffff0000)` is `Bits` (`&`, `^`, shifts).
- **A `+`/`-` expression is exact.** It is summed in full precision (products included) and
  wraps once, at the end, so an intermediate that would pass 2 does no harm if the result
  fits (§4, "reference semantics").
- **Module-level names are compile-time constants.** They can be any Python, including `math`.
  Only the value that reaches the DMP has to lie in [-2, 2): `GYRO_DT = math.radians(1000) /
  RATE` is fine even though its factors are not.
- **The sensors are Q30.** `s.accel.x` is raw/16384, which is g at ±2 g full scale. `s.gyro.y`
  is raw/16384, which is 1000 dps per unit at ±2000 dps. `s.temp` is also available, and
  `s.fsync` on the ICM. The triples index like tuples (`s.gyro[i]`). The full-scale ranges are
  `@dmp.program` arguments, and the host runtime applies them.
- **Packets are named.** `dmp.emit(angle, s.accel.z)` gives the fields `angle` and `accel_z`,
  and `dmp.emit(tilt=a - b)` names an expression. The width comes from the type: Q30 is 4 B and
  a raw sensor is its int16, 2 B.
- **Loops.** `for i in range(N)` with a constant N always unrolls. `for _ in dmp.loop(N):` with
  `break` is a bounded loop and `while` an unbounded one: real backward branches on both chips.
- **Helpers.** A helper `def` is inlined and may update the state lists passed to it. On the
  ICM, a `@dmp.sub` called more than once is one routine, with a real call. Its arguments and
  results live in the sub's own fixed words and are copied in and out, because the DMP has no
  indirect addressing.

### Style suggestions (writing good DMP programs)

- Keep signals within ±1. That leaves headroom for sums, and the range checker stays quiet.
- Write sums of products as sums of products. `K*a + (1-K)*b` becomes a single
  multiply-accumulate run.
- Write filters as weighted sums, `(1-a)*y + a*x`, not `y + a*(x - y)`: the difference can
  reach 4 and wrap before it is scaled.
- Declare a range on every state that feeds back into itself, and clamp it if a real input can
  push it past that range (see `examples/tilt.py`).
- Emit at the end of the pass, unconditionally. v1 packets have a fixed length.
- Use `Int` only for counters and flags, and `Bits` only for masks.

The 12 programs in `examples/` (see `examples/README.md`) follow these rules and pass a range
check on synthetic motion.

## 3. What the compiler prints

`opendmp build` prints the listing and a summary, and writes `build/tilt.{asm,lst,bin,json}`.
The listing below is a **format mock-up** for the program above, hand-compiled by the rules in
§6 (`opendmp build --list` prints the real one, in the same columns). Its compare was a `cmp`
until Finding 40 showed that the `f5` compare tests signs only; it is now an RSUB. Its bytes were checked by assembling the `.asm` with opendmp's assembler and with
`InvensenseDMP/tools/dmpasm.py` (47 bytes, identical). They also **run in the emulator** as `examples/emu/tilt.toml`, where the
filter tracks the generated motion and the times below are its measured cycle counts.

```
; tilt.lst  opendmp 0.1  chip mpu6050 (8.199 MHz)  whole program  entry 0x0400
; addr  bytes     asm                effect                                   cycles    us
        ; tilt.py:9   angle = K * (angle + s.gyro.y * GYRO_DT) + (1 - K) * s.accel.x
0400    d8        always             predicate off                                 5  0.61
0401    f1        fmode 1            32-bit stores                                 5  0.61
0402    b0 80     p0 -> 0x000        P0 = samples {ax|ay, az|temp, gx|gy, gz|-}   10  1.22
0404    b9 a1     p2 -> 0x110        P2 = work.gy                                 10  1.22
0406    dc        store              route -> RAM                                  5  0.61
0407    de        clr                gy := 0                                      12  1.46
0408    df        clr.next           P2 = work.ax; ax := 0                        12  1.46
0409    f2        fmode 2            upper half-word lane                          5  0.61
040a    c0        mov 0              ax.hi := accel_x                             18  2.20
040b    a1        p2page 1           P2 = work.gy                                  5  0.61
040c    cc        movx 2             gy.hi := gyro_y (low half, swapped up)       18  2.20
040d    f1        fmode 1                                                          5  0.61
040e    b1 81     p0 -> 0x110        P0 = work {gy, ax, t, angle}                 10  1.22
0410    b5 90     p1 -> 0x100        P1 = const {GYRO_DT, 1.0, K, 0.02}           10  1.22
0412    d0        p2inc2             P2 = work.t                                   5  0.61
0413    02        mul 0,0            t = gy*GYRO_DT                               58  7.07
0414    f0        fmode 0            multiply-accumulate                           5  0.61
0415    68        mac 3,1            t = acc + angle*1.0                          60  7.32
0416    f1        fmode 1                                                          5  0.61
0417    56        mul 2,2,inc        P2 = work.angle; angle = t*K                 58  7.07
0418    f0        fmode 0            (the move restarted the accumulator)          5  0.61
0419    38        mac 1,3            angle = acc + ax*0.02                        60  7.32
        ; tilt.py:10  dmp.emit(angle, s.accel.z)
041a    dd        fifo               route -> FIFO                                 5  0.61
041b    f1        fmode 1                                                          5  0.61
041c    c6        mov 3              emit angle     packet[0:4]  q30              18  2.20
041d    b0 80     p0 -> 0x000        P0 = samples                                 10  1.22
041f    f2        fmode 2                                                          5  0.61
0420    c2        mov 1              emit accel_z   packet[4:6]  i16              18  2.20
        ; tilt.py:11  if angle > limit:
0421    b1 81     p0 -> 0x110        P0 = work                                    10  1.22
0423    92        p1page 2           P1 = param {limit}                            5  0.61
0424    dc        store              route -> RAM                                  5  0.61
0425    a1        p2page 1           P2 = work.gy, a scratch by now                5  0.61
0426    f1        fmode 1                                                          5  0.61
0427    61        rsub 3,0           gy := limit - angle; F := it < 0 (exact)     28  3.42
0428    da        if nf                                                            5  0.61
0429    f4 02     branch 0x042d      angle <= limit: skip the irq                 15  1.83
042b    d8        always             angle > limit: fall through                   5  0.61
        ; tilt.py:12  dmp.irq()
042c    fe        int                raise DMP_INT                                 5  0.61
042d    d8        always             join                                          5  0.61
042e    ff        end                end of pass                                   5  0.61

; data   0x100 const  GYRO_DT 059652bd  1.0 40000000  K 3eb851ec  0.02 0147ae14
;        0x110 work   gy  ax  t  angle(state, init 00000000)
;        0x120 param  limit 10000000
; packet 6 B   angle q30 [0:4]   accel_z i16 [4:6]    (generated parser: build/tilt.json)
; pass   47 B: 645 cycles = 78.7 us after the tick (650 when it interrupts), 1.6 % of 5000 us
```

The `.asm` has the same content in opendmp assembly (`opendmp.isa`), which is
`examples/emu/tilt.asm` here: `python -m opendmp asm tilt.asm` prints the plain form of this
listing, and the emulator runs the file directly. It carries the source line as a header
before each statement's instructions, a comment on every instruction naming the words the
pointers select, and the data (`.dw`), so it can be read, edited and reassembled by hand.
`examples/emu/tilt.s` is the same program in InvensenseDMP's `dmpasm` syntax (identical bytes).

## 4. The language, v1

| construct | MPU-6050 | ICM-20948 |
|---|---|---|
| `+ - *`, unary `-`, `abs`, `min`, `max`, `dmp.clamp` | yes | yes |
| `x * CONST`, `x / CONST` (a multiply by the reciprocal), `x * 2**k` (adds, a multiply, or the ICM shifter) | yes | yes |
| comparisons, `and`/`or`/`not`, `a if c else b` | yes, never overflow (§6.4) | yes |
| `if`/`elif`/`else`, nested | yes (branch-skip, hops past ±32 B) | yes (jumps) |
| `for i in range(N)`, N constant | unrolled | unrolled |
| `for _ in dmp.loop(N)` + `break` | a real loop | a real loop |
| `while` | a real loop (unbounded pass time) | a real loop (unbounded pass time) |
| helper `def` | inlined | inlined, or `@dmp.sub` = call (depth ≤ 4, no recursion) |
| state lists, constant indices, tuple unpacking, several return values | yes | yes |
| `Int` `+= 1`, `-= 1`, compare with 0..63 | `INC`/`DEC`, compare via ALU | `INC`/`DEC`, `fc cK Y` immediate compare |
| `Bits` `&`, `^`, `<<`, `>>` | yes | yes |
| `dmp.emit`, `dmp.irq`, `return` (ends the pass), `dmp.assume` | yes | yes |
| divide by a variable, `sqrt`, `**` (other than `x*x`), runtime list index, recursion, emit under a condition, `<<` on Q30 | **error**, with a hint (`dmp.rsqrt` does a Newton step; tables need constant indices) | same |

**Reference semantics** (settled in Phase 2; `opendmp/lang/reference.py`). `tilt.simulate(samples)`
evaluates the source directly over the Q30 model. The rules, which the compiled code must match
bit for bit:

- **A `+`/`-` expression is one exact Sum.** Its terms can be values, integer multiples
  (`2*x`) and two-factor products (`a*b`, `K*x`, `x/C`). They are added in full precision
  (products in Q60), floored once and wrapped once. That is what one multiply-accumulate run
  computes, so `K*a + (1-K)*b` has one exact meaning, and parentheses inside a sum change
  nothing. A product's factors are values: an inner product or sum is floored and wrapped
  first. A minus sign on a factor moves to the term. A lone product is a one-term sum.
- **Constants** are literals, module-level names, `range()` indices, helper arguments that are
  constants, and expressions of these. They are folded in Python as written, then rounded to
  the nearest Q30 when they meet a DMP value; outside [-2, 2) is an error. **Assigning to a
  local always makes a DMP value** (`h = 0.75` is a word, not a constant).
- **The literal picks the type**: `0.0` is Q30, `0` is Int, `dmp.bits()` is Bits. A variable
  keeps its type, so `x = 1` into a Q30 variable is an error (write `1.0`).
- Compares, `min`, `max` and `clamp` are exact. `abs(-2.0)` wraps to -2.0. Int arithmetic wraps
  at 32 bits.
- **Lists** exist only at compile time: every index is a constant, so each element is its own
  word. A named list (`y = z[0]`, or a helper argument) is shared; a list display or a
  helper's result is a copy.
- `and`/`or` evaluate every operand (a helper call in one runs regardless), unless a constant
  operand decides the result.

The reference shares only the reading of the source with the compiler: finding the function,
what a module-level name means, and constant folding (`lang/common.py`). The arithmetic,
flattening, control flow, inlining and lists are implemented twice, once in the reference and
once in lowering plus `ir/interp.py`. The two agree bit for bit on every example. A deliberately
broken IR interpreter (a floor per product) is caught on every example tried. The emulator
running the compiled bytes will be the third oracle.

## 5. Architecture

```
tilt.py ─ast─> front end ──> IR (typed values, storage classes, structured control)
   │             │               │  fold · unroll · inline · MAC fusion · CSE · range analysis
   │             │               v
   │             │          backend[mpu6050 | icm20948]
   │             │            layout (pages) → select → schedule machine state → control flow
   │             │               v
   │             │          bytes + data image + symbols + packet schema + cycle count
   │             │            ├─ .lst .asm .bin .json (bundle)
   │             v            v
   └──> reference model   emulator (both chips)   ←→   rig (/dev/ttyUSB0, /dev/ttyUSB1)
        (AST rewrite)     host.Rig / host.verify: python -m opendmp run [--verify]
```

```
opendmp/
  __init__.py      the whole user API, re-exported from api.py              [done]
  api.py           program, state, param, emit, irq, clamp, assume, loop, sub, bits; Program
                     (.ir, .check(), .simulate(samples, engine, params))
  quat.py          dmp.quat: gravity, rate_step, normalize, written in the language
  sim.py           SimResult: packets, interrupts, decoded fields, actual wraps, failed assumptions
  q30.py           the number model                                        [done]
  emu/             the cycle-accurate emulator, both chips                 [done, §8]
    machine.py       state, step, pass; the rules a result rests on as Model switches (measured defaults)
    chips.py         memory geometry, clocks, the measured cycle table, sample layouts
    sched.py         the pass timeline: ticks, rate drop (MPU) / 1/pass (ICM), mid-pass samples, stalls
    files.py         sensor CSV (raw or SI units) or one fixed sample, program/RAM images, pokes
    gen.py           synthetic sensor data (rest, tilt, rotate, walk, taps, vibration, ...)
    disasm.py        mode-aware disassembly for traces
    __main__.py      python -m opendmp.emu [run.toml] [options]
  isa/             both ISAs as tables                                     [done]
    core.py          Insn, decode/encode (inverses, checked on 13 vendor images), cost keys
    asm.py           opendmp assembly: .chip/.entry/.org/.dw/labels, mode tracking, disassembly
    facts.py         every machine behaviour relied on, with status and evidence (§7)
  lang/            the front end                                           [done]
    errors.py        DmpError / warnings: file:line:col, the line, a caret, a hint
    source.py        a function's AST and the names it sees (module globals, closures)
    common.py        what names mean, constant folding (shared by the two semantics)
    lower.py         AST -> IR: types, folding, exact Sums, inlining, @dmp.sub, lists, control flow
    reference.py     the reference semantics: evaluates the source; counts wraps, failed assumes
  ir/              the IR                                                  [done]
    nodes.py         Var (state/param/temp/arg), Const Load Sensor Sum(Term, Prod) MinMax Abs
                     Select BitOp Shift Assume; Cmp BoolOp Not; Assign If Block Loop Break Emit
                     Irq Check CallSub; SubIR, ProgramIR (packet schema)
    interp.py        the IR, executed
    ranges.py        interval analysis: where a value can wrap, with the fix; per-node intervals
    show.py          the IR as text
  backend/         layout, select, sched, flow, emit (asm/lst/bin/json)
  host/            bridge.py (the ESP32 protocol, including W), rig.py, verify.py
  cli.py           python -m opendmp check | sim | ir | build | run | dis | asm | facts   [done]
examples/          13 programs (examples/README.md); examples/emu/ = a runnable emulator demo
tests/             hardware-transcribed ISA vectors, cycle counts, scheduling, every example
                   against the reference on both chips, the host runtime, the CLI
```

Implementation style: the core uses only the standard library (`ast`, `dataclasses`,
`tomllib`), with `pyserial` as an optional `[bench]` extra. Python 3.11+, type hints, pytest
and ruff. ISA encodings are tables, not `if` chains. Every rule the code relies on cites its
Finding or probe, as the RE repo does. The tools **refuse rather than guess**: the emulator
raises `Unmodelled` for anything with no measured meaning.

## 6. The hard parts

### 6.1 Chip profiles (measured: ISA2.md Findings 37–54)

| | mpu6050 (mantis) | icm20948 (DMP3) |
|---|---|---|
| data | 0x000–0x3FF, one block; never emit `bc`–`be` (the block persists across passes) | 0x0000–0x0FFF, blocks via `bc`–`be` (reset every pass) |
| code | entry 0x0400, up to 0x0BFF (3 KB of memory in all) | entry 0x1000, **up to 0x2FFF** (8 KB): 0x3000–0x3FFF is the FIFO |
| hardware-written RAM | **0x000–0x00D only**: ax ay az temp gx gy gz as int16 BE, every pass, only while the DMP runs (B2). 0x00E–0x00F are free | **0x0000–0x0011**: 0–3 cleared, then ax ay az gx gy gz temp (FSYNC is temp bit 0); **plus every FIFO byte, into 0x3000–0x3FFF** (B2) |
| boot | **no vendor image** (B1): device reset (it keeps DMP memory), minimal registers, zero all memory, load, `DMP_CFG_1/2` = entry | the same (B1), but the device reset writes 0x0080–0x0087; HW_FIX_DISABLE / SINGLE_FIFO_PRIORITY_SEL not needed |
| flag | **one flag F on both chips**; `d1` makes it R's word == 0, R being the last ADD/RSUB/MUL/NMUL/MAC/INC/DEC result or CLAMP's compare (Finding 45); `d9` IF F, `da` IF ¬F; **no Z**. ADD/RSUB/MUL set the exact sign (RSUB: unless A = −2.0); **`f5` op 1 only tests signs** (Finding 40) | same |
| control flow | **signed 6-bit branch: −32..+31 B** (`f4 41`–`7f` is the interrupt); **every `f4` leaves mode 4**, where a byte below 0x40 is a **one-byte branch** (Findings 40, 42); no absolute jump | 16-bit jump and call (4 deep), `e0`, loops; mode-4 hygiene |
| multiply-accumulate | `f0`; R is the 64-bit accumulator; **a P2 move zeroes R, and that is the whole restart**: a MAC after INC/DEC/CLAMP continues from their result (Findings 44, 54) | `f0` too; **only a MAC's op&4 pre-advance restarts it**; MUL then SHL 30 is the integer product (Finding 46) |
| `f2`/`f3` | **16-bit lanes, not just store widths**: `f3` computes on sign-extended low halves (a MUL there is tiny), `f2` on high halves; the same for `f6`/`f7` | same |
| `f5` op 0 | **a symmetric clamp**: A ≥ 0 ? min(A, B) : max(A, −B); F := clamped (Finding 41) | same |
| interrupt | `fe` | `e2` |
| clock, cycles (cheap / MOV / ADD / MUL / MAC) | 8.199 MHz: 5 / 18 / 28 / 58 / 60 | 20.975 MHz: 4 / 17 / 26 / 25 / 27 |
| pass start after the tick | 110 cycles (13.4 µs) | 130 cycles (6.2 µs) |
| a sample write inside a pass | stalls it 14–28 cycles; the pass reads the new sample (Finding 47) | the pass reads the new sample (Finding 47) |
| FIFO overflow | keeps the newest 1 KB (Finding 52) | a 4 KB ring at 0x3000, keeps the newest (Finding 52) |
| prologue | `d8 f1` | `f1` first (mode 4 survives across passes), then `d8` |
| state kept across passes | pointers, P2 index, route, mode, F, R; **not** the `fc`/`fd` exchange | route, mode (mode 4 too), F, R, the return stack; **pointers, P2 index and blocks restart** (Finding 51) |

### 6.2 Layout and pointer scheduling (the core problem)

Every ALU op needs its A operand on P0's page, its B operand on P1's page and its destination
at P2's walked word index. The index only moves forward: `inc` adds 1, `d0` adds 2, and a page
load resets it to 0. **As built** (`backend/layout.py`):
1. A first placement packs words into four-word pages by affinity: consecutive accesses by one
   pointer to two words want them on one page.
2. A seeded, bounded local search then moves and swaps words. It keeps a move when the
   scheduler's real output (bytes, then cycles) does not get worse.

Pointer loads were 38 % of all code bytes with the declaration-order placement; this halved
them. Before layout, `backend/opt.py` removes plain copies (propagation over the control-flow
graph) and stores nothing reads.

The scheduler tracks the whole machine state: the three pointers (block, bank, page), the P2
word index, routing, the mode register, the `fc`/`fd` toggles, `bf`, the predicate, mode 4, and
on the MPU whether R is 0 (the MAC restart). It emits only the bytes that change state. At a join, state
that differs between the paths becomes unknown and is reloaded. At a loop head the state is the
join of the entry and every back edge, found by iterating the schedule to a fixed point. **Width modes are lane modes**:
nothing arithmetic may run under `f2`/`f3` unless a 16-bit lane op is intended.

### 6.3 Instruction selection

| IR | selection |
|---|---|
| `a + b`, `a * b`, `-(a*b)` | ADD / MUL / NMUL; either operand may take either side |
| `a - b` | RSUB (B − A) with `b` in the A slot |
| any `+`/`-` expression with a product, or more than two terms | one `f0` MAC run: products as MAC ops, a plain term `t` as `t × 1.0`. It starts fresh on the MPU when the destination has moved (P2 page reload, `d0`, or the first op's pre-advance) and on the ICM when its first op pre-advances (so the ICM wants MAC destinations at odd words, and the layout puts them there) |
| copy, emit | MOV from P0 (18 cycles on the MPU); an int16 sample emitted as `f2`/`f3` MOV straight from the sample page |
| `s.accel.x`, `s.gyro.y` | `de`, then `f2` MOV or MOVX: the exact int16 lands in the upper half with a zero lower half |
| 16-bit data | a lane op under `f2`/`f3` is cheaper (22 / 52 cycles on the MPU) but computes on half-words |
| `2*a`, `4*a` | `a + a` with P0 and P1 on the same page |
| `dmp.clamp(x, -c, c)` | **one CLAMP** (`f5` op 0, Finding 41); an asymmetric clamp shifts to symmetric about its midpoint where `x − m` cannot wrap |
| `min`, `max` | one CLAMP when one side is a constant and the other's range allows; else compare-and-select: MOV, compare, skip the second MOV |
| `abs` | compare, then NMUL by 1.0 |
| `@dmp.sub` (ICM, called 2+ times) | one routine after the pass's `end`; `call`, `ret`; nothing assumed after a call |
| `a & b`, `a ^ b` (Bits) | `f5` AND / XOR (they leave F alone) |
| `n += 1` (Int) | INC |
| `a < b` and friends | RSUB into a scratch word (the `f5` "compare" only tests signs). Where range analysis says the subtrahend can be exactly −2.0, a guard RSUB against −2.0 + 1 lsb decides that case first. `==`/`!=`: RSUB then `d1`, with the operand that cannot be −2.0 as A, or the XOR of both against 0 |
| `for _ in dmp.loop(N)` | a real loop on both chips: a counter from N−1, DEC, branch back while not negative |

### 6.4 Conditions and control flow

Predicates are **sticky**, and **F freezes while a predicate is armed**, on both chips. So the
compiler computes every condition under `d8` and ends every conditional region with `d8`.

- **One lowering for both chips.** `a < b` is RSUB with `b` as A into a scratch word (F :=
  a − b < 0, exact), then `if F`/`if ¬F`; the others swap operands or negate. The exception is
  a subtrahend of exactly −2.0, which RSUB adds instead of subtracting (it wraps −A), and range
  analysis says where to guard it (§6.3). **Do not use the `f5` "compare"**: it tests signs
  only (Finding 40).
- **Every `f4` leaves mode 4 on both chips**, where a low byte is a jump: after any branch,
  taken or not, the next ALU op sets its mode first. MOV, CLR, INC and page loads need not.
- **MPU.** A branch-skip (`d9 f4 k d8 <then> d8`) nests naturally, because the body runs with no
  predicate armed. The branch reaches **−32..+31 bytes**; farther targets go through hops (a
  branch the fall-through steps over with `f4 02`), shared by every far branch to the same
  target, forward and backward. A taken branch costs 15 cycles; the bytes it jumps over cost
  nothing.
- **ICM.** After every `jump`/`call` (taken, not taken or skipped), the compiler inserts
  `d8 f1` at the target and on the fall-through. A loop's DEC always runs under `d8`, and the
  static call depth must be ≤ 4. `Int` comparisons with 0..63 use `fc cK Y`.

### 6.5 Packets, timing, range

- The emits, in program order, fix the packet at compile time. The bundle holds a schema, and
  the host runtime returns named fields scaled to floats.
- **Pass time is exact.** Worst-path cycles through `if`/`else` (ICM loops need a bound) plus
  the start latency, from the measured table, then compared with the sample period. Tick stalls
  only matter for passes that overrun the period.
- **Range analysis exploits the hardware** (`ir/ranges.py`, done). A Sum is exact and wraps once,
  and compares cannot overflow. So the only places a program can wrap are a Sum's result and
  `abs(-2.0)`. A warning names the place, the interval, and where the wrapped value goes: a
  multiply, a store, an emit, a compare, min/max. It also names the fix: declare a range,
  scale down, or `dmp.assume`.
  - **States without a declared range are inferred.** The pass is run to a fixpoint: widened
    to the full range if it does not settle, then narrowed, so a clamped state comes back.
  - **A declared range is an assumption, checked where the state is stored.**
  - **A branch on `v < c` narrows `v`.**
  - **`dmp.assume` states what interval arithmetic cannot see**, and the simulator checks every
    assumption on every pass. Examples: a stable filter's gain (the impulse response's L1
    norm), a resonator's energy, |q| = 1, a Newton invariant.

  Writing the examples against it found real problems:
  - `goertzel` could reach 1.28 on a full-scale resonant input, so its scale was halved.
  - `unit_vectors` lost 12 % of |v| on tiny vectors: it kept `m*4` instead of recomputing
    |v|², and 14 doublings were one too few.

  The per-node intervals are kept for the backend. For one: can MIN's `b − a` overflow?

## 7. ISA facts the compiler depends on, and the bench work behind them

`isa/facts.py` records every behaviour the backend relies on, with chip, status
(HW / VENDOR / INFER / UNMEASURED) and source (`python -m opendmp facts`): 66 facts, 64 of them
HW. The emulator's `Model` switches (defaults = measured) and `chips.py` (cycle counts with
their probes) are the executable form of the same table.

| id | question | chips | result |
|---|---|---|---|
| B1 | Does a program run with **no vendor image** (RAM zeroed, only our code and data)? | both | **measured: yes** (probe91 B1). The count ran gapless, one pass per sample (10,003 µs at 100 Hz), and 26/26 and 29/29 packets matched the emulator bit for bit. A device reset keeps DMP memory |
| B2 | Clobber map: which RAM does the hardware write? | both | **measured** (probe91 B2/FIFO): MPU 0x000–0x00d; ICM 0x0000–0x0011 plus **the FIFO at 0x3000–0x3fff**. Nothing while the DMP is off. MPU 0x00e's "constant 12" was a tag artefact |
| B3 | What survives from pass to pass? | both | **measured**: route, mode, F and R on both; the MPU keeps its pointers and P2 index, DMP3 restarts them; `fc`/`fd` do not (probe86 C, Finding 51) |
| B4 | What restarts the MAC sum? | both | **measured, and the chips differ**: MPU = a P2 move, which zeroes R (and nothing else: Finding 54); ICM = a MAC's op&4 only (probe86 B, probe88, probe116) |
| B5 | MAC precision and rounding | both | **measured**: exact products summed in a 64-bit register, floored once; MUL/NMUL floor and wrap mod 2³² (probe86 B, probe88, Finding 44) |
| B6 | DMP3 `f0`, the `f5` family | ICM | **measured**: `f0` is the MAC; `f5`–`f7` as on the MPU (probe88) |
| B7 | Clears under routing and widths | MPU | **measured**: a clear is a routed store of 0 and honours the lane (probe86 A) |
| new | the flags | both | **one exact-sign F, no Z**; AND/XOR leave F alone; CLAMP sets F := it clamped; `d1` tests R's word, R being the last ADD/RSUB/MUL/NMUL/MAC/INC/DEC result or CLAMP's compare (probe86 E–H, probe88, Finding 45) |
| new | `f2`/`f3`/`f6`/`f7` | both | **16-bit lanes** (probe86 A/E/F, probe88) |
| new | timing | both | **cycle counts, clocks, start latency, tick stalls** (probe90, with the bridge's new `W` command) |
| new | control flow and compares, found by the first silicon runs | both | **every `f4` leaves mode 4** on the MPU too; the MPU branch is **signed 6-bit**; the `f5` "compare" **tests signs only**; RSUB's F is exact except A = −2.0; MOV/CLR/INC under every mode as modelled (probes 92–101, Finding 40) |
| new | `f5` op 0, measured before compiling to it | both | **a symmetric clamp**, not MIN; F := clamped; lanes too (probe103, 1,920 cells per part, Finding 41) |
| new | the MPU's mode-4 low bytes | MPU | **one-byte branches**, 10 cycles taken, predicable; no absolute jump; the vendor's eMD code chains hops with them (probe104, Finding 42) |
| new | the edges, measured before the manual rewrite | both | the 64-bit MAC register and MUL's wrap (F44); what `test` reads (F45); DMP3's integer multiply (F46); nothing preempts a pass, the sample page moves under it (F47); all of each code space usable (F48); every cycle class (F49); INC/DEC into the FIFO (F50); the state a pass starts with (F51); FIFO overflow (F52); every manual example on silicon (F53); a MAC after a move continues from R (F54) (probes 106–116) |

## 8. How it is verified

1. **Emulator against silicon, cell by cell.** probe86 (MPU, 120 cells) and probe88 (ICM, 36
   cells) run every cell on the rig and in the emulator side by side. All match. The stock
   MotionApps20 image emits its 42-byte packets at the measured rate for FIFO-rate dividers
   0–3 (200/100/66.8/50 per second on silicon). From the cycle table alone it reproduces
   probe41's pass rates at all five sample rates, including the non-monotonic 500 Hz dip.
2. **Unit tests** transcribed from the recorded probe results (probes 7/9, 32, 35, 47, 51, 53,
   67–70, 74, 83b, 86, 88, 90, 91, 106–116), plus cycle sums and scheduling.
3. **The front end.** Every example lowers with no range warning. The reference and the IR
   interpreter agree bit for bit on every example, over every kind of synthetic motion, with
   no wrap and no failed assumption. Hand-computed words cover the rules where a plausible
   alternative gives different bits: one floor per sum, exact integer multiples, wrapping,
   typing, lists, control flow, helpers and subs. A suite of 40 refused constructs checks each
   message, hint and caret.
4. **The ISA tables.** Every byte value, in every mode and with every operand tail, decodes
   and re-encodes to itself. All 13 vendor images (about 45,000 instructions) disassemble to
   text that reassembles byte for byte. The listing's cycle class equals the emulator's for
   every instruction the emulator executes.
5. **Differential testing.** Every example, on both chips, emulated from its bytes against the
   reference on all kinds of synthetic motion; tilt on 10,000 random full-scale samples. For every example and test program, the reference model
   (AST rewrite) and the emulator (compiled bytes) must agree bit for bit.
6. **Listing self-check.** The `.asm` a build writes reassembles to the same bytes (tests). Decoding the emitted bytes must reproduce the intended
   instruction stream, and the `.asm` must reassemble to identical bytes.
7. **Silicon.** `opendmp run --verify` loads a *trace build* (each packet starts with the pass's
   raw sample words), streams the FIFO, and replays the samples through the same bytes in the
   emulator and through the reference. **All 12 examples are bit-exact on both rigs**: every
   packet, every interrupt count, about 3,700 passes in all. The first runs were not. The four
   behaviours behind the misses were isolated on the rigs with probes 92–101 and are recorded
   as Finding 40.

## 9. Phases and milestones

| phase | contents | status |
|---|---|---|
| **1 Emulator** | Q30 model; machine for both chips; pass loop, FIFO, INT; cycle model; file I/O; B3–B7 | **done** 2026-09-23 |
| **0 Ground truth** | ISA tables, encoder, decoder and `dis` for both chips; `facts.py`; **B1, B2** on both rigs | **done** 2026-09-23 (probe91, Finding 39) |
| **2 Front end** | ast subset, errors, IR, reference semantics, range analysis | **done** 2026-09-23: the examples parse, simulate (two engines, bit-exact) and range-check clean; 40 refused constructs give their documented messages |
| **3 Backends** | layout, selection, scheduler, MPU and ICM control flow, `.lst`/`.asm`/`.bin`/`.json` | **done** 2026-09-23: M1 met; all 12 examples build for both chips, emulator = reference |
| **4 Runtime and silicon** | `host/` (bridge, whole-program init, params, packet stream), trace builds | **done** 2026-09-23: M2 met, and more: **all 12 examples bit-exact on both rigs** |
| **5 Depth and polish** | a layout that minimises pointer loads, CLAMP (was "MIN where it is safe"), ICM `call` for `@dmp.sub`, state readback and live param writes in the runtime, tutorial and language reference | **done** 2026-09-23: −39 % code, −22 % worst-case cycles, all 12 bit-exact on both rigs |

## 10. Risks

| risk | mitigation |
|---|---|
| ~~B1 fails~~ | **retired**: B1 passed on both parts (probe91) |
| the ICM FIFO shares RAM with code | code and data end below 0x3000 (`Chip.code_limit`); the emulator writes FIFO bytes into 0x3000+ so a program that strays there fails in simulation as it would on silicon |
| Clock tolerance: another part runs at a different MHz | the cycle counts are properties of the core; `--clock-mhz` takes the part's clock, and the host runtime can measure it with `W` |
| Q30 range surprises users | range analysis that names the fix; declared ranges; examples that keep signals within ±1 |
| a host param write is torn mid-pass (four bytes, non-atomic) | the runtime writes params between passes (halt, write, resume), or double-buffers them behind a version word |
| layout quality eats the MPU code budget | measure cycles on the examples first; the MPU budget is 2016 B and `tilt` is 45 |
| Python-looking code implies Python semantics that don't hold | the reference rewrite defines the semantics; the docs list every difference (wrap, floor, no aliasing, lanes) |

## 11. Not in v1

Extending stock firmware and patching vendor slots; automatic fixed-point scaling; conditional
emits and variable-length packets; runtime-indexed arrays; the MPU-6500/9250 profiles (same
mantis ISA, add later); the 3050 lineage; an optimal (ILP) layout.

## 12. Still to decide (non-blocking)

- License for opendmp.
- Whether to copy the ESP32 bridge sketch and prebuilt images into this repo, or keep pointing
  at `InvensenseDMP/bench/esp32_dmp_bridge`.
- Whether `opendmp build` prints the full `.lst` by default or only the summary.

## 13. Issues found in InvensenseDMP (fixed there where noted)

1. **Fixed 2026-09-23.** `dmpasm.py` accepted `branch 0x40`–`0x7f`, but the displacement is 6
   bits (`f4 41`–`7f` raise DMP_INT, `f4 40` does nothing) and signed (item 7). It now takes
   −32..31 or a label, and refuses an ALU op after a `branch` until a mode byte. The manual is
   corrected.
2. **Fixed 2026-09-23.** Manual §2.9 said there was no AND/XOR and no compare other than
   `== 0`/MIN.
3. **Fixed 2026-09-23.** `dmpasm.py` had no mantis mnemonic for `de` (clear in place); it is `clr`.
4. **Superseded by Findings 37/38; the manual was rewritten for them on 2026-09-23:**
   - the MPU's "N and Z, d9 = IF (N ∨ Z)";
   - "f2/f3 are store widths";
   - "f0 on DMP3 is unmeasured";
   - the 0.58/2.0/3.9/7.9 µs + 56 µs timing model;
   - Finding 28's "d9 fires on 0", which was the op's own zero emit read as the marker.
5. **Superseded by Finding 39** (the manual was corrected): MPU "0x00e reads a constant
   12" (probe84's own tag), and the ICM's "free RAM to 0x3fff" (the FIFO lives at
   0x3000–0x3fff).
6. `dmpasm.py`'s mnemonics predate Findings 37–39 in places: `if le0`/`if ge0` on mantis (it is
   `if f`/`if nf`, which it now also accepts), `mov16` (MOVX), the ALU op under `f0` spelled `add` (it is a MAC), `disable`
   (end of pass). opendmp's assembler uses the corrected names and accepts `disable`/`mov16` as
   aliases. The two syntaxes produce identical bytes for `examples/emu/tilt`.
7. **Superseded by Finding 40** (the manual was rewritten for it on 2026-09-23, and the
   vendor's eMD 6.12 images turn out to loop with it): the MPU branch is signed
   (−32..+31), not "forward 0..63" (dmpasm accepts 0..0x7f); every `f4` leaves mode 4 on the
   MPU too, and mode-4 low bytes are jumps (Finding 33's "no jump" is wrong for them); the
   `f5` op 1 "compare" tests signs only (Finding 37's "F := a − b < 0" rests on opposite-sign
   cells).
8. **Superseded by Findings 44–54** (2026-10-08; the manual was rewritten for them, in the
   MCS-51 manual's order): Finding 37's "`d1` tests the last adder result" (it reads the 64-bit
   result register R, which INC, DEC and CLAMP also write); "the MAC accumulator resets every
   pass" (R carries over); "DMP3 keeps its pointers across passes" (it restarts them); "any
   destination move restarts the MPU's MAC" (the move zeroes R, and nothing else restarts it);
   "`movz` makes MOV store zero" (it stores a FIFO-dependent value); "clearing DMP_EN stops the
   DMP" (a running pass goes on). opendmp followed: the compiler, the emulator, `facts.py` and
   `Rig.stop` (see the status, 2026-10-08).
