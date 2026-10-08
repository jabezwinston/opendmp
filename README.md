# opendmp

A Python-like language, compiler and **cycle-accurate emulator** for the InvenSense DMP, the
motion processor inside the MPU-6050 and the ICM-20948. The plan of record is
[PLAN.md](PLAN.md). The language is specified by example in [examples/](examples/). The ISA
knowledge comes from the sibling reverse-engineering repo, `~/Git_repos/InvensenseDMP`.

**Start with [docs/tutorial.md](docs/tutorial.md)**; the language is specified in
[docs/language.md](docs/language.md).

**Today:**
- **The language's front end.** It parses, types, simulates and range-checks programs, and
  there are two independent executable semantics that agree bit for bit.
- **The compiler**, for both chips. All 13 examples build, and their bytes run in the emulator
  equal the reference semantics on every pass.
- **The host runtime.** It loads a program over the bench bridge, streams its packets, reads
  states and writes params while it runs, and verifies a run on silicon. **All 13 examples are
  bit-exact on both parts.**
- **The cycle-accurate emulator.**
- **The ISA tools:** assembler, disassembler, and the machine facts.

A program needs **no vendor firmware**: it runs from zeroed memory on both parts, as measured
on the bench (below).

## The language

```python
import opendmp as dmp

@dmp.program(rate=200)
def tilt(s):
    angle: dmp.state(-1.5, 1.5) = 0.0            # kept across passes; the range is an assumption
    limit: dmp.param = 0.25                      # the host may change it while running
    angle = dmp.clamp(0.98 * (angle + s.gyro.y * 0.0873) + 0.02 * s.accel.x, -1.5, 1.5)
    dmp.emit(angle, s.accel.z)                   # packet: angle q30 (4 B), accel_z int16 (2 B)
    if angle > limit:
        dmp.irq()
```

```bash
python -m opendmp check examples/tilt.py        # types, packet layout, anything that can wrap
python -m opendmp sim examples/tilt.py --gen tilt,walk -o out.csv    # or --sensors data.csv
python -m opendmp ir examples/biquad.py          # what the front end made of it
```

From Python: `tilt.simulate(samples)` runs the reference semantics, and `tilt.check()` returns
the range warnings. `tilt.build("mpu6050")` compiles it (below). The rules are in PLAN.md §2 and §4. The one to know: a `+`/`-` expression
is summed exactly and wraps once, which is what the DMP's multiply-accumulate computes.

An error points at the construct and says what to do instead:

```
tilt.py:14:17: error: division by a variable: the DMP has no divide
        angle = x / y
                ^^^^^
hint: multiply by a constant reciprocal (x * (1 / C)), or use a Newton step (examples/unit_vectors.py)
```

A range warning names the interval, where the wrapped value goes, and the fix:

```
acc.py:7:11: warning: `acc + s.gyro.z * 0.5` can reach [-3, 3], outside [-2, 2): it wraps before it is stored in acc
        acc = acc + s.gyro.z * 0.5
              ^^^^^^^^^^^^^^^^^^^^
hint: declare a range on `acc` (dmp.state(lo, hi)) if it is narrower, or scale the terms down (x * 0.5), or dmp.assume(x, lo, hi) what interval arithmetic cannot see
```

## The compiler

```bash
python -m opendmp build examples/tilt.py            # build/tilt.mpu6050.{bin,lst,asm,json}, and icm20948
python -m opendmp build examples/tilt.py --chip mpu --list    # print the listing
```

A build writes four files per chip:
- `.bin`: the memory image, code and data;
- `.lst`: the listing: every instruction with its address, bytes, cycles and the source line
  it came from, and the data words by name;
- `.asm`: the same program as opendmp assembly, annotated, which reassembles to the same bytes;
- `.json`: the packet layout, the symbols (state, params, constants) and the entry point, for
  the host.

```
; tilt  opendmp  chip mpu6050 (8.199 MHz)  entry 0x0400  200 Hz
; addr  bytes     asm                    effect                                   cycles
0400  d8        always                 predicate off                                5
0401  f1        fmode 1                32-bit                                       5
...
        ; tilt.py:33  angle = dmp.clamp(K * (angle + s.gyro.y * GYRO_DT) + (1 - K) * s.accel.x, -1.5, 1.5)
0410  f0        fmode 0                multiply-accumulate                          5
...
0416  78        mac 3,3                acc + P0.3 * P1.3                           60
```

From Python, `b = tilt.build("icm20948")` gives the image, `b.listing()`, `b.asm()`,
`b.schema()`, and `b.emulate(samples)` runs the bytes in the emulator.

What the compiler does (PLAN.md §6):
- **Every `+`/`-` expression with a product is one MAC run**: products summed exactly and
  floored once, as the language defines.
- **`dmp.clamp(x, -c, c)` is one instruction.** The DMP's `f5` op 0, long read as MIN, clamps
  A to [−B, B] (ISA2 Finding 41). Asymmetric clamps shift to symmetric, and `min`/`max` with a
  constant use it where the value's range allows. Other `min`/`max`/`abs` are compare-and-skip.
- **Compares are RSUBs** into a scratch word; the DMP's `f5` "compare" tests signs only.
  Where range analysis cannot rule out a subtrahend of exactly −2.0, a guard decides it first.
- **Loops**: `dmp.loop(N)` and `while` are real loops on both chips. At a loop head the
  scheduler keeps whatever pointer and mode state the entry and every back edge agree on.
- **Branches.** The MPU's reach −32..+31 bytes. Far branches are relaxed into shared hops,
  and in mode 4 (after any branch) a branch is one byte (ISA2 Finding 42), as in the vendor's
  own eMD code. The ICM has 16-bit jumps.
- **Subs**: on the ICM a `@dmp.sub` called more than once is one routine (`call`/`ret`).
- **Division**: `x / C` is a multiply; `x / y` is exact restoring division (`dmp.div`) and
  `n // m` on Int values is Python's floor division (`dmp.idiv`), both written in the language
  (`opendmp/lib.py`) and exact for every input.
- **Sums without products** are ADD/RSUB chains: the same value modulo 2³², cheaper than a MAC
  run.
- **Clean-up and layout.** Copy propagation and dead-store elimination run over the
  control-flow graph. The layout of words in RAM is then searched, scored by the scheduler's
  actual bytes and cycles (pointer loads were over a third of all code). On the ICM, MAC
  destinations go at odd words.

| example | MPU-6050 bytes | cycles/pass (max) | ICM-20948 bytes | cycles/pass (max) | packet |
|---|---:|---:|---:|---:|---:|
| hello_fifo | 11 | 204 | 12 | 217 | 6 |
| tilt | 45 | 672 | 49 | 529 | 6 |
| pid_level | 56 | 1,043 | 55 | 711 | 8 |
| goertzel | 60 | 1,190 | 63 | 772 | 4 |
| vibration | 70 | 983 | 81 | 726 | 8 |
| gyro_quat | 80 | 2,255 | 90 | 1,307 | 16 |
| pedometer | 95 | 1,121 | 108 | 873 | 8 |
| biquad | 97 | 1,834 | 98 | 1,185 | 12 |
| taps | 116 | 1,128 | 136 | 998 | 8 |
| mahony | 190 | 4,575 | 221 | 2,668 | 16 |
| despike | 341 | 3,306 | 383 | 2,841 | 12 |
| unit_vectors | 359 | 23,071 | 261 | 16,245 | 28 |
| slope (`x / y`) | 266 | 11,467 | 287 | 9,433 | 8 |

Cycles are the emulator's on walking and shaking motion. `opendmp build` also lists, per
construct and line, the method it chose (`docs/language.md` §11). Every build also carries an
upper bound
on a pass, loops at their trip counts (`build.cycles_max()`, the listing's last line): 23,071
cycles on the MPU for unit_vectors, 2.8 ms of its 10 ms period, against 20,828 on its worst
inputs. Before the Phase 5 optimisations the twelve came to 5,099 bytes;
they are now 3,078 (−40 %).

## Running on silicon

```bash
python -m opendmp run examples/tilt.py --port /dev/ttyUSB0 --seconds 5 -o tilt.csv
python -m opendmp run examples/mahony.py --port /dev/ttyUSB1 --seconds 5 --verify
```

`run` detects the part on the bench bridge (InvensenseDMP `bench/`), resets it, writes a
minimal register set and the program over zeroed DMP memory, starts it and decodes its packets.
The program is the whole firmware: no vendor image is loaded.

`--verify` loads a **trace build** instead: every packet starts with that pass's raw sample
words. The samples are replayed through the same bytes in the emulator and through the
reference semantics, and every packet and interrupt must match:

```
mahony on the icm20948: 155 passes at 50.0 Hz; emulator 0 and reference 0 packets differ; interrupts 0 on silicon, 0 emulated; 4394-4394 cycles a pass: BIT-EXACT
```

All 13 examples verify on both rigs, about 4,000 passes in all, before and after the Phase 5
optimisations. The first runs did not; the misses led to four ISA behaviours that no one had
documented (ISA2.md Finding 40, below).

Params and states are live:

```bash
python -m opendmp run examples/pid_level.py --port /dev/ttyUSB0 --seconds 4 \
    --param setpoint=0.1 --set 2:setpoint=-0.1 --watch integ,pitch
```

`--param` sets before the start, `--set T:NAME=VALUE` during the run, and `--watch` reads states
as it goes (`--watch-output` saves them as CSV). A write outside a param's declared range is
refused. A program that reads `s.fsync` gets the FSYNC pin (the runtime sets `EXT_SYNC_SET`).

From Python: `from opendmp.host import Rig, verify`; `rig = Rig(port)`, `rig.load(build)`,
`rig.run(seconds, watch=[...], changes=[(t, name, value)])`, `rig.read(name)`,
`rig.write(name, value)`.

Hand-written assembly runs the same way, loaded alone:

```bash
python -m opendmp run count.asm --port /dev/ttyUSB1 --passes 3 --poke 0x100=0x0f0f0f0f \
    --dump 0x200:4 --verify
```

It assembles the file (for its `.chip`, or the bridge's part), runs one pass (or `--passes N`,
or `--seconds`), and prints the FIFO and the `--dump` ranges. `--verify` replays the image in
the emulator, which gives every pass `--sample` (zeros by default), and reports BIT-EXACT or
each difference. From Python: `rig.load_image(image, entry, rate)`, `rig.step()`,
`rig.run_raw(passes=3)`.

## The emulator

```bash
python -m opendmp.emu run.toml                   # or everything as options, see --help
python -m opendmp.emu --costs --chip icm20948    # the cycle table and where each number comes from
python -m opendmp.emu --load prog.asm --sample ax=100,gz=-5 --passes 3 --fifo out.bin
python -m opendmp.emu.gen walk,taps --rate 200 -o data.csv    # synthetic sensor data
```

It runs DMP machine code for either chip, one pass per sensor sample, and takes everything
from files:

| input | format |
|---|---|
| sensor data | CSV, one row per sample tick; header names any of `ax ay az gx gy gz temp fsync` (or `accel_x` …). Raw int16 counts, or `units = "si"` for g, deg/s and deg C at the given full scales |
| program and RAM | `load = ["prog.asm"]`: opendmp assembly, which places its own code and data (its `.chip` picks the chip). Or `"prog.bin@0x400"`: binary, `.hex` text, or `.s` assembled with InvensenseDMP's `dmpasm.py`. Load a whole RAM image the same way |
| one fixed sample | `sample = "ax=100,gz=-5"`: with no sensor file, every pass sees this sample (zeros if omitted), for `passes` passes |
| constants | `poke = ["0x100=0.98q,40000000"]`: Q30 values with a `q`, or raw hex words |
| DMP3 I/O space | `regs = "regs.json"`: `{"bank:reg": value}`, what a page load under mode 4 reads |

| output | contents |
|---|---|
| `log` | CSV, one row per pass: sample seen, start time, duration in µs **and cycles**, instructions, interrupts, the FIFO bytes it emitted, and any `watch`ed RAM words as Q30 |
| `fifo` | the raw FIFO stream |
| `trace` | per-instruction trace of the first `trace_passes` passes: address, bytes, disassembly, cycles, effect (`M[0x110] := …`, `fifo <- …`, `call 0x1010`) |
| `dump` | DMP memory after the run |

A complete example is [examples/emu/](examples/emu/): the tilt filter from PLAN.md §3 as
assembly (`tilt.asm`), a run config and generated motion.

From Python:

```python
from opendmp.emu import Machine, run, read_sensor_csv
m = Machine("mpu6050")
m.load(code, 0x400)
m.poke_words(0x100, [0x3EB851EC])
results = run(m, read_sensor_csv("walk.csv"), rate_hz=200)
results[0].cycles, results[0].fifo
```

### What "cycle-accurate" means here

Every instruction class costs an integer number of DMP clock cycles, measured on the bench
rigs. The bridge's `W` command times two interrupts inside one pass, and every measured cost
came out an integer multiple of one clock period (InvensenseDMP `bench/probe90.py`, ISA2.md
Finding 38):

| | MPU-6050 | ICM-20948 |
|---|---:|---:|
| clock (this bench's parts) | 8.199 MHz | 20.975 MHz |
| predicate / mode / pointer / skipped byte / `int` | 5 | 4 |
| CLR | 12 | 11 |
| MOV | 18 | 17 |
| CMP | 17 | 16 |
| INC, DEC | 23 | 21 |
| AND, XOR | 23 | 22 |
| ADD, RSUB, CLAMP | 28 | 26 |
| MUL | 58 | 25 |
| MAC | 60 | 27 |
| SHL, SAR | | 15 |
| taken branch / jump | 15 | 13 |
| a branch in mode 4 (one byte) | 10 | 9 |
| call, ret | | 13, 5 |
| sample tick → first instruction | 110 | 130 |

On a 16-bit lane (`f2`/`f3`, `f6`/`f7`) the ALU ops are cheaper: ADD 22/20, MUL 52/19.

`--costs` prints the full table with the probe behind each number. Only `end` is unmeasured
(charged as a mode byte); the rows for instructions a core does not have say so too. The
scheduler models what the sample clock does to a pass that overruns:
- **MPU:** it drops to an integer submultiple of the rate, and every sample write inside a
  pass stalls it about 20 cycles.
- **ICM:** it runs at 1/pass time.
- **Both:** the pass reads the newer samples written while it runs, as silicon does.

The clock is the bench parts' own; `--clock-mhz` takes yours.

### How it was validated

- **Semantics.** 156 cells (probe86 on the MPU-6050, probe88 on the ICM-20948) run on silicon
  and in the emulator side by side; all match.
- **The edges.** Probes 106–116 (Findings 44–54): the 64-bit MAC register, what `test` reads,
  DMP3's integer multiply, pass carry, mid-pass samples, FIFO overflow. The emulator matches
  every cell it models.
- **The manual.** Every `asm` example in InvensenseDMP's manual gives its stated result in the
  emulator (132 runs) and on silicon (62 on the MPU, 69 on the ICM).
- **The vendor's firmware.** The stock MotionApps20 image (1929 B) emits its 42-byte packets at
  exactly the rates silicon does for FIFO-rate dividers 0–3. Its pass (20,465 cycles)
  reproduces the measured pass rates at five sample rates, including the non-monotonic dip at
  500 Hz.
- **Unit tests** (`pytest`) transcribe the recorded hardware results.

Whatever has no measured meaning raises `Unmodelled` instead of being guessed: MPU `bf`,
DMP3 `fc X Y` with X < 0xc0, DMP3 `fe`, a MOV under DMP3 `movz`, both operand exchanges at
once. `Model` (`--model key=value`) names the rules a result rests on, such as how a MAC run
restarts and what survives a pass: the defaults are the measured behaviour, the other values
are for experiments, and an unknown value is refused.

### Machine facts the emulator encodes

Measured for this project (ISA2.md Findings 37–54); InvensenseDMP's manual (`docs/manual/`)
now states them all:
- **One flag on both chips.** ADD, RSUB and MUL set F to the sign of the **exact** result
  (RSUB: unless A = −2.0, which it adds instead of subtracting). There is **no Z**: `d9` = IF F,
  `da` = IF ¬F.
- **`test` (`d1`) reads the result register R**: F := (R's word == 0), at the lane in force. R
  holds the last ADD, RSUB, MUL, NMUL, MAC, INC or DEC result, or CLAMP's compare; AND, XOR,
  CMP, MOV and CLR leave it. So RSUB then `test` is an exact `==` for every pair, −2.0 included.
- **The `f5` "compare" tests signs only**: F := A < 0 ≤ B. Earlier probes had used operands of
  opposite signs, where it agrees with A − B < 0.
- **The MPU's branch is signed**: `f4 d` reaches −32..+31 bytes. **Every `f4` leaves mode 4**
  on the MPU as on the ICM, taken, not taken or skipped. An ALU op there reads as a jump, so
  set the mode again after a branch.
- **`f2`/`f3` are 16-bit lane modes, not just store widths.** Under `f3` the ALU computes on the
  sign-extended low halves, so a MUL there is `(lo·lo) >> 30`. Under `f2` it computes on the
  high halves.
- **`f5` op 0 is a symmetric clamp, not MIN**: A ≥ 0 ? min(A, B) : max(A, −B), F := it clamped
  (1,920 cells per part, Finding 41). **AND/XOR** leave F alone.
- **In the MPU's mode 4 a byte below 0x40 is a one-byte branch** (Finding 42): there is no
  absolute jump, but a branch after a branch costs one byte and 10 cycles.
- **R is the 64-bit MAC accumulator** (Q60 in [−8, 8), wrapping): products are summed exactly
  and floored once, when stored. MUL wraps its word mod 2³² and keeps an exact-sign F. On the
  MPU a P2 move zeroes R, and that is the whole restart: a MAC after INC, DEC or CLAMP adds to
  their result. On the ICM only a MAC's own `op&4` restarts the sum. **DMP3's `f0` is the MAC.**
- **DMP3 multiplies integers in two instructions**: MUL loads the shifter with the whole
  product, and SHL 30 stores its low word (what `n * m` compiles to on the ICM).
- **Across passes** both chips keep the route, mode, F and R. The MPU also keeps its pointers
  and P2 index; DMP3 restarts them every pass, and keeps its return stack. The `fc`/`fd`
  exchanges do not survive. After DMP_RST: route RAM, pointers 0, mode `f1`, F and R 0.
- **Nothing preempts a pass.** Clearing DMP_EN lets a running pass go on (`Rig.stop` pulses
  DMP_RST), and the sample page is rewritten at every tick, even under a running pass.
  **The FIFO keeps the newest bytes**: 1 KB on the MPU, a 4 KB ring on the ICM.
- **No vendor image is needed** (Finding 39). The hardware writes only the sample page:
  0x000–0x00d on the MPU; on the ICM it clears 0x0000–0x0003 and writes 0x0004–0x0011.
  **The ICM's FIFO is DMP RAM 0x3000–0x3fff**, so code must end below 0x3000; the emulator
  writes FIFO bytes there too. A device reset keeps DMP memory, except that the ICM's writes
  8 bytes at 0x0080–0x0087.
- **DMP3's `movz`** makes MOV store a FIFO-dependent value, and its `fe` raises no interrupt:
  use neither (the emulator refuses both).

## The ISA tools

```bash
python -m opendmp asm examples/emu/tilt.asm -o tilt.bin       # listing with cycles; memory image
python -m opendmp dis tilt.bin --start 0x400                   # annotated disassembly
python -m opendmp dis image.bin --chip icm20948 --org 0x90 --asm > image.asm   # reassemblable
python -m opendmp facts --chip icm20948                        # what the tools rely on, and why
```

`opendmp.isa` holds both instruction sets as tables. `decode` and `encode` are exact inverses:
every byte value in every mode decodes and re-encodes to itself, and all 13 vendor images
disassemble to text that reassembles byte for byte. The assembly language is dmpasm's, with
the names corrected by Findings 37–39:
- `if f` / `if nf`;
- `movx`;
- `fmode 0` then `mac A,B` for the multiply-accumulate;
- `fmode 5` then `clamp`/`cmp`/`and`/`xor` (`min` is accepted for `clamp`);
- `branch.m4` for the MPU's one-byte branch in mode 4;
- `end`.

The assembler tracks the mode register and refuses code the hardware would read differently:
- an ALU op from the wrong family (`min` under `fmode 1`);
- any ALU op while DMP3's mode 4 is set;
- an op at a jump target that different paths reach in different modes;
- a branch outside the MPU's −32..+31 bytes.

```asm
        .chip   mpu6050
        .org    0x100
const:  .dw     0.0873q, 1.0q, 0.98q, 0.02q
        .org    0x400                   ; .entry defaults to 0x400 (MPU), 0x1000 (ICM)
        always
        store                           ; route -> RAM
        fmode   1
        rsub    3,0                     ; P2's word := P1.0 - P0.3; F := it < 0
        if nf
        branch  join                    ; P0.3 <= P1.0
        always
        int
join:   always
        end
```

## Layout

```
opendmp/q30.py         the number model
opendmp/api.py         the language's names (dmp.program, dmp.state, dmp.emit, ...); quat.py: dmp.quat
opendmp/lang/          the front end: lowering to the IR, and the reference semantics
opendmp/ir/            the IR, its interpreter, range analysis
opendmp/sim.py         simulate(): packets, interrupts, decoded fields, wraps, failed assumptions
opendmp/backend/       the compiler: selection (tac), clean-up (opt), layout search, scheduler, assembly (build)
opendmp/host/          the bench bridge, Rig (load, start, stream) and verify
opendmp/isa/           both ISAs as tables: encode/decode, the assembler, facts.py
opendmp/cli.py         python -m opendmp check | sim | ir | build | run | dis | asm | facts
opendmp/emu/           the emulator (machine, chips, sched, files, gen, disasm, CLI)
examples/              12 programs in the opendmp language; examples/emu/, a hand-written one in assembly
tests/                 pytest: hardware-transcribed vectors, cycles, scheduling, the ISA round trips,
                       the language (semantics, 40 refused constructs), the compiler (every example,
                       both chips, bit-exact against the reference), CLIs
docs/                  tutorial.md (start here), language.md (the reference)
PLAN.md                the plan of record
```

Tests: `python -m pytest`. The corpus-based tests look for `../InvensenseDMP` and skip
without it; no vendor firmware is stored here.
