# opendmp: plan of record

A Python-like language and compiler for the InvenSense DMP. You write one Python function, and
opendmp turns it into a DMP program for the **MPU-6050** and the **ICM-20948**. It also
produces an annotated listing, a cycle-accurate emulator run and a host runtime that loads the
program onto the bench rigs and decodes its FIFO packets by name.

The ISA knowledge comes from `~/Git_repos/InvensenseDMP`: the programmer's manual in
`docs/manual/`, the evidence log in `ISA2.md` (Findings 37 and 38 were measured for this
project) and the bench in `bench/`. This repo holds only original code. **No vendor firmware
is shipped or needed** (see B1 below).

**Status, 2026-09-23.**
- **Done:**
  - the plan and 12 example programs in `examples/`;
  - the **emulator**, `opendmp/emu/`: cycle-accurate, fed from files, and validated against
    both rigs cell by cell (§8);
  - **Phase 0**:
    - the ISA tables, encoder, decoder, assembler and `dis` for both chips (`opendmp/isa/`);
    - `facts.py`;
    - **B1 and B2 measured**. A program runs with **no vendor image** on both parts, and the
      clobber map is known. On the ICM the FIFO is DMP RAM 0x3000–0x3fff.
- **Next:** Phase 2, the front end.

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
  `break` is a bounded loop: a real backward jump on the ICM, unrolled with forward exits on
  the MPU. `while` is ICM-only.
- **Helpers.** A helper `def` is inlined and may update the state lists passed to it. On the
  ICM, `@dmp.sub` makes a real call. Its arguments and results live in the sub's own fixed
  page and are copied in and out, because the DMP has no indirect addressing.

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
§6. Its bytes were checked by assembling the `.asm` with opendmp's assembler and with
`InvensenseDMP/tools/dmpasm.py` (45 bytes, identical). They also **run in the emulator** as `examples/emu/tilt.toml`, where the
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
0424    f5        fmode 5                                                          5  0.61
0425    61        cmp 3,0            F := angle - limit < 0 (exact: no wrap)      17  2.07
0426    d9        if f                                                             5  0.61
0427    f4 02     branch 0x042b      angle <= limit: skip the irq                 15  1.83
0429    d8        always             angle > limit: fall through                   5  0.61
        ; tilt.py:12  dmp.irq()
042a    fe        int                raise DMP_INT                                 5  0.61
042b    d8        always             join                                          5  0.61
042c    ff        end                end of pass                                   5  0.61

; data   0x100 const  GYRO_DT 059652bd  1.0 40000000  K 3eb851ec  0.02 0147ae14
;        0x110 work   gy  ax  t  angle(state, init 00000000)
;        0x120 param  limit 10000000
; packet 6 B   angle q30 [0:4]   accel_z i16 [4:6]    (generated parser: build/tilt.json)
; pass   45 B: 624 cycles = 76.1 us after the tick (629 when it interrupts), 1.5 % of 5000 us
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
| `if`/`elif`/`else`, nested | yes (branch-skip, trampolines past 63 B) | yes (jumps) |
| `for i in range(N)`, N constant | unrolled | unrolled |
| `for _ in dmp.loop(N)` + `break` | unrolled, with forward exits | a real loop |
| `while` | **error**: the MPU has no backward branch | a real loop; warns: unbounded pass time |
| helper `def` | inlined | inlined, or `@dmp.sub` = call (depth ≤ 4, no recursion) |
| state lists, constant indices, tuple unpacking, several return values | yes | yes |
| `Int` `+= 1`, `-= 1`, compare with 0..63 | `INC`/`DEC`, compare via ALU | `INC`/`DEC`, `fc cK Y` immediate compare |
| `Bits` `&`, `^`, `<<`, `>>` | yes | yes |
| `dmp.emit`, `dmp.irq`, `return` (ends the pass), `dmp.assume` | yes | yes |
| divide by a variable, `sqrt`, `**` (other than `x*x`), runtime list index, recursion, emit under a condition, `<<` on Q30 | **error**, with a hint (`dmp.rsqrt` does a Newton step; tables need constant indices) | same |

**Reference semantics.** `tilt.simulate(samples)` runs the source through a plain-Python
rewrite of its AST over the Q30 model: wrapping add, a multiply that floors, signed min, and
the **exact-sign flag** (§6.4). The rewrite shares only the parser with the compiler, so it is
an independent oracle for the whole backend. The emulator is the second oracle, working from
the bytes.

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
        (AST rewrite)     same Rig API: dmp.connect("emu:mpu6050") / dmp.connect("/dev/ttyUSB0")
```

```
opendmp/
  __init__.py      the whole user API (program, state, param, emit, irq, clamp, connect, ...)
  q30.py           the number model                                        [done]
  emu/             the cycle-accurate emulator, both chips                 [done, §8]
    machine.py       state, step, pass; every unmeasured behaviour a Model switch
    chips.py         memory geometry, clocks, the measured cycle table, sample layouts
    sched.py         the pass timeline: ticks, rate drop (MPU) / 1/pass (ICM), tick stalls
    files.py         sensor CSV (raw or SI units), program/RAM images, pokes, dmpasm sources
    gen.py           synthetic sensor data (rest, tilt, rotate, walk, taps, vibration, ...)
    disasm.py        mode-aware disassembly for traces
    __main__.py      python -m opendmp.emu [run.toml] [options]
  isa/             both ISAs as tables                                     [done]
    core.py          Insn, decode/encode (inverses, checked on 13 vendor images), cost keys
    asm.py           opendmp assembly: .chip/.entry/.org/.dw/labels, mode tracking, disassembly
    facts.py         every machine behaviour relied on, with status and evidence (§7)
  lang/            parse, check (subset and errors with a caret and a hint), lower to IR, reference rewrite
  ir/              nodes, interp, range analysis, passes
  backend/         layout, select, sched, flow, emit (asm/lst/bin/json)
  host/            bridge.py (the ESP32 protocol, including W), rig.py, packets.py
  cli.py           python -m opendmp dis | asm | facts   [done]; build | sim | run | check to come
examples/          12 programs (examples/README.md); examples/emu/ = a runnable emulator demo
tests/             37 tests: hardware-transcribed ISA vectors, cycle counts, scheduling, the CLI
```

Implementation style: the core uses only the standard library (`ast`, `dataclasses`,
`tomllib`), with `pyserial` as an optional `[bench]` extra. Python 3.11+, type hints, pytest
and ruff. ISA encodings are tables, not `if` chains. Every rule the code relies on cites its
Finding or probe, as the RE repo does. The tools **refuse rather than guess**: the emulator
raises `Unmodelled` for anything with no measured meaning.

## 6. The hard parts

### 6.1 Chip profiles (measured: ISA2.md Findings 37/38)

| | mpu6050 (mantis) | icm20948 (DMP3) |
|---|---|---|
| data | 0x000–0x3FF, one block; never emit `bc`–`be` (the block persists across passes) | 0x0000–0x0FFF, blocks via `bc`–`be` (reset every pass) |
| code | entry 0x0400, up to 0x0BFF (3 KB of memory in all) | entry 0x1000, **up to 0x2FFF** (8 KB): 0x3000–0x3FFF is the FIFO |
| hardware-written RAM | **0x000–0x00D only**: ax ay az temp gx gy gz as int16 BE, every pass, only while the DMP runs (B2). 0x00E–0x00F are free | **0x0000–0x0011**: 0–3 cleared, then ax ay az gx gy gz temp (FSYNC is temp bit 0); **plus every FIFO byte, into 0x3000–0x3FFF** (B2) |
| boot | **no vendor image** (B1): device reset (it keeps DMP memory), minimal registers, zero all memory, load, `DMP_CFG_1/2` = entry | the same (B1); HW_FIX_DISABLE / SINGLE_FIFO_PRIORITY_SEL not needed |
| flag | **one flag F on both chips**: exact result < 0; `d1` makes it result == 0; `d9` IF F, `da` IF ¬F. **No Z flag** | same |
| control flow | forward branch of **0–63 B** (`f4 41`–`7f` is the interrupt) | 16-bit jump and call (4 deep), `e0`, loops; mode-4 hygiene |
| multiply-accumulate | `f0`; **any destination move restarts the sum** | `f0` too; **only an op&4 pre-advance restarts it** |
| `f2`/`f3` | **16-bit lanes, not just store widths**: `f3` computes on sign-extended low halves (a MUL there is tiny), `f2` on high halves; the same for `f6`/`f7` | same |
| interrupt | `fe` | `e2` |
| clock, cycles (cheap / MOV / ADD / MUL / MAC) | 8.199 MHz: 5 / 18 / 28 / 58 / 60 | 20.975 MHz: 4 / 17 / 26 / 25 / 27 |
| pass start after the tick | 110 cycles (13.4 µs) | 130 cycles (6.2 µs) |
| a sample write inside a pass | stalls it 14–28 cycles | no effect |
| prologue | `d8 f1` | `f1` first (mode 4 survives across passes), then `d8` |
| state kept across passes | pointers, P2 index, route, mode, F; **not** the `fc`/`fd` exchange | pointers, mode 4, F |

### 6.2 Layout and pointer scheduling (the core problem)

Every ALU op needs its A operand on P0's page, its B operand on P1's page and its destination
at P2's walked word index. The index only moves forward: `inc` adds 1, `d0` adds 2, and a page
load resets it to 0. The allocator packs values into four-word pages by operand affinity. Values
that meet as A/B operands go on pages that stay loaded, and destinations are ordered within a
page by first write so that P2 only walks forward. Temporaries are reused by liveness. The first
version is greedy with a peephole pass, measured in **cycles** on the examples; the emulator
reports them exactly. A better optimiser comes only if the examples show it is needed.

The scheduler tracks the whole machine state: the three pointers (block, bank, page), the P2
word index, routing, the mode register, the `fc`/`fd` toggles, `bf`, the predicate, mode 4, and
what the MAC accumulator holds. It emits only the bytes that change state. At a join, state
that differs between the paths becomes unknown and is reloaded. **Width modes are lane modes**:
nothing arithmetic may run under `f2`/`f3` unless a 16-bit lane op is intended.

### 6.3 Instruction selection

| IR | selection |
|---|---|
| `a + b`, `a * b`, `-(a*b)` | ADD / MUL / NMUL; either operand may take either side |
| `a - b` | RSUB (B − A) with `b` in the A slot; `fc` + RSUB when the layout forces it |
| `Σ aᵢ·bᵢ` | `f0` MAC run. It starts fresh on the MPU when the destination moves (a new word or page) and on the ICM when its first op pre-advances (`op&4`); otherwise open with an `f1` MUL |
| copy, emit | MOV when the source is on P0 (18 cycles on the MPU), otherwise ADD with 0.0 (28) |
| `s.accel.x`, `s.gyro.y` | `de`, then `f2` MOV or MOVX: the exact int16 lands in the upper half with a zero lower half |
| 16-bit data | a lane op under `f2`/`f3` is cheaper (22 / 52 cycles on the MPU) but computes on half-words |
| `2*a`, `4*a` | `a + a` with P0 and P1 on the same page |
| `min`, `max`, `abs`, `clamp` | `f5` MIN (beware: MIN returns −1 when `b − a` overflows positively); negation; compare-and-select |
| `a & b`, `a ^ b` (Bits) | `f5` AND / XOR (they leave F alone) |
| `n += 1` (Int) | INC |

### 6.4 Conditions and control flow

Predicates are **sticky**, and **F freezes while a predicate is armed**, on both chips. So the
compiler computes every condition under `d8` and ends every conditional region with `d8`.

- **One lowering for both chips.** F is the sign of the *exact* result, so `cmp a,b` (F :=
  a − b < 0) never overflows. `a < b` is `cmp a,b; if F`, `a >= b` is `if ¬F`, and `a > b` and
  `a <= b` swap the operands. `==` and `!=` use `d1` after an RSUB. There is no need for the
  old "compare halves" trick.
- **MPU.** A branch-skip (`d9 f4 k d8 <then> d8`) nests naturally, because the body runs with no
  predicate armed. A body longer than 63 B uses predication when it holds no inner condition,
  and otherwise a trampoline (`f4 02` over an `f4 3f` hop). A taken branch costs 15 cycles;
  the bytes it jumps over cost nothing.
- **ICM.** After every `jump`/`call` (taken, not taken or skipped), the compiler inserts
  `d8 f1` at the target and on the fall-through. A loop's DEC always runs under `d8`, and the
  static call depth must be ≤ 4. `Int` comparisons with 0..63 use `fc cK Y`.

### 6.5 Packets, timing, range

- The emits, in program order, fix the packet at compile time. The bundle holds a schema, and
  the host runtime returns named fields scaled to floats.
- **Pass time is exact.** Worst-path cycles through `if`/`else` (ICM loops need a bound) plus
  the start latency, from the measured table, then compared with the sample period. Tick stalls
  only matter for passes that overrun the period.
- **Range analysis exploits the hardware.** Add/sub chains are exact modulo 2³², so an
  intermediate wrap is harmless when the final value fits. Compares cannot overflow at all.
  The analysis therefore warns only where a possibly-wrapped value reaches a multiply, a store,
  an emit or MIN, and it names the fix.

## 7. ISA facts the compiler depends on, and the bench work behind them

`isa/facts.py` records every behaviour the backend relies on, with chip, status
(HW / VENDOR / INFER / UNMEASURED) and source (`python -m opendmp facts`): 53 facts, 50 of them
HW. The emulator's `Model` switches (defaults = measured) and `chips.py` (cycle counts with
their probes) are the executable form of the same table.

| id | question | chips | result |
|---|---|---|---|
| B1 | Does a program run with **no vendor image** (RAM zeroed, only our code and data)? | both | **measured: yes** (probe91 B1). The count ran gapless, one pass per sample (10,003 µs at 100 Hz), and 26/26 and 29/29 packets matched the emulator bit for bit. A device reset keeps DMP memory |
| B2 | Clobber map: which RAM does the hardware write? | both | **measured** (probe91 B2/FIFO): MPU 0x000–0x00d; ICM 0x0000–0x0011 plus **the FIFO at 0x3000–0x3fff**. Nothing while the DMP is off. MPU 0x00e's "constant 12" was a tag artefact |
| B3 | What survives from pass to pass? | MPU | **measured**: pointers, P2 index, route, mode, F survive; `fc`/`fd` do not (probe86 C) |
| B4 | What restarts the MAC sum? | both | **measured, and the chips differ**: MPU = any destination move; ICM = op&4 only (probe86 B, probe88) |
| B5 | MAC precision and rounding | both | **measured**: exact products summed, floored once; MUL/NMUL floor (probe86 B, probe88) |
| B6 | DMP3 `f0`, the `f5` family | ICM | **measured**: `f0` is the MAC; `f5`–`f7` as on the MPU (probe88) |
| B7 | Clears under routing and widths | MPU | **measured**: a clear is a routed store of 0 and honours the lane (probe86 A) |
| new | the flags | both | **one exact-sign F, no Z**; AND/XOR leave F alone; MIN sets F := a > b; `d1` tests the last adder result (probe86 E–H, probe88) |
| new | `f2`/`f3`/`f6`/`f7` | both | **16-bit lanes** (probe86 A/E/F, probe88) |
| new | timing | both | **cycle counts, clocks, start latency, tick stalls** (probe90, with the bridge's new `W` command) |

## 8. How it is verified

1. **Emulator against silicon, cell by cell.** probe86 (MPU, 120 cells) and probe88 (ICM, 36
   cells) run every cell on the rig and in the emulator side by side. All match. The stock
   MotionApps20 image emits its 42-byte packets at the measured rate for FIFO-rate dividers
   0–3 (200/100/66.8/50 per second on silicon). From the cycle table alone it reproduces
   probe41's pass rates at all five sample rates, including the non-monotonic 500 Hz dip.
2. **Unit tests** transcribed from the recorded probe results (probes 7/9, 32, 35, 47, 51, 53,
   67–70, 74, 83b, 86, 88, 90, 91), plus cycle sums and scheduling.
3. **The ISA tables.** Every byte value, in every mode and with every operand tail, decodes
   and re-encodes to itself. All 13 vendor images (about 45,000 instructions) disassemble to
   text that reassembles byte for byte. The listing's cycle class equals the emulator's for
   every instruction the emulator executes.
4. **Differential testing** (to come). For every example and test program, the reference model
   (AST rewrite) and the emulator (compiled bytes) must agree bit for bit.
5. **Listing self-check** (to come). Decoding the emitted bytes must reproduce the intended
   instruction stream, and the `.asm` must reassemble to identical bytes.
6. **Silicon** (to come). A *trace build* also emits each pass's raw sample words. The emulator
   replays them and must match the rig's packets bit for bit.

## 9. Phases and milestones

| phase | contents | status |
|---|---|---|
| **1 Emulator** | Q30 model; machine for both chips; pass loop, FIFO, INT; cycle model; file I/O; B3–B7 | **done** 2026-09-23 |
| **0 Ground truth** | ISA tables, encoder, decoder and `dis` for both chips; `facts.py`; **B1, B2** on both rigs | **done** 2026-09-23 (probe91, Finding 39) |
| **2 Front end** | ast subset, errors, IR, reference rewrite, range analysis | next. Examples parse and simulate; a rejected-construct suite gives the documented messages |
| **3 Backends** | layout, selection, scheduler, MPU and ICM control flow, `.lst`/`.asm`/`.bin`/`.json` | **M1 hello-FIFO** and `tilt` build for both chips; emulator equals reference on 10k inputs |
| **4 Runtime and silicon** | `host/` (bridge incl. `W`, whole-program init, params, state readback, packet stream), trace builds | **M2**: `tilt` bit-exact against the emulator on both rigs |
| **5 Depth and polish** | nested control flow and trampolines, ICM loops and `@dmp.sub`, the standard library, the 12 examples on both rigs, tutorial and language reference | **M3/M4** |

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

## 13. Issues found in InvensenseDMP (not fixed there, except as noted)

1. `dmpasm.py` accepts `branch 0x40`–`0x7f`, but the displacement is 6 bits: `f4 41`–`7f`
   raises DMP_INT and `f4 40` does nothing. So `branch 65` silently assembles an interrupt.
   Manual §1.6, §3.6 and §7.1 still say "7-bit, ≤ 127".
2. Manual §2.9 says there is no AND/XOR and no compare other than `== 0`/MIN. This has been
   stale since Finding 29.
3. `dmpasm.py` has no mantis mnemonic for `de` (clear in place).
4. **Superseded by Findings 37/38 (recorded in ISA2.md; the manual chapters are not yet
   rewritten):**
   - the MPU's "N and Z, d9 = IF (N ∨ Z)";
   - "f2/f3 are store widths";
   - "f0 on DMP3 is unmeasured";
   - the 0.58/2.0/3.9/7.9 µs + 56 µs timing model;
   - Finding 28's "d9 fires on 0", which was the op's own zero emit read as the marker.
5. **Superseded by Finding 39** (manual chapters 6–8 corrected): MPU "0x00e reads a constant
   12" (probe84's own tag), and the ICM's "free RAM to 0x3fff" (the FIFO lives at
   0x3000–0x3fff).
6. `dmpasm.py`'s mnemonics predate Findings 37–39 in places: `if le0`/`if ge0` on mantis (it is
   `if f`/`if nf`), `mov16` (MOVX), the ALU op under `f0` spelled `add` (it is a MAC), `disable`
   (end of pass). opendmp's assembler uses the corrected names and accepts `disable`/`mov16` as
   aliases. The two syntaxes produce identical bytes for `examples/emu/tilt`.
