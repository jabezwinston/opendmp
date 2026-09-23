# opendmp

A Python-like language, compiler and **cycle-accurate emulator** for the InvenSense DMP, the
motion processor inside the MPU-6050 and the ICM-20948. The plan of record is
[PLAN.md](PLAN.md). The language is specified by example in [examples/](examples/). The ISA
knowledge comes from the sibling reverse-engineering repo, `~/Git_repos/InvensenseDMP`.

**Today:** the emulator and the ISA tools (assembler, disassembler, the machine facts). The
compiler is next (PLAN.md §9). A program needs **no vendor firmware**: it runs from zeroed
memory on both parts, as measured on the bench (below).

## The emulator

```bash
python -m opendmp.emu run.toml                   # or everything as options, see --help
python -m opendmp.emu --costs --chip icm20948    # the cycle table and where each number comes from
python -m opendmp.emu.gen walk,taps --rate 200 -o data.csv    # synthetic sensor data
```

It runs DMP machine code for either chip, one pass per sensor sample, and takes everything
from files:

| input | format |
|---|---|
| sensor data | CSV, one row per sample tick; header names any of `ax ay az gx gy gz temp fsync` (or `accel_x` …). Raw int16 counts, or `units = "si"` for g, deg/s and deg C at the given full scales |
| program and RAM | `load = ["prog.asm"]`: opendmp assembly, which places its own code and data. Or `"prog.bin@0x400"`: binary, `.hex` text, or `.s` assembled with InvensenseDMP's `dmpasm.py`. Load a whole RAM image the same way |
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
| predicate / mode / pointer / skipped byte | 5 | 4 |
| MOV | 18 | 17 |
| ADD, RSUB, MIN | 28 | 26 |
| MUL | 58 | 25 |
| MAC | 60 | 27 |
| taken branch / jump | 15 | 13 |
| sample tick → first instruction | 110 | 130 |

`--costs` prints the full table, with the probe behind each number and which classes are still
unmeasured (they fall back to a measured class and say so). The scheduler models what the
sample clock does to a pass that overruns:
- **MPU:** it drops to an integer submultiple of the rate, and every sample write inside a
  pass stalls it about 20 cycles.
- **ICM:** it runs at 1/pass time.

The clock is the bench parts' own; `--clock-mhz` takes yours.

### How it was validated

- **Semantics.** 156 cells (probe86 on the MPU-6050, probe88 on the ICM-20948) run on silicon
  and in the emulator side by side; all match.
- **The vendor's firmware.** The stock MotionApps20 image (1929 B) emits its 42-byte packets at
  exactly the rates silicon does for FIFO-rate dividers 0–3. Its pass (20,465 cycles)
  reproduces the measured pass rates at five sample rates, including the non-monotonic dip at
  500 Hz.
- **Unit tests** (`pytest`) transcribe the recorded hardware results.

Whatever has no measured meaning raises `Unmodelled` instead of being guessed: MPU `bf`,
DMP3 `fc X Y` with X < 0xc0, both operand exchanges at once. The few behaviours that are
unmeasured but unavoidable are named switches in `Model` (`--model key=value`), with the
measured behaviour as the default.

### Machine facts the emulator encodes that the manual does not (yet)

Measured for this project on 2026-09-23 (ISA2.md Findings 37–39):
- **One flag on both chips.** F is the sign of the **exact** result, so compares never
  overflow. There is **no Z**: `d9` = IF F, `da` = IF ¬F, `d1` makes F = (result == 0).
- **`f2`/`f3` are 16-bit lane modes, not just store widths.** Under `f3` the ALU computes on the
  sign-extended low halves, so a MUL there is `(lo·lo) >> 30`. Under `f2` it computes on the
  high halves.
- **MIN** sets F := a > b and returns −1 when `b − a` overflows positively. **AND/XOR** leave F
  alone.
- **The MAC restarts** on any destination move on the MPU, but only on an `op&4` pre-advance on
  the ICM. Products are summed exactly and floored once. **DMP3's `f0` is the MAC.**
- **Across passes**, pointers, the P2 index, route, mode and F survive; the `fc`/`fd`
  exchanges do not.
- **No vendor image is needed** (Finding 39). The hardware writes only the sample page:
  0x000–0x00d on the MPU, and 0x0000–0x0011 on the ICM. **The ICM's FIFO is DMP RAM
  0x3000–0x3fff**, so code must end below 0x3000; the emulator writes FIFO bytes there too.
  A device reset keeps DMP memory.

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
- `fmode 5` then `min`/`cmp`/`and`/`xor`;
- `end`.

The assembler tracks the mode register and refuses code the hardware would read differently:
- an ALU op from the wrong family (`min` under `fmode 1`);
- any ALU op while DMP3's mode 4 is set;
- an op at a jump target that different paths reach in different modes;
- a branch outside the MPU's 0–63 forward bytes.

```asm
        .chip   mpu6050
        .org    0x100
const:  .dw     0.0873q, 1.0q, 0.98q, 0.02q
        .org    0x400                   ; .entry defaults to 0x400 (MPU), 0x1000 (ICM)
        always
        fmode   5
        cmp     3,0                     ; F := P0.3 - P1.0 < 0
        if f
        branch  join
        always
        int
join:   always
        end
```

## Layout

```
opendmp/q30.py         the number model
opendmp/isa/           both ISAs as tables: encode/decode, the assembler, facts.py
opendmp/cli.py         python -m opendmp dis | asm | facts
opendmp/emu/           the emulator (machine, chips, sched, files, gen, disasm, CLI)
examples/              12 programs in the opendmp language; examples/emu/ runs today
tests/                 pytest: hardware-transcribed vectors, cycles, scheduling, the ISA round trips, CLIs
PLAN.md                the plan of record
```

Tests: `python -m pytest`. The corpus-based tests look for `../InvensenseDMP` and skip
without it; no vendor firmware is stored here.
