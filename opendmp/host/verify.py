"""Run a program on silicon and check it against the emulator and the reference, bit for bit.

The program is built as a trace build: every packet starts with the raw sample words of its
pass.  The samples are replayed through the same bytes in the emulator (which must give the
whole packet, the sample words included), and through the reference semantics (which must give
the program's own fields).
"""
from dataclasses import dataclass, field

from .rig import Rig


@dataclass
class Verification:
    program: str
    chip: str
    passes: int
    rate: float
    interrupts: int
    emu_interrupts: int
    emu_mismatch: list = field(default_factory=list)     # pass indices
    ref_mismatch: list = field(default_factory=list)
    cycles: tuple = (0, 0)
    fields: list = field(default_factory=list)            # decoded, per pass

    @property
    def ok(self):
        return self.passes > 0 and not self.emu_mismatch and not self.ref_mismatch and \
            abs(self.interrupts - self.emu_interrupts) <= 1

    def summary(self):
        verdict = "BIT-EXACT" if self.ok else "MISMATCH"
        return (f"{self.program} on the {self.chip}: {self.passes} passes at {self.rate:.1f} Hz; "
                f"emulator {len(self.emu_mismatch)} and reference {len(self.ref_mismatch)} packets differ; "
                f"interrupts {self.interrupts} on silicon, {self.emu_interrupts} emulated; "
                f"{self.cycles[0]}-{self.cycles[1]} cycles a pass: {verdict}")


def verify(program, port: str, seconds: float = 3.0, rate: float | None = None, rig: Rig | None = None):
    own = rig is None
    rig = rig or Rig(port)
    try:
        b = program.build(rig.chip, trace=True)
        rig.load(b, rate)
        packets, ints = rig.run(seconds)
    finally:
        if own:
            rig.close()
    samples = [b.trace_sample(p) for p in packets]
    emu = b.emulate(samples)
    ref = program.simulate(samples)
    t = b.trace_bytes
    v = Verification(program.name, rig.chip, len(packets), rig.rate, ints, sum(emu.irqs))
    v.emu_mismatch = [i for i, (h, e) in enumerate(zip(packets, emu.packets)) if h != e]
    v.ref_mismatch = [i for i, (h, r) in enumerate(zip(packets, ref.packets)) if h[t:] != r]
    v.cycles = (min(emu.cycles), max(emu.cycles)) if emu.cycles else (0, 0)
    from ..sim import decode
    v.fields = [decode(program.ir.packet, p[t:]) for p in packets]
    v.hw_packets, v.emu_packets = packets, emu.packets
    return v
