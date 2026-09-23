"""Simulate a program on sensor samples: `prog.simulate(samples)`.

Two engines give the same bytes: "reference" evaluates the source (and also counts actual
wraps and failed assumptions), "ir" runs what the front end produced.
"""
from collections import Counter
from dataclasses import dataclass, field

from .ir import nodes as N
from .q30 import ONE, s32


@dataclass
class SimResult:
    program: str
    engine: str
    packet: list                                  # [ir.Field]
    packets: list = field(default_factory=list)   # bytes per pass
    irqs: list = field(default_factory=list)      # bool per pass
    wraps: Counter = field(default_factory=Counter)          # Loc -> passes it wrapped in
    assume_fails: Counter = field(default_factory=Counter)   # Loc -> passes it failed in
    cycles: list = field(default_factory=list)    # per pass, from the emulator

    @property
    def passes(self):
        return len(self.packets)

    def fields(self) -> list[dict]:
        return [decode(self.packet, p) for p in self.packets]

    def column(self, name) -> list:
        return [decode(self.packet, p)[name] for p in self.packets]

    def summary(self) -> str:
        lines = [f"{self.program}: {self.passes} passes ({self.engine}), {sum(self.irqs)} interrupts, "
                 f"packet {sum(f.width for f in self.packet)} B"]
        if self.cycles:
            from .emu.chips import get_chip
            ns = get_chip(self.engine.split(":")[1]).cycle_ns
            lines.append(f"  pass time   {min(self.cycles)}-{max(self.cycles)} cycles = "
                         f"{min(self.cycles) * ns / 1000:.1f}-{max(self.cycles) * ns / 1000:.1f} us")
        cols = {f.name: self.column(f.name) for f in self.packet} if self.packets else {}
        for f in self.packet:
            c = cols[f.name]
            lines.append(f"  {f.name:12s} {f.type:4s}  min {min(c):>12.6g}  max {max(c):>12.6g}  "
                         f"last {c[-1]:>12.6g}")
        for loc, n in sorted(self.wraps.items(), key=lambda kv: (kv[0].line, kv[0].col)):
            lines.append(f"  {loc}: wrapped in {n} passes")
        for loc, n in sorted(self.assume_fails.items(), key=lambda kv: (kv[0].line, kv[0].col)):
            lines.append(f"  {loc}: dmp.assume failed in {n} passes")
        return "\n".join(lines)


def decode(packet, data: bytes) -> dict:
    out = {}
    for f in packet:
        b = data[f.offset:f.offset + f.width]
        if f.type == N.S16:
            v = int.from_bytes(b, "big", signed=True)
            out[f.name] = v / 16384                    # raw / 16384: the Q30 meaning
        else:
            v = s32(int.from_bytes(b, "big"))
            out[f.name] = v / ONE if f.type == N.Q30 else v
    return out


def simulate(program, samples, engine="reference", params: dict | None = None) -> SimResult:
    ir = program.ir
    if engine.startswith("emu:"):
        return program.build(engine[4:]).emulate(samples, params)
    if engine == "reference":
        from .lang.reference import Reference
        m = Reference(program, ir)
    elif engine == "ir":
        from .ir.interp import Interp
        m = Interp(ir)
    else:
        raise ValueError(f"unknown engine {engine!r} (reference, ir, emu:mpu6050, emu:icm20948)")
    from .q30 import to_q30
    for k, v in (params or {}).items():
        m.set_param(k, to_q30(v) if isinstance(v, float) else v)
    res = SimResult(program.name, engine, ir.packet)
    for s in samples:
        pk, irq = m.run_pass(s)
        res.packets.append(pk)
        res.irqs.append(irq)
    res.wraps = getattr(m, "wraps", Counter())
    res.assume_fails = m.assume_fails
    return res
