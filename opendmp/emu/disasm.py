"""One-instruction disassembly for traces, from the shared ISA tables (opendmp.isa).  Mode-aware:
the same ALU byte is `add` under f1, `mac` under f0 and `min` under f5, and on DMP3 a byte below
0x80 is a jump target while mode 4 is set."""
from ..isa.core import decode


def disasm(machine, pc: int) -> str:
    mode = machine.mode
    if machine.icm and mode == 0 and machine.model.icm_f0 == "width":
        mode = 1                                    # the Model switch that reads DMP3 f0 as a width
    insn, _ = decode(machine.mem, pc, machine.chip.name, mode)
    return insn.text()
