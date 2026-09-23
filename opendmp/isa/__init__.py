"""The DMP instruction sets of the MPU-6050 (mantis) and the ICM-20948 (DMP3), as tables.

    from opendmp.isa import Insn, encode, decode, assemble, disassemble
    encode(Insn("add", (0, 1, False)), "mpu6050")          # b'\\x08'
    prog = assemble(open("tilt.asm").read())                 # .chip / .org / labels / .dw
    for line in disassemble(code, "icm20948", org=0x1000): print(line.addr, line.insn)

`facts` records every machine behaviour the tools rely on, with its evidence.
"""
from .asm import AsmError, Line, Program, assemble, disassemble, to_text
from .core import ALU_OPS, ICM, MPU, EncodeError, Insn, chip_name, cost_key, decode, effect, encode, \
    length, next_mode

__all__ = ["AsmError", "Line", "Program", "assemble", "disassemble", "to_text", "ALU_OPS", "ICM",
           "MPU", "EncodeError", "Insn", "chip_name", "cost_key", "decode", "effect", "encode", "length",
           "next_mode"]
