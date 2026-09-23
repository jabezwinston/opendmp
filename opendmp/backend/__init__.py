"""The backends: the IR -> machine code for the MPU-6050 and the ICM-20948.

    tac.py      instruction selection: IR -> three-address code over abstract words
    layout.py   words -> RAM addresses
    sched.py    TAC -> instructions, tracking pointers, P2's index, mode, route, MAC restarts
    build.py    assembly with branch relaxation; the image, listing, .asm, packet schema
"""
from .build import Build, build

__all__ = ["Build", "build"]
