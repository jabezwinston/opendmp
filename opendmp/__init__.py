"""opendmp: a Python-like language, compiler and emulator for the InvenSense DMP.

    import opendmp as dmp

    @dmp.program(rate=200)
    def tilt(s):
        angle: dmp.state = 0.0
        angle = 0.98 * (angle + s.gyro.y * 0.0873) + 0.02 * s.accel.x
        dmp.emit(angle)

    tilt.simulate(samples)          # the reference semantics, pass by pass
    tilt.check()                    # range warnings

Everything a program uses is `dmp.<name>`.  The emulator is `opendmp.emu`; the instruction
sets are `opendmp.isa`.  See PLAN.md for what is built and what is next.
"""
__version__ = "0.0.1"

from .api import (Bits, Program, Sub, assume, bits, clamp, emit, irq, loop, param, program,  # noqa: E402
                  state, sub)
from . import quat  # noqa: E402
from .lib import div  # noqa: E402

__all__ = ["Bits", "Program", "Sub", "assume", "bits", "clamp", "emit", "irq", "loop", "param",
           "program", "state", "sub", "quat", "div"]
