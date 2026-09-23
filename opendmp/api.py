"""The names a DMP program uses, all reached as `dmp.<name>` after `import opendmp as dmp`.

A program's body is never run by Python: `@dmp.program` records the function, and the front
end reads its source.  So `dmp.emit`, `dmp.irq`, `dmp.clamp` and `dmp.assume` exist for the
reader and the editor; calling them from ordinary Python raises.
"""
from functools import cached_property

GYRO_FS = (250, 500, 1000, 2000)          # deg/s
ACCEL_FS = (2, 4, 8, 16)                  # g


class _Decl:
    """`dmp.state` / `dmp.param`, bare or with an assumed range: `dmp.state(-1.5, 1.5)`."""

    def __init__(self, kind, lo=None, hi=None):
        self.kind, self.lo, self.hi = kind, lo, hi

    def __call__(self, lo, hi):
        return _Decl(self.kind, lo, hi)

    def __repr__(self):
        return f"dmp.{self.kind}" + ("" if self.lo is None else f"({self.lo}, {self.hi})")


state = _Decl("state")
param = _Decl("param")


class Bits:
    """A 32-bit mask constant: `dmp.bits(0xffff0000)`.  Supports & ^ | ~ << >>."""
    __slots__ = ("value",)

    def __init__(self, value: int):
        self.value = int(value) & 0xFFFFFFFF

    def __eq__(self, other):
        return isinstance(other, Bits) and other.value == self.value

    def __hash__(self):
        return hash(("bits", self.value))

    def __repr__(self):
        return f"dmp.bits({self.value:#010x})"


def bits(value: int) -> Bits:
    return Bits(value)


def loop(n: int):
    """`for _ in dmp.loop(N):` a loop of at most N passes of its body, left with `break`."""
    return range(n)


def _dmp_only(name):
    def f(*a, **k):
        raise RuntimeError(f"dmp.{name} is part of a DMP program: the body of a @dmp.program is "
                           "compiled, not run by Python (use prog.simulate(samples))")
    f.__name__ = name
    f.__qualname__ = f"dmp.{name}"
    return f


emit = _dmp_only("emit")
irq = _dmp_only("irq")
clamp = _dmp_only("clamp")
assume = _dmp_only("assume")


class Sub:
    """`@dmp.sub`: a helper that is a real subroutine on the ICM-20948 (inlined on the MPU)."""

    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def __call__(self, *a, **k):
        raise RuntimeError(f"{self.__name__} is a @dmp.sub: call it from a DMP program")

    def __repr__(self):
        return f"<dmp.sub {self.__name__}>"


def sub(fn):
    return Sub(fn)


class Program:
    """A compiled-on-demand DMP program: `@dmp.program(rate=200)` on a function of one
    argument, the sensor sample."""

    def __init__(self, fn, rate: float, gyro_fs: int = 2000, accel_fs: int = 2, name: str | None = None):
        if not rate or rate <= 0:
            raise ValueError("@dmp.program needs rate=<samples per second>")
        if gyro_fs not in GYRO_FS:
            raise ValueError(f"gyro_fs must be one of {GYRO_FS} (deg/s)")
        if accel_fs not in ACCEL_FS:
            raise ValueError(f"accel_fs must be one of {ACCEL_FS} (g)")
        self.fn = fn
        self.name = name or fn.__name__
        self.opts = {"rate": rate, "gyro_fs": gyro_fs, "accel_fs": accel_fs}
        self.__doc__ = fn.__doc__

    def __call__(self, *a, **k):
        raise TypeError(f"{self.name} is a DMP program: it is not called; use "
                        f"{self.name}.simulate(samples), or build it for a chip")

    def __repr__(self):
        return f"<dmp.program {self.name} @ {self.opts['rate']:g} Hz>"

    @cached_property
    def ir(self):
        """The program lowered to the IR (raises DmpError with a location if it is not valid)."""
        from .lang.lower import lower
        return lower(self)

    def check(self):
        """Range analysis: a list of warnings (empty when nothing can wrap)."""
        from .ir.ranges import analyse
        return analyse(self.ir).warnings

    def build(self, chip: str, trace: bool = False):
        """Compile for a chip ("mpu6050" or "icm20948"): an opendmp.backend.Build (image, listing,
        .asm, packet schema; .emulate(samples) runs the bytes).  A trace build also sends each
        pass's raw sample, so a run on silicon can be replayed in the emulator."""
        from .backend import build
        from .isa.core import chip_name
        cache = self.__dict__.setdefault("_builds", {})
        key = (chip_name(chip), trace)
        if key not in cache:
            cache[key] = build(self, key[0], trace)
        return cache[key]

    def simulate(self, samples, engine: str = "reference", params: dict | None = None):
        """Run the program on raw sensor samples (opendmp.emu.Sample), one pass each.

        engine="reference" evaluates the source directly (the semantic oracle); engine="ir"
        interprets the lowered IR; engine="emu:mpu6050" / "emu:icm20948" builds the program and
        runs its bytes in the emulator.  All must agree bit for bit.  params sets dmp.param values
        before the first pass ({"limit": 0.3})."""
        from .sim import simulate
        return simulate(self, samples, engine, params)


def program(rate: float = None, gyro_fs: int = 2000, accel_fs: int = 2, name: str | None = None):
    """`@dmp.program(rate=200)`: the function runs once per sensor sample on the DMP."""
    if callable(rate):
        raise TypeError("write @dmp.program(rate=...), with the sample rate")

    def deco(fn):
        return Program(fn, rate, gyro_fs, accel_fs, name)
    return deco
