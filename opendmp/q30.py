"""Q30 fixed point: the DMP's number format.

A word is 32-bit two's complement, big-endian in memory, value = int32 / 2**30, so the range is
[-2, 2) with a resolution of 2**-30.  Arithmetic wraps; it never saturates.
"""
ONE = 1 << 30
MASK = 0xFFFFFFFF
LO30 = ONE - 1


def s32(v: int) -> int:
    """Wrap an integer to signed 32 bits."""
    v &= MASK
    return v - (1 << 32) if v & 0x80000000 else v


def sx30(v: int) -> int:
    """Sign-extend a 30-bit field from bit 29 (R's low word, `lo`: manual chapter 5)."""
    v &= LO30
    return v - ONE if v & (1 << 29) else v


def to_q30(x: float) -> int:
    """Nearest Q30 word to x, as a signed int.  Raises for values outside [-2, 2)."""
    r = round(x * ONE)
    if not -(1 << 31) <= r < (1 << 31):
        raise ValueError(f"{x} is outside the Q30 range [-2, 2)")
    return r


def from_q30(w: int) -> float:
    return s32(w) / ONE


def word_bytes(w: int) -> bytes:
    return (w & MASK).to_bytes(4, "big")


def parse_word(text: str) -> int:
    """'40000000' or '0x40000000' -> raw word; '1.0q' or '-0.25q' -> Q30; '100i' or '-3i' -> a
    decimal integer."""
    t = text.strip().lower()
    if t.endswith("q"):
        return to_q30(float(t[:-1])) & MASK
    if t.endswith("i"):                         # 100i, -3i, 0x100i: an integer, decimal unless prefixed
        body = t[:-1]
        return int(body, 0 if body.lstrip("+-").startswith(("0x", "0b", "0o")) else 10) & MASK
    return int(t, 16) & MASK


def sx16(v: int) -> int:
    """Sign-extend the low half-word."""
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v
