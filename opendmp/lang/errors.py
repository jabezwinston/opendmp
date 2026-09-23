"""Errors and warnings with a location, the source line, a caret and a hint.

    tilt.py:14:17: error: division by a variable: the DMP has no divide
        angle = x / y
                    ^
    hint: multiply by a constant reciprocal, or use a Newton step (examples/unit_vectors.py)
"""
import ast
import linecache
import pathlib
from dataclasses import dataclass


def library_file(file: str) -> bool:
    """A file of opendmp itself (dmp.div, dmp.quat): its inner workings are not the user's."""
    return pathlib.Path(file).resolve().parent == pathlib.Path(__file__).resolve().parent.parent


def snippet(loc, width: int = 48) -> str:
    """The source text a location covers, on one line, shortened to `width`."""
    if loc is None:
        return ""
    line = linecache.getline(loc.file, loc.line).rstrip("\n")
    text = line[loc.col:loc.end_col] if loc.end_line == loc.line else line[loc.col:].rstrip() + " ..."
    if "#" in text and "'" not in text and '"' not in text:     # a comment, not the code
        text = text[:text.index("#")].rstrip() + (" ..." if loc.end_line != loc.line else "")
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 4] + " ..."


@dataclass(frozen=True)
class Loc:
    file: str
    line: int
    col: int                 # 0-based, as ast gives it
    end_line: int
    end_col: int

    @classmethod
    def of(cls, node, file: str) -> "Loc | None":
        if node is None or not hasattr(node, "lineno"):
            return None
        return cls(file, node.lineno, node.col_offset,
                   getattr(node, "end_lineno", None) or node.lineno,
                   getattr(node, "end_col_offset", None) or node.col_offset + 1)

    def __str__(self):
        return f"{pathlib.Path(self.file).name}:{self.line}:{self.col + 1}"

    def excerpt(self) -> str:
        text = linecache.getline(self.file, self.line).rstrip("\n")
        if not text:
            return ""
        end = self.end_col if self.end_line == self.line else len(text)
        width = max(1, end - self.col)
        return f"    {text}\n    {' ' * self.col}{'^' * width}"


def _format(kind: str, msg: str, loc: Loc | None, hint: str | None) -> str:
    head = f"{loc}: {kind}: {msg}" if loc else f"{kind}: {msg}"
    parts = [head]
    if loc:
        ex = loc.excerpt()
        if ex:
            parts.append(ex)
    if hint:
        parts.append(f"hint: {hint}")
    return "\n".join(parts)


class DmpError(Exception):
    """A program the front end refuses.  str() is the full, formatted message."""

    def __init__(self, msg: str, loc: Loc | None = None, hint: str | None = None):
        self.msg, self.loc, self.hint = msg, loc, hint
        super().__init__(_format("error", msg, loc, hint))


@dataclass(frozen=True)
class Warning_:
    msg: str
    loc: Loc | None = None
    hint: str | None = None

    def __str__(self):
        return _format("warning", self.msg, self.loc, self.hint)


def node_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:                       # pragma: no cover
        return type(node).__name__
