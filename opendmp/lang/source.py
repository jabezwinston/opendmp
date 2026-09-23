"""Where a function's code is: its file, its `ast.FunctionDef`, and the module namespace its
names resolve in.  The body of a DMP program or helper is never run by Python; it is read from
here."""
import ast
import inspect
import linecache
from dataclasses import dataclass
from functools import lru_cache

from .errors import DmpError


@dataclass(eq=False)
class FuncSource:
    fn: object               # the Python function object
    file: str
    node: ast.FunctionDef
    globals: dict

    @property
    def name(self) -> str:
        return self.node.name


@lru_cache(maxsize=64)
def _module_ast(file: str, text: str) -> ast.Module:
    return ast.parse(text, file)


def namespace(fn) -> dict:
    """The names fn's body can see: its module's globals, and the variables it closes over
    (so a program defined inside a function can use that function's locals)."""
    ns = dict(fn.__globals__)
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            ns[name] = cell.cell_contents
        except ValueError:                      # an empty cell: not assigned yet
            pass
    return ns


def function_source(fn) -> FuncSource:
    """Find `fn`'s definition in its source file."""
    code = getattr(fn, "__code__", None)
    if code is None:
        raise DmpError(f"{fn!r} is not a Python function")
    file = code.co_filename
    linecache.checkcache(file)
    lines = linecache.getlines(file)
    if not lines:
        raise DmpError(f"cannot read the source of {fn.__name__} ({file}); DMP programs and helpers "
                       "must be defined in a .py file")
    tree = _module_ast(file, "".join(lines))
    first = code.co_firstlineno
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn.__name__:
            start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            if start == first or node.lineno == first:
                if isinstance(node, ast.AsyncFunctionDef):
                    raise DmpError(f"{fn.__name__} is async: DMP code is not")
                return FuncSource(fn, file, node, namespace(fn))
    try:
        src = inspect.getsource(fn)
    except OSError:
        src = None
    raise DmpError(f"cannot find the definition of {fn.__name__} in {file}"
                   + ("" if src else " (was it defined with exec or in a REPL?)"))
