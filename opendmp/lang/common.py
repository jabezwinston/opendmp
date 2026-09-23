"""What the lowering and the reference evaluator share: the meaning of Python objects a program
names (constants, modules, helpers, the dmp API) and constant folding.  Neither is DMP
semantics -- constants are Python values -- so sharing them keeps both oracles on one reading
of the source without making them share the arithmetic."""
import ast
import builtins
import types

from .. import api
from .errors import node_text

SENSOR_GROUPS = {"accel": "a", "gyro": "g"}
FIELD_NAME = {"ax": "accel_x", "ay": "accel_y", "az": "accel_z", "gx": "gyro_x", "gy": "gyro_y",
              "gz": "gyro_z", "temp": "temp", "fsync": "fsync"}
CMP_OPS = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge", ast.Eq: "eq", ast.NotEq: "ne"}
CONST_BUILTINS = {"abs", "min", "max", "int", "float", "round", "len"}
API_NAMES = {"emit", "irq", "clamp", "assume", "bits", "loop"}
API_OBJECTS = {id(getattr(api, n)): n for n in API_NAMES}


class SensorNS:
    """The program's sample parameter, `s`."""


class SensorGroup:
    def __init__(self, prefix):
        self.prefix = prefix


class DmpNS:
    """The `opendmp` module as a DMP program sees it."""


class LoopVar:
    """The target of `for _ in dmp.loop(N)`: it has no value."""


class ApiFn:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, ApiFn) and other.name == self.name

    def __hash__(self):
        return hash(("api", self.name))


class Builtin:
    def __init__(self, name):
        self.name = name


def is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_const(v) -> bool:
    return is_number(v) or isinstance(v, api.Bits)


def is_math(f) -> bool:
    return isinstance(f, (types.BuiltinFunctionType, types.FunctionType)) and \
        getattr(f, "__module__", None) == "math"


def classify(v, what, err):
    """A Python object named by a program -> what it means there.  err(msg, hint) raises."""
    if isinstance(v, bool):
        raise err(f"{what} is a bool", "use 1 and 0")
    if is_const(v):
        return v
    if isinstance(v, (list, tuple)):
        return [classify(x, f"{what}[{i}]", err) for i, x in enumerate(v)]
    if isinstance(v, types.ModuleType):
        return DmpNS() if v.__name__ == "opendmp" else v
    if isinstance(v, api.Sub):
        return v
    if isinstance(v, api.Program):
        raise err(f"{what} is another DMP program; a program cannot use one", None)
    if isinstance(v, api._Decl):
        raise err(f"dmp.{v.kind} belongs in a declaration: `x: dmp.{v.kind} = 0.0`", None)
    if id(v) in API_OBJECTS:
        return ApiFn(API_OBJECTS[id(v)])
    if is_math(v) or isinstance(v, types.FunctionType):
        return v
    raise err(f"{what} ({type(v).__name__}) is not usable in a DMP program", None)


def global_name(name, g, err):
    if name in g:
        return classify(g[name], name, err)
    if hasattr(builtins, name):
        if name in CONST_BUILTINS or name == "range":
            return Builtin(name)
        raise err(f"{name} is not available on the DMP", None)
    raise err(f"unknown name {name!r}", None)


def module_attr(base, attr, err):
    """dmp.<attr> or <module>.<attr>."""
    if isinstance(base, DmpNS):
        if attr in API_NAMES:
            return ApiFn(attr)
        if attr == "quat":
            from .. import quat
            return quat
        if attr == "div":                   # the standard library, written in the language
            from .. import lib
            return getattr(lib, attr)
        if attr in ("state", "param"):
            raise err(f"dmp.{attr} belongs in a declaration: `x: dmp.{attr} = 0.0`", None)
        if attr in ("sub", "program"):
            raise err(f"dmp.{attr} is a decorator", None)
        raise err(f"dmp has no {attr!r}", None)
    if not hasattr(base, attr):
        raise err(f"{base.__name__} has no {attr!r}", None)
    return classify(getattr(base, attr), f"{base.__name__}.{attr}", err)


class Folder:
    """Which expressions are compile-time constants, and their values.

    `resolve(node)` gives the object a side-effect-free path (a name, attributes and
    constant-index subscripts of one) stands for, or raises; `err(msg, node)` makes an error."""

    def __init__(self, resolve, err, cache: dict | None = None):
        self.resolve, self.err = resolve, err
        self.cache = cache        # node -> is_static, when the caller can invalidate it

    def pure_path(self, node) -> bool:
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            if isinstance(node, ast.Subscript) and not self.is_static(node.slice):
                return False
            node = node.value
        return isinstance(node, ast.Name)

    def _res(self, node):
        try:
            return self.resolve(node)
        except Exception:
            return None

    def callee(self, func):
        """'math' / Builtin / 'bits' for a constant-foldable callee, else None."""
        if not self.pure_path(func):
            return None
        f = self._res(func)
        if is_math(f):
            return f
        if isinstance(f, Builtin) and f.name in CONST_BUILTINS - {"len"}:
            return f
        if f == ApiFn("bits"):
            return f
        return None

    def is_static(self, node) -> bool:
        if self.cache is None:
            return self._static(node)
        r = self.cache.get(node)             # keyed by the node itself: an id could be reused
        if r is None:
            r = self.cache[node] = self._static(node)
        return r

    def _static(self, node) -> bool:
        if isinstance(node, ast.Constant):
            return is_number(node.value)
        if isinstance(node, (ast.Name, ast.Attribute)):
            return self.pure_path(node) and is_const(self._res(node))
        if isinstance(node, ast.Subscript):
            if not (self.is_static(node.slice) and self.pure_path(node.value)):
                return False
            base = self._res(node.value)
            return isinstance(base, list) and all(is_const(x) for x in base)
        if isinstance(node, ast.BinOp):
            return self.is_static(node.left) and self.is_static(node.right)
        if isinstance(node, ast.UnaryOp):
            return not isinstance(node.op, ast.Not) and self.is_static(node.operand)
        if isinstance(node, ast.Call):
            return self.callee(node.func) is not None and not node.keywords and \
                all(self.is_static(a) for a in node.args)
        return False

    def fold(self, node):
        """Evaluate a static expression in Python."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Name, ast.Attribute)):
            return self.resolve(node)
        if isinstance(node, ast.Subscript):
            return self.resolve(node.value)[self.fold(node.slice)]
        if isinstance(node, ast.UnaryOp):
            v = self.fold(node.operand)
            if isinstance(node.op, ast.USub):
                if not is_number(v):
                    raise self.err(f"{node_text(node)}: dmp.bits do not negate", node)
                return -v
            if isinstance(node.op, ast.UAdd):
                return v
            if isinstance(node.op, ast.Invert):
                return api.Bits(~(v.value if isinstance(v, api.Bits) else v))
        if isinstance(node, ast.BinOp):
            a, b = self.fold(node.left), self.fold(node.right)
            bits = isinstance(a, api.Bits) or isinstance(b, api.Bits)
            ra = a.value if isinstance(a, api.Bits) else a
            rb = b.value if isinstance(b, api.Bits) else b
            ops = {ast.Add: lambda: ra + rb, ast.Sub: lambda: ra - rb, ast.Mult: lambda: ra * rb,
                   ast.Div: lambda: ra / rb, ast.FloorDiv: lambda: ra // rb, ast.Mod: lambda: ra % rb,
                   ast.Pow: lambda: ra ** rb, ast.LShift: lambda: ra << rb, ast.RShift: lambda: ra >> rb,
                   ast.BitAnd: lambda: ra & rb, ast.BitOr: lambda: ra | rb, ast.BitXor: lambda: ra ^ rb}
            if type(node.op) not in ops:
                raise self.err(f"{node_text(node)} is not a constant expression", node)
            try:
                r = ops[type(node.op)]()
            except (ZeroDivisionError, OverflowError, ValueError, TypeError) as e:
                raise self.err(f"{node_text(node)}: {e}", node) from None
            if isinstance(r, complex):
                raise self.err(f"{node_text(node)} is complex", node)
            return api.Bits(r) if bits else r
        if isinstance(node, ast.Call):
            f = self.callee(node.func)
            raw = [x.value if isinstance(x, api.Bits) else x for x in (self.fold(a) for a in node.args)]
            try:
                if f == ApiFn("bits"):
                    return api.Bits(raw[0])
                if isinstance(f, Builtin):
                    return getattr(builtins, f.name)(*raw)
                return f(*raw)
            except (TypeError, ValueError, OverflowError, IndexError) as e:
                raise self.err(f"{node_text(node)}: {e}", node) from None
        raise self.err(f"{node_text(node)} is not a constant expression", node)
