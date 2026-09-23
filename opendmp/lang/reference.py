"""The reference semantics: evaluate a DMP program's source directly, pass by pass.

This is the oracle the compiler is checked against.  It shares only the reading of the source
with the front end (where a function is, what a module-level name means, constant folding,
all in `common`), not the arithmetic, the flattening of sums, control flow, inlining or lists:
those are implemented here a second time, from the rules in PLAN.md section 4.

The machine's arithmetic:
  a Sum (a whole +/- expression, with integer multiples and two-factor products) is exact,
  then floored and wrapped once; min/max/compares are exact; abs(-2.0) wraps to -2.0.

It also records what the IR interpreter cannot see: every place a value actually wrapped
(`wraps`), and every `dmp.assume` that failed (`assume_fails`).
"""
import ast
from collections import Counter

from .. import api
from ..ir import nodes as N
from ..q30 import MASK, ONE, s32, to_q30
from .common import (SENSOR_GROUPS, ApiFn, Builtin, Folder, LoopVar, SensorGroup, SensorNS,
                     global_name, is_const, is_math, module_attr)
from .errors import DmpError, Loc
from .source import function_source

Q30, INT, BITS, S16 = N.Q30, N.INT, N.BITS, N.S16
LO, HI = -(1 << 31), (1 << 31) - 1


class R:
    """A runtime value: its type and raw signed 32-bit word."""
    __slots__ = ("t", "v")

    def __init__(self, t, v):
        self.t, self.v = t, v

    def __repr__(self):
        return f"R({self.t}, {self.v / ONE if self.t in (Q30, S16) else self.v})"


def arith(t):
    return Q30 if t == S16 else t


def lit(c) -> R:
    """A constant made a runtime value by its literal's type."""
    if isinstance(c, api.Bits):
        return R(BITS, s32(c.value))
    if isinstance(c, int):
        return R(INT, c)
    return R(Q30, to_q30(float(c)))


def conv(c, t) -> int:
    """A constant's raw value as an operand of type t."""
    if t == Q30:
        return to_q30(float(c))
    if isinstance(c, api.Bits):
        return s32(c.value)
    return s32(c) if t == BITS else c


class _Break(Exception):
    pass


class _Return(Exception):
    def __init__(self, value=None):
        self.value = value


class Frame:
    def __init__(self, src, kind):
        self.src, self.kind = src, kind
        self.names: dict = {}
        self.decls: set = set()
        self.folder = None

    def bind(self, name, value):
        """Bind a local; forget what was known to be constant if this can change it."""
        old = self.names.get(name, _UNBOUND)
        self.names[name] = value
        if self.folder is not None and (old is _UNBOUND or is_const(old) != is_const(value)):
            self.folder.cache.clear()

    def unbind(self, name):
        self.names.pop(name, None)
        if self.folder is not None:
            self.folder.cache.clear()


_UNBOUND = object()


class Reference:
    MAX_WHILE = 100_000

    def __init__(self, program, ir: N.ProgramIR):
        self.program = program
        self.schema = [f.type for f in ir.packet]
        self.src = function_source(program.fn)
        self.state: dict = {}
        self.kinds: dict = {}
        self.wraps: Counter = Counter()
        self.assume_fails: Counter = Counter()
        self.augmented: dict = {}
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        fn = self.src.node
        self.state.clear()
        for st in fn.body:
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                ann = st.annotation.func if isinstance(st.annotation, ast.Call) else st.annotation
                self.kinds[st.target.id] = ann.attr if isinstance(ann, ast.Attribute) else ann.id
                init = eval(compile(ast.Expression(st.value), self.src.file, "eval"),
                            dict(self.src.globals))
                self.state[st.target.id] = self.deep(init)

    def deep(self, v):
        if isinstance(v, (list, tuple)):
            return [self.deep(x) for x in v]
        return lit(v) if is_const(v) else R(v.t, v.v)

    def set_param(self, name, raw):
        if self.kinds.get(name) != "param":
            raise KeyError(f"no param {name!r}")
        self.state[name] = R(self.state[name].t, raw)

    # ------------------------------------------------------------------ a pass
    def run_pass(self, sample):
        self.sample = sample
        self.packet = bytearray()
        self.field = 0
        self.irq = False
        fr = Frame(self.src, "program")
        fr.bind(self.src.node.args.args[0].arg, SensorNS())
        fr.decls = set(self.state)
        body = [s for s in self.src.node.body if not isinstance(s, ast.AnnAssign)
                and not self.docstring(s, self.src.node)]
        try:
            self.block(body, fr)
        except _Return:
            pass
        return bytes(self.packet), self.irq

    @staticmethod
    def docstring(s, fn):
        return s is fn.body[0] and isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) \
            and isinstance(s.value.value, str)

    def loc(self, node, fr):
        return Loc.of(node, fr.src.file)

    def err(self, msg, node=None, fr=None):
        return DmpError(msg, self.loc(node, fr) if fr and node is not None else None)

    # ------------------------------------------------------------------ statements
    def block(self, stmts, fr):
        for s in stmts:
            self.stmt(s, fr)

    def stmt(self, s, fr):
        if isinstance(s, ast.Assign):
            v = self.ev(s.value, fr)
            self.assign(s.targets[0], v, fr, s.value)
        elif isinstance(s, ast.AugAssign):
            binop = self.augmented.get(s)
            if binop is None:                           # x += e is x = x + e, built once
                t = s.target
                load = ast.Name(t.id, ast.Load()) if isinstance(t, ast.Name) else \
                    ast.Subscript(t.value, t.slice, ast.Load())
                ast.copy_location(load, t)
                binop = ast.fix_missing_locations(ast.copy_location(ast.BinOp(load, s.op, s.value), s))
                self.augmented[s] = binop
            self.assign(s.target, self.ev(binop, fr), fr, binop)
        elif isinstance(s, ast.If):
            c = self.cond(s.test, fr)
            self.block(s.body if c else s.orelse, fr)
        elif isinstance(s, ast.For):
            self.for_(s, fr)
        elif isinstance(s, ast.While):
            n = 0
            try:
                while self.cond(s.test, fr):
                    n += 1
                    if n > self.MAX_WHILE:
                        raise RuntimeError(f"while loop ran {self.MAX_WHILE} times")
                    self.block(s.body, fr)
            except _Break:
                pass
        elif isinstance(s, ast.Break):
            raise _Break()
        elif isinstance(s, ast.Return):
            raise _Return(None if s.value is None else self.ev(s.value, fr))
        elif isinstance(s, ast.Expr):
            if isinstance(s.value, ast.Call):
                self.call(s.value, fr, stmt=True)
        elif isinstance(s, ast.Pass):
            pass
        else:
            raise self.err(f"unexpected {type(s).__name__}", s, fr)

    def for_(self, s, fr):
        it = s.iter
        f = self.resolve(it.func, fr)
        args = [self.folder(fr).fold(a) for a in it.args]
        name = s.target.id
        try:
            if isinstance(f, Builtin):                  # range: unrolled, the index a constant
                for i in range(*args):
                    fr.bind(name, i)
                    self.block(s.body, fr)
            else:                                       # dmp.loop(N)
                fr.bind(name, LoopVar())
                for _ in range(args[0]):
                    self.block(s.body, fr)
        except _Break:
            pass

    def assign(self, target, v, fr, rhs):
        if isinstance(target, ast.Name):
            name = target.id
            if fr.kind == "program" and name in fr.decls:
                cur = self.state[name]
                if isinstance(cur, list):
                    self.store_list(cur, v)
                else:
                    self.state[name] = self.retype(cur, v)
                return
            if isinstance(v, list):
                fr.bind(name, v if self.named(rhs) else self.copy(v))
            else:
                fr.bind(name, lit(v) if is_const(v) else v)
            return
        if isinstance(target, ast.Subscript):
            base = self.ev(target.value, fr)
            i = self.folder(fr).fold(target.slice)
            if isinstance(base[i], list):
                self.store_list(base[i], v)
            else:
                base[i] = self.retype(base[i], v)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            vals = [self.copy(x) if isinstance(x, list) else (lit(x) if is_const(x) else x) for x in v]
            for t, x in zip(target.elts, vals):
                self.assign(t, x, fr, None)
            return
        raise self.err("cannot assign", target, fr)

    @staticmethod
    def named(rhs):
        while isinstance(rhs, ast.Subscript):
            rhs = rhs.value
        return isinstance(rhs, ast.Name)

    def copy(self, v):
        if isinstance(v, list):
            return [self.copy(x) for x in v]
        return lit(v) if is_const(v) else R(v.t, v.v)

    def retype(self, cur: R, v) -> R:
        x = lit(v) if is_const(v) else v
        t = x.t if cur.t in (Q30, S16) and x.t in (Q30, S16) else cur.t
        return R(t, x.v)

    def store_list(self, cur: list, v):
        snap = self.copy(v)                         # the right side is complete before any store
        for i, x in enumerate(snap):
            if isinstance(cur[i], list):
                self.store_list(cur[i], x)
            else:
                cur[i] = self.retype(cur[i], x)

    # ------------------------------------------------------------------ names
    def folder(self, fr):
        fo = fr.folder
        if fo is None:
            fo = fr.folder = Folder(lambda n: self.resolve(n, fr), lambda msg, node: self.err(msg, node, fr),
                                    cache={})
        return fo

    def resolve(self, node, fr):
        """What a name, attribute or subscript path stands for (no side effects)."""
        if isinstance(node, ast.Name):
            if fr.kind == "program" and node.id in fr.decls:
                return self.state[node.id]
            if node.id in fr.names:
                return fr.names[node.id]
            return global_name(node.id, fr.src.globals, lambda m, h: self.err(m, node, fr))
        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value, fr)
            a = node.attr
            if isinstance(base, SensorNS):
                if a in SENSOR_GROUPS:
                    return SensorGroup(SENSOR_GROUPS[a])
                return self.sensor(a)
            if isinstance(base, SensorGroup):
                return self.sensor(base.prefix + a)
            return module_attr(base, a, lambda m, h: self.err(m, node, fr))
        if isinstance(node, ast.Subscript):
            base = self.resolve(node.value, fr)
            i = self.folder(fr).fold(node.slice)
            if isinstance(base, SensorGroup):
                return self.sensor(base.prefix + "xyz"[i])
            return base[i]
        raise self.err("not a name", node, fr)

    def sensor(self, f):
        s = self.sample
        if f == "fsync":
            return R(INT, s.fsync & 1)
        raw = {"ax": s.ax, "ay": s.ay, "az": s.az, "gx": s.gx, "gy": s.gy, "gz": s.gz,
               "temp": s.temp}[f]
        return R(S16, s32(raw << 16))

    # ------------------------------------------------------------------ expressions
    def ev(self, node, fr):
        fo = self.folder(fr)
        if fo.is_static(node):
            return fo.fold(node)
        if isinstance(node, (ast.Name, ast.Attribute)):
            return self.resolve(node, fr)
        if isinstance(node, ast.Subscript):
            base = self.ev(node.value, fr)
            i = fo.fold(node.slice)
            if isinstance(base, SensorGroup):
                return self.sensor(base.prefix + "xyz"[i])
            return base[i]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.ev(e, fr) for e in node.elts]
        if isinstance(node, ast.ListComp):
            g = node.generators[0]
            out = []
            old = fr.names.get(g.target.id)
            for i in range(*[fo.fold(a) for a in g.iter.args]):
                fr.bind(g.target.id, i)
                out.append(self.ev(node.elt, fr))
            if old is None:
                fr.unbind(g.target.id)
            else:
                fr.bind(g.target.id, old)
            return out
        if isinstance(node, ast.BinOp):
            op = node.op
            if isinstance(op, ast.Mult) and isinstance(node.left, (ast.List, ast.ListComp, ast.Tuple)):
                return self.ev(node.left, fr) * fo.fold(node.right)
            if isinstance(op, ast.Mult) and isinstance(node.right, (ast.List, ast.ListComp, ast.Tuple)):
                return fo.fold(node.left) * self.ev(node.right, fr)
            if isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                return self.linear(node, fr)
            if isinstance(op, (ast.BitAnd, ast.BitXor, ast.BitOr)):
                a = self.bits_raw(self.ev(node.left, fr))
                b = self.bits_raw(self.ev(node.right, fr))
                r = {ast.BitAnd: a & b, ast.BitXor: a ^ b, ast.BitOr: a | b}[type(op)]
                return R(BITS, s32(r))
            if isinstance(op, (ast.LShift, ast.RShift)):
                a = self.bits_raw(self.ev(node.left, fr))
                k = fo.fold(node.right)
                return R(BITS, s32(a << k) if isinstance(op, ast.LShift) else a >> k)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Invert):
                return R(BITS, s32(self.bits_raw(self.ev(node.operand, fr)) ^ -1))
            return self.linear(node, fr)
        if isinstance(node, ast.IfExp):
            c, static = self.cond_s(node.test, fr)
            if static:
                return self.ev(node.body if c else node.orelse, fr)
            a, b = self.ev(node.body, fr), self.ev(node.orelse, fr)
            t = self.unify([a, b])
            ra, rb = self.raw(a, t), self.raw(b, t)
            ta = a.t if isinstance(a, R) else t
            tb = b.t if isinstance(b, R) else t
            return R(ta if ta == tb else t, ra if c else rb)
        if isinstance(node, ast.Call):
            return self.call(node, fr, stmt=False)
        raise self.err(f"unexpected {type(node).__name__}", node, fr)

    @staticmethod
    def bits_raw(v):
        if isinstance(v, R):
            return v.v
        return s32(v.value if isinstance(v, api.Bits) else v)

    @staticmethod
    def unify(vals):
        ts = {arith(v.t) for v in vals if isinstance(v, R)}
        if ts:
            return ts.pop()
        c = vals[0]
        return BITS if isinstance(c, api.Bits) else INT if isinstance(c, int) else Q30

    @staticmethod
    def raw(v, t):
        return v.v if isinstance(v, R) else conv(v, t)

    # ------------------------------------------------------------------ sums
    def linear(self, node, fr):
        fo = self.folder(fr)
        if fo.is_static(node):
            return fo.fold(node)
        terms = []
        self.terms(node, 1, terms, fr)
        kinds = set()
        for coef, kind, x in terms:
            if kind == "prod":
                kinds.add(Q30)
            elif kind == "val":
                kinds.add(arith(x.t))
        typ = kinds.pop() if kinds else Q30
        vals = []
        for coef, kind, x in terms:
            if kind == "const":
                raw = conv(x, typ)
                if raw:
                    vals.append((coef, "val", R(typ, raw)))
            else:
                vals.append((coef, kind, x))
        if len(vals) == 1 and vals[0][0] == 1 and vals[0][1] == "val":
            return vals[0][2]
        if not vals:
            return R(typ, 0)
        total = 0
        for coef, kind, x in vals:
            if kind == "prod":
                total += coef * x[0] * x[1]
            else:
                total += coef * x.v * (1 if typ == INT else ONE)
        r = total if typ == INT else total >> 30
        if not LO <= r <= HI:
            self.wraps[self.loc(node, fr)] += 1
        return R(typ, s32(r))

    def terms(self, node, coef, out, fr):
        fo = self.folder(fr)
        if fo.is_static(node):
            out.append((coef, "const", fo.fold(node)))
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            self.terms(node.left, coef, out, fr)
            self.terms(node.right, -coef if isinstance(node.op, ast.Sub) else coef, out, fr)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            self.terms(node.operand, -coef if isinstance(node.op, ast.USub) else coef, out, fr)
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for k_node, x_node in ((node.left, node.right), (node.right, node.left)):
                if fo.is_static(k_node):
                    k = fo.fold(k_node)
                    if isinstance(k, int) and not isinstance(k, bool):
                        if k:
                            self.terms(x_node, coef * k, out, fr)
                        return
            sa, fa = self.unneg(node.left)
            sb, fb = self.unneg(node.right)
            a, b = self.factor(fa, fr), self.factor(fb, fr)
            out.append((coef * sa * sb, "prod", (a, b)))
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            c = fo.fold(node.right)
            sa, fa = self.unneg(node.left)
            out.append((coef * sa, "prod", (self.factor(fa, fr), to_q30(1 / c))))
            return
        v = self.ev(node, fr)
        if is_const(v):
            v = lit(v)
            if not v.v:
                return
        out.append((coef, "val", v))

    @staticmethod
    def unneg(node):
        s = 1
        while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            if isinstance(node.op, ast.USub):
                s = -s
            node = node.operand
        return s, node

    def factor(self, node, fr) -> int:
        v = self.ev(node, fr)
        return to_q30(float(v)) if is_const(v) else v.v

    # ------------------------------------------------------------------ conditions
    def cond(self, node, fr) -> bool:
        return self.cond_s(node, fr)[0]

    def cond_s(self, node, fr):
        """(value, static?).  Every operand is evaluated (no short circuit: an operand's helper
        call happens either way); a constant operand that decides an and/or decides it."""
        if isinstance(node, ast.Compare):
            return self.compare(node, fr)
        if isinstance(node, ast.BoolOp):
            is_and = isinstance(node.op, ast.And)
            parts = [self.cond_s(v, fr) for v in node.values]
            for c, static in parts:
                if static and c != is_and:
                    return c, True
            rt = [c for c, static in parts if not static]
            if not rt:
                return is_and, True
            return (all(rt) if is_and else any(rt)), False
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            c, static = self.cond_s(node.operand, fr)
            return not c, static
        v = self.ev(node, fr)
        if is_const(v):
            return bool(v.value if isinstance(v, api.Bits) else v), True
        return v.v != 0, False

    def compare(self, node, fr):
        ops = {ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b,
               ast.GtE: lambda a, b: a >= b, ast.Eq: lambda a, b: a == b,
               ast.NotEq: lambda a, b: a != b}
        left = self.ev(node.left, fr)
        results = []
        for op, rn in zip(node.ops, node.comparators):
            right = self.ev(rn, fr)
            f = ops[type(op)]
            if is_const(left) and is_const(right):
                a = left.value if isinstance(left, api.Bits) else left
                b = right.value if isinstance(right, api.Bits) else right
                results.append((f(a, b), True))
            else:
                t = self.unify([left, right])
                results.append((f(self.raw(left, t), self.raw(right, t)), False))
            left = right
        if any(static and not c for c, static in results):
            return False, True
        rt = [c for c, static in results if not static]
        return (all(rt), False) if rt else (True, True)

    # ------------------------------------------------------------------ calls
    def call(self, node, fr, stmt):
        f = self.resolve(node.func, fr)
        fo = self.folder(fr)
        if isinstance(f, api.Sub):
            return self.run_function(f.fn, [self.ev(a, fr) for a in node.args], node, fr, sub=True)
        if callable(f) and not is_math(f) and not isinstance(f, (ApiFn, Builtin)):
            return self.run_function(f, [self.ev(a, fr) for a in node.args], node, fr, sub=False)
        if fo.is_static(node):
            return fo.fold(node)
        if isinstance(f, Builtin):
            if f.name == "len":
                return len(self.ev(node.args[0], fr))
            if f.name == "abs":
                v = self.ev(node.args[0], fr)
                t = arith(v.t)
                if v.v < 0:
                    if v.v == LO:
                        self.wraps[self.loc(node, fr)] += 1
                    return R(t, s32(-v.v))
                return R(t, v.v)
            vals = [self.ev(a, fr) for a in node.args]
            t = self.unify(vals)
            acc = self.raw(vals[0], t)
            for v in vals[1:]:
                acc = min(acc, self.raw(v, t)) if f.name == "min" else max(acc, self.raw(v, t))
            return R(t, acc)
        name = f.name
        if name == "emit":
            for a in list(node.args) + [k.value for k in node.keywords]:
                v = self.ev(a, fr)
                raw = self.raw(v, arith(v.t) if isinstance(v, R) else self.unify([v]))
                if self.schema[self.field] == S16:
                    self.packet += ((raw >> 16) & 0xFFFF).to_bytes(2, "big")
                else:
                    self.packet += (raw & MASK).to_bytes(4, "big")
                self.field += 1
            return None
        if name == "irq":
            self.irq = True
            return None
        if name in ("clamp", "assume"):
            x = self.ev(node.args[0], fr)
            lo, hi = fo.fold(node.args[1]), fo.fold(node.args[2])
            if is_const(x):
                x = lit(x)
            if name == "assume":
                val = x.v / ONE if x.t in (Q30, S16) else x.v
                if not lo <= val <= hi:
                    self.assume_fails[self.loc(node, fr)] += 1
                return None if stmt else x
            t = arith(x.t)
            return R(t, min(max(x.v, conv(lo, t)), conv(hi, t)))
        raise self.err(f"dmp.{name} here", node, fr)

    def run_function(self, fn, args, node, fr, sub):
        src = function_source(fn)
        callee = Frame(src, "sub" if sub else "helper")
        for p, v, an in zip(src.node.args.args, args, node.args):
            if sub:
                v = lit(v) if is_const(v) else R(arith(v.t), v.v)
            elif isinstance(v, list):
                v = v if self.named(an) else self.copy(v)
            callee.bind(p.arg, v)
        body = [s for s in src.node.body if not self.docstring(s, src.node)]
        try:
            self.block(body, callee)
        except _Return as r:
            return self.result(r.value)
        return None

    def result(self, v):
        if isinstance(v, list):
            return [self.result(x) for x in v]
        if is_const(v):
            return lit(v)
        return R(arith(v.t), v.v)
