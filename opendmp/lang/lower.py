"""The front end: a @dmp.program's Python AST -> the IR (opendmp.ir.nodes).

It resolves names, checks the subset, types every value, folds constants, inlines helpers and
unrolls `for ... in range()`.  Anything it cannot give a DMP meaning is a DmpError with the
location and a hint.

The rules it implements (docs: PLAN.md section 4, "reference semantics"):

- **Constants** are literals, module-level names, `range()` indices, helper parameters whose
  argument is a constant, and expressions of these; they are folded in Python, as written.
  **Assigning to a local always makes a DMP value**, even `h = 0.75`.
- **The literal picks the type**: `0.0` is Q30, `0` is Int, `dmp.bits(...)` is Bits.  A
  constant meeting a Q30 value becomes Q30 (rounded to nearest); out of [-2, 2) is an error.
- **A +/- expression is one exact Sum**: its terms, integer multiples (`2*x`) and two-factor
  products (`a*b`, `K*x`, `x/C`) are added in full precision and floored and wrapped once.
  A product's factors are values (an inner product or sum is floored and wrapped first); a
  minus sign on a factor moves to the term.
- **Lists** exist only here: every index is a constant, so each element is its own variable.
  Assigning a named list (`y = z[0]`) or passing one to a helper shares it; a list display or
  a helper's result is copied.
"""
import ast
import types

from .. import api
from ..ir import nodes as N
from ..q30 import s32, to_q30
from .common import (CMP_OPS, FIELD_NAME, SENSOR_GROUPS, ApiFn, Builtin, DmpNS, Folder, LoopVar,
                     SensorGroup, SensorNS, global_name, is_const, is_math, is_number, module_attr)
from .errors import DmpError, Loc, node_text
from .source import function_source

HINT_DIV = "multiply by a constant reciprocal (x * (1 / C)), or use a Newton step (examples/unit_vectors.py)"
HINT_MIX = ("an int literal makes an Int (a counter); write 0.0 for a Q30 value, or keep counters "
            "and Q30 values apart")


class Frame:
    def __init__(self, src, kind: str):
        self.src = src
        self.kind = kind                  # "program" | "helper" | "sub"
        self.names: dict = {}             # local name -> Var | list | constant | SensorNS | LoopVar
        self.assigned: set = set()        # locals definitely assigned here
        self.decls: set = set()           # the program's state/param names (never rebound)
        self.ret_label = None
        self.ret = None                   # result Var(s), once the first `return` fixes the shape
        self.loops: list = []             # [(label, kind)] innermost last

    @property
    def file(self):
        return self.src.file


class Lowerer:
    def __init__(self, program):
        self.program = program
        self.vars: list = []
        self.subs: dict = {}
        self.uses: set = set()
        self.frames: list = []
        self.calls: list = []             # helper call stack (recursion check)
        self.n = 0
        self.depth = 0                    # conditional nesting (emit placement)
        self.cond_return = False          # a conditional `return` happened in the program
        self.out: list = None             # the statement list being built
        self.falls = True                 # does control reach the current point?
        self.folder = Folder(self.expr, lambda msg, node: self.err(msg, node))

    # ------------------------------------------------------------------ errors, helpers
    @property
    def f(self) -> Frame:
        return self.frames[-1]

    def loc(self, node):
        return Loc.of(node, self.f.file)

    def err(self, msg, node=None, hint=None):
        return DmpError(msg, self.loc(node) if node is not None else None, hint)

    def label(self, what):
        self.n += 1
        return f"{what}{self.n}"

    def emit_stmt(self, stmt):
        if self.falls:
            self.out.append(stmt)

    def new_var(self, name, kind, typ, node=None, **kw):
        v = N.Var(name, kind, typ, loc=self.loc(node) if node is not None else None, **kw)
        self.vars.append(v)
        return v

    def temp(self, typ, hint="t", node=None):
        self.n += 1
        prefix = "" if self.f.kind == "program" else f"{self.f.src.name}."
        return self.new_var(f"{prefix}{hint}#{self.n}", "temp", typ, node)

    def materialize(self, e: N.Expr, node, hint="t") -> N.Expr:
        """Evaluate e into a fresh temp now; return a Load of it."""
        if isinstance(e, (N.Const, N.Sensor)):
            return e
        t = self.temp(e.type, hint, node)
        self.emit_stmt(N.Assign(t, e, self.loc(node)))
        return N.Load(t, loc=e.loc)

    # ------------------------------------------------------------------ entry
    def run(self) -> N.ProgramIR:
        src = function_source(self.program.fn)
        fr = Frame(src, "program")
        self.frames.append(fr)
        fn = src.node
        a = fn.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.defaults or a.posonlyargs or len(a.args) != 1:
            raise self.err(f"a DMP program takes one argument, the sensor sample: def {fn.name}(s):",
                           fn)
        fr.names[a.args[0].arg] = SensorNS()
        fr.assigned.add(a.args[0].arg)
        body = []
        for st in fn.body:
            if isinstance(st, ast.AnnAssign):
                self.declare(st)
            elif self.is_docstring(st, fn):
                continue
            else:
                body.append(st)
        self.out = []
        label = "pass"
        fr.ret_label = label
        self.block(body)
        stmts = self.out
        if self.contains(fn.body, ast.Return):
            stmts = [N.Block(label, stmts)]
        packet, off = [], 0
        seen = set()
        for e in self.walk_emits(stmts):
            for name, x in e.fields:
                if name in seen:
                    raise DmpError(f"the packet already has a field named {name!r}", e.loc,
                                   "name the fields apart: dmp.emit(name=value)")
                seen.add(name)
                packet.append(N.Field(name, x.type, off))
                off += N.WIDTH[x.type]
        ordered = [v for v in self.vars if v.kind in ("state", "param")] + \
                  [v for v in self.vars if v.kind not in ("state", "param")]
        return N.ProgramIR(self.program.name, src.file, dict(self.program.opts), ordered, stmts,
                           packet, self.subs, self.uses)

    @staticmethod
    def is_docstring(st, fn):
        return (st is fn.body[0] and isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)
                and isinstance(st.value.value, str))

    @staticmethod
    def contains(stmts, kind):
        return any(isinstance(n, kind) for st in stmts for n in ast.walk(st))

    def walk_emits(self, stmts):
        for st in stmts:
            if isinstance(st, N.Emit):
                yield st
            elif isinstance(st, N.Block):
                yield from self.walk_emits(st.body)

    # ------------------------------------------------------------------ declarations
    def declare(self, st: ast.AnnAssign):
        fr = self.f
        if not isinstance(st.target, ast.Name):
            raise self.err("declare a whole name: `x: dmp.state = 0.0`", st.target)
        name = st.target.id
        decl = self.resolve_annotation(st.annotation)
        if name in fr.names:
            raise self.err(f"{name} is declared twice", st.target)
        if st.value is None:
            raise self.err(f"{name} needs an initial value: `{name}: dmp.{decl.kind} = 0.0`", st,
                           "the value is loaded with the program; 0.0 makes a Q30, 0 an Int")
        try:
            init = eval(compile(ast.Expression(st.value), fr.file, "eval"), dict(fr.src.globals))
        except Exception as e:
            raise self.err(f"the initial value of {name} must be a constant expression ({e})",
                           st.value) from None
        rng = None
        if decl.lo is not None:
            rng = (float(decl.lo), float(decl.hi))
            if not rng[0] <= rng[1]:
                raise self.err(f"the range of {name} is empty ({decl.lo} > {decl.hi})", st.annotation)

        def make(v, path):
            if isinstance(v, (list, tuple)):
                if not v:
                    raise self.err(f"{name}: an empty list", st.value)
                return [make(x, path + (i,)) for i, x in enumerate(v)]
            vname = name + "".join(f"[{i}]" for i in path)
            typ, raw = self.const_type_raw(v, st.value, f"the initial value of {vname}")
            return self.new_var(vname, decl.kind, typ, st, init=raw, range=rng)

        fr.names[name] = make(init, ())
        fr.decls.add(name)
        fr.assigned.add(name)

    def resolve_annotation(self, ann):
        base, args = ann, []
        if isinstance(ann, ast.Call):
            base, args = ann.func, ann.args
            if ann.keywords or len(args) != 2:
                raise self.err("a range is dmp.state(lo, hi)", ann)
        obj = self.try_attr_path(base)
        if not isinstance(obj, api._Decl):
            raise self.err(f"annotate a declaration with dmp.state or dmp.param, not {node_text(base)}",
                           ann, "a plain local needs no annotation: `x = 0.0`")
        if args:
            lo, hi = (self.const_expr(a) for a in args)
            if not (is_number(lo) and is_number(hi)):
                raise self.err("range bounds must be numbers", ann)
            return api._Decl(obj.kind, lo, hi)
        return obj

    def try_attr_path(self, node):
        """Resolve dmp.state / opendmp.param without lowering."""
        if isinstance(node, ast.Name):
            return self.f.src.globals.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.try_attr_path(node.value)
            return getattr(base, node.attr, None) if base is not None else None
        return None

    def const_type_raw(self, v, node, what):
        if isinstance(v, bool):
            raise self.err(f"{what} is a bool; use 1 or 0", node)
        if isinstance(v, api.Bits):
            return N.BITS, s32(v.value)
        if isinstance(v, int):
            if not -(1 << 31) <= v < (1 << 31):
                raise self.err(f"{what} = {v} does not fit 32 bits", node)
            return N.INT, v
        if isinstance(v, float):
            return N.Q30, self.q30(v, node, what)
        raise self.err(f"{what} must be a number (Q30 float, Int, or dmp.bits), not {type(v).__name__}",
                       node)

    def q30(self, v, node, what="this constant"):
        try:
            return to_q30(float(v))
        except (ValueError, OverflowError):
            raise self.err(f"{what} = {v:g} is outside the Q30 range [-2, 2)", node,
                           "scale it into range; to double a value exactly, multiply by the integer 2"
                           ) from None

    # ------------------------------------------------------------------ statements
    def block(self, stmts):
        for st in stmts:
            if not self.falls:
                break                               # after break/return: unreachable
            self.stmt(st)

    def sub_block(self, stmts, depth=1):
        """Lower stmts into a new list; returns (list, falls through?)."""
        saved_out, saved_falls = self.out, self.falls
        self.out, self.falls = [], True
        self.depth += depth
        try:
            self.block(stmts)
            return self.out, self.falls
        finally:
            self.depth -= depth
            self.out, self.falls = saved_out, saved_falls

    def stmt(self, st):
        m = getattr(self, "st_" + type(st).__name__, None)
        if m is None:
            raise self.unsupported_stmt(st)
        m(st)

    def unsupported_stmt(self, st):
        name = type(st).__name__
        hints = {
            "Assert": ("assert", "state what holds with dmp.assume(x, lo, hi): the simulator checks it"),
            "Try": ("try/except", "nothing on the DMP raises"),
            "With": ("with", None), "Raise": ("raise", None), "Delete": ("del", None),
            "Global": ("global", "state lives in dmp.state declarations in the program"),
            "Nonlocal": ("nonlocal", None), "Import": ("import inside a program", "import at module level"),
            "ImportFrom": ("import inside a program", "import at module level"),
            "ClassDef": ("a class", None),
            "FunctionDef": ("a nested def", "define helpers at module level"),
            "Continue": ("continue", "use if/else inside the loop"),
            "Match": ("match", "use if/elif"), "AsyncFor": ("async for", None),
            "AsyncWith": ("async with", None), "TypeAlias": ("type", None),
        }
        what, hint = hints.get(name, (name, None))
        return self.err(f"{what} is not supported in a DMP program", st, hint)

    def st_Pass(self, st):
        pass

    def st_AnnAssign(self, st):
        if self.f.kind != "program" or self.depth:
            raise self.err("declare dmp.state and dmp.param at the top of the program body", st,
                           "helpers keep no state of their own; pass them a state list")
        raise self.err("declarations go at the top level of the program body", st)

    def st_Expr(self, st):
        v = st.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return                                  # a string statement: a comment
        if not isinstance(v, ast.Call):
            raise self.err("this expression does nothing", st,
                           "assign it (x = ...), emit it (dmp.emit(...)) or remove it")
        self.call(v, stmt=True)

    def st_Assign(self, st):
        if len(st.targets) != 1:
            raise self.err("chained assignment (a = b = ...) is not supported", st)
        value = self.expr(st.value)
        self.assign(st.targets[0], value, st, rhs=st.value)

    def st_AugAssign(self, st):
        if isinstance(st.target, ast.Name):
            load = ast.Name(st.target.id, ast.Load())
        elif isinstance(st.target, ast.Subscript):
            load = ast.Subscript(st.target.value, st.target.slice, ast.Load())
        else:
            raise self.err("augmented assignment needs a name or an element", st.target)
        ast.copy_location(load, st.target)
        binop = ast.copy_location(ast.BinOp(load, st.op, st.value), st)
        ast.fix_missing_locations(binop)
        self.assign(st.target, self.expr(binop), st, rhs=binop)

    def st_If(self, st):
        c = self.cond(st.test)
        if isinstance(c, bool):                     # a constant condition: only one branch exists
            self.block(st.body if c else st.orelse)
            return
        fr = self.f
        before = set(fr.assigned)
        then, t_falls = self.sub_block(st.body)
        after_then = set(fr.assigned)
        fr.assigned = set(before)
        els, e_falls = self.sub_block(st.orelse)
        after_else = set(fr.assigned)
        if t_falls and e_falls:
            fr.assigned = after_then & after_else
        elif t_falls:
            fr.assigned = after_then
        elif e_falls:
            fr.assigned = after_else
        else:
            fr.assigned = after_then & after_else
        self.emit_stmt(N.If(c, then, els, self.loc(st)))
        if not (t_falls or e_falls):
            self.falls = False

    def st_For(self, st):
        if st.orelse:
            raise self.err("for ... else is not supported", st)
        it = st.iter
        if not isinstance(st.target, ast.Name):
            raise self.err("a loop variable is one name", st.target)
        kind = None
        if isinstance(it, ast.Call):
            callee = self.callee(it.func)
            if callee == ("builtin", "range"):
                kind = "range"
            elif callee == ("dmp", "loop"):
                kind = "loop"
        if kind is None:
            raise self.err("a DMP loop is `for i in range(N)` (unrolled) or `for _ in dmp.loop(N)`",
                           st.iter, "loops over lists: use `for i in range(len(xs))` and xs[i]")
        args = [self.const_expr(a) for a in it.args]
        if it.keywords or not args or not all(isinstance(a, int) and not isinstance(a, bool) for a in args):
            raise self.err("the loop bound must be a constant integer", it)
        fr = self.f
        if kind == "range":
            rng = range(*args)
            if len(rng) > 4096:
                raise self.err(f"range({', '.join(map(str, args))}) unrolls to {len(rng)} copies", it)
            has_break = self.loop_has_break(st.body)
            label = self.label("for")
            fr.loops.append((label, "range"))
            saved_out, saved_falls = self.out, self.falls
            if has_break:
                self.out, self.falls = [], True
                self.depth += 1
            try:
                for i in rng:
                    if not self.falls:
                        break
                    fr.names[st.target.id] = i
                    fr.assigned.add(st.target.id)
                    self.block(st.body)
                    if has_break:
                        self.falls = True           # a Break leaves the block, not the loop body
            finally:
                fr.loops.pop()
                if has_break:
                    self.depth -= 1
                    body = self.out
                    self.out, self.falls = saved_out, saved_falls
                    self.emit_stmt(N.Block(label, body))
            return
        n = args[0] if len(args) == 1 else None
        if n is None or n < 1:
            raise self.err("dmp.loop(N) takes one positive constant", it)
        label = self.label("loop")
        fr.names[st.target.id] = LoopVar()
        fr.assigned.add(st.target.id)
        fr.loops.append((label, "loop"))
        before = set(fr.assigned)
        try:
            body, _ = self.sub_block(st.body)
        finally:
            fr.loops.pop()
        fr.assigned = before
        self.emit_stmt(N.Loop(label, n, body, self.loc(st)))

    @staticmethod
    def loop_has_break(stmts):
        def scan(node):
            if isinstance(node, ast.Break):
                return True
            if isinstance(node, (ast.For, ast.While, ast.FunctionDef)):
                return False
            return any(scan(c) for c in ast.iter_child_nodes(node))
        return any(scan(s) for s in stmts)

    def st_While(self, st):
        if st.orelse:
            raise self.err("while ... else is not supported", st)
        c = self.cond(st.test)
        if c is False:
            return
        fr = self.f
        label = self.label("while")
        fr.loops.append((label, "while"))
        before = set(fr.assigned)
        try:
            body, _ = self.sub_block(st.body)
        finally:
            fr.loops.pop()
        fr.assigned = before
        head = [] if c is True else [N.If(N.Not(c), [N.Break(label)], [], self.loc(st.test))]
        self.emit_stmt(N.Loop(label, None, head + body, self.loc(st)))

    def st_Break(self, st):
        if not self.f.loops:
            raise self.err("break outside a loop", st)
        self.emit_stmt(N.Break(self.f.loops[-1][0], self.loc(st)))
        self.falls = False

    def st_Return(self, st):
        fr = self.f
        if fr.kind == "program":
            if st.value is not None:
                raise self.err("a DMP program returns nothing", st, "send values with dmp.emit")
            if self.depth:
                self.cond_return = True
            self.emit_stmt(N.Break(fr.ret_label, self.loc(st)))
            self.falls = False
            return
        if st.value is None:
            raise self.err(f"{fr.src.name} must return a value on every return", st)
        val = self.expr(st.value)
        if fr.ret is None:
            fr.ret = self.result_vars(val, st)
        self.store_into(fr.ret, val, st, what=f"the result of {fr.src.name}")
        self.emit_stmt(N.Break(fr.ret_label, self.loc(st)))
        self.falls = False

    def result_vars(self, val, node):
        if isinstance(val, list):
            return [self.result_vars(v, node) for v in val]
        v = self.value(val, node)
        typ = N.Q30 if v.type == N.S16 else v.type
        return self.temp(typ, "ret", node)

    def store_into(self, target, val, node, what):
        """Assign val (value or list) into target (Var or nested list of Vars), same shape."""
        if isinstance(target, list):
            if not isinstance(val, list) or len(val) != len(target):
                raise self.err(f"{what}: expected {len(target)} values", node)
            vals = [v if isinstance(v, list) else self.value(v, node) for v in val]
            if any(self.reads_any(v, target) for v in vals if not isinstance(v, list)):
                vals = [v if isinstance(v, list) else self.materialize(v, node) for v in vals]
            for t, v in zip(target, vals):
                self.store_into(t, v, node, what)
            return
        if isinstance(val, list):
            raise self.err(f"{what}: a list where one value was expected", node)
        self.emit_stmt(N.Assign(target, self.coerce(val, target, node), self.loc(node)))

    def coerce(self, val, var: N.Var, node) -> N.Expr:
        """val, to be stored in var.  A constant keeps its literal's type, so `x = 1` into a Q30
        variable is an error (write 1.0), as is 0.5 into an Int."""
        return self.fit(self.value(val, node), var, node)

    def fit(self, e: N.Expr, var: N.Var, node) -> N.Expr:
        """Check that e may be stored in var; widen a sensor-typed temp that now holds Q30."""
        t, vt = e.type, var.type
        if t == vt or (vt == N.Q30 and t == N.S16):
            return e
        if var.kind == "temp" and vt == N.S16 and t == N.Q30:
            var.type = N.Q30
            return e
        raise self.err(f"{var.name} holds {self.tname(vt)}, but this is {self.tname(t)}", node,
                       HINT_MIX if N.INT in (t, vt) else None)

    @staticmethod
    def tname(t):
        return {N.Q30: "a Q30 value", N.INT: "an Int", N.BITS: "dmp.bits", N.S16: "a sensor sample"}[t]

    def reads_any(self, e, targets) -> bool:
        flat = set()

        def fl(t):
            if isinstance(t, list):
                for x in t:
                    fl(x)
            else:
                flat.add(id(t))
        fl(targets)
        return any(id(v) in flat for v in self.reads(e))

    def reads(self, e):
        if isinstance(e, N.Load):
            yield e.var
        elif isinstance(e, N.Sum):
            for t in e.terms:
                if isinstance(t.x, N.Prod):
                    yield from self.reads(t.x.a)
                    yield from self.reads(t.x.b)
                else:
                    yield from self.reads(t.x)
        elif isinstance(e, (N.MinMax, N.BitOp)):
            yield from self.reads(e.a)
            yield from self.reads(e.b)
        elif isinstance(e, (N.Abs, N.Shift, N.Assume)):
            yield from self.reads(e.x)
        elif isinstance(e, N.Select):
            yield from self.reads_cond(e.cond)
            yield from self.reads(e.a)
            yield from self.reads(e.b)

    def reads_cond(self, c):
        if isinstance(c, N.Cmp):
            yield from self.reads(c.a)
            yield from self.reads(c.b)
        elif isinstance(c, N.BoolOp):
            yield from self.reads_cond(c.a)
            yield from self.reads_cond(c.b)
        elif isinstance(c, N.Not):
            yield from self.reads_cond(c.x)

    # ------------------------------------------------------------------ assignment
    def assign(self, target, value, node, rhs):
        fr = self.f
        if isinstance(target, ast.Name):
            name = target.id
            cur = fr.names.get(name)
            if name in fr.decls:
                if isinstance(cur, N.Var) and cur.kind == "param" or (
                        isinstance(cur, list) and self.first(cur).kind == "param"):
                    raise self.err(f"{name} is a dmp.param: the host writes it, the program only reads it",
                                   target, "copy it into a local or a dmp.state to change it")
                self.store_into(cur, value, node, what=name)
                return
            if isinstance(value, list):
                if self.is_named(rhs):
                    fr.names[name] = value              # share it
                else:
                    fr.names[name] = self.copy_list(value, name, node)
                fr.assigned.add(name)
                return
            if isinstance(cur, N.Var) and cur.kind in ("temp", "arg"):
                self.emit_stmt(N.Assign(cur, self.coerce(value, cur, node), self.loc(node)))
            else:
                e = self.value(value, node)
                if isinstance(cur, list):
                    raise self.err(f"{name} was a list; give the value another name", target)
                v = self.temp(e.type, name, target)
                self.emit_stmt(N.Assign(v, e, self.loc(node)))
                fr.names[name] = v
            fr.assigned.add(name)
            return
        if isinstance(target, ast.Subscript):
            base = self.expr(target.value)
            if not isinstance(base, list):
                raise self.err(f"{node_text(target.value)} is not a list", target.value)
            elem = self.index(base, target, store=True)
            if isinstance(elem, N.Var) and elem.kind == "param":
                raise self.err(f"{elem.name} is a dmp.param: the host writes it", target)
            if not isinstance(elem, (N.Var, list)):
                raise self.err(f"{node_text(target)} cannot be assigned", target)
            self.store_into(elem, value, node, what=node_text(target))
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, list) or len(value) != len(target.elts):
                raise self.err(f"unpacking needs {len(target.elts)} values", node)
            vals = [v if isinstance(v, list) else self.materialize(self.value(v, node), node)
                    for v in value]
            for t, v in zip(target.elts, vals):
                if isinstance(t, ast.Starred):
                    raise self.err("starred unpacking is not supported", t)
                self.assign(t, v, node, rhs=None)
            return
        raise self.err(f"cannot assign to {node_text(target)}", target)

    @staticmethod
    def first(lst):
        while isinstance(lst, list):
            lst = lst[0]
        return lst

    @staticmethod
    def is_named(rhs):
        """A named list (shared on assignment): a name or an element of one."""
        while isinstance(rhs, ast.Subscript):
            rhs = rhs.value
        return isinstance(rhs, ast.Name)

    def copy_list(self, value, name, node, path=()):
        out = []
        for i, v in enumerate(value):
            if isinstance(v, list):
                out.append(self.copy_list(v, name, node, path + (i,)))
            else:
                e = self.value(v, node)
                t = self.temp(e.type, name + "".join(f"[{k}]" for k in path + (i,)), node)
                self.emit_stmt(N.Assign(t, e, self.loc(node)))
                out.append(t)
        return out

    # ------------------------------------------------------------------ values
    def value(self, v, node) -> N.Expr:
        """An operand as an IR value (a constant takes its literal's type)."""
        if isinstance(v, N.Expr):
            return v
        if isinstance(v, N.Var):
            return N.Load(v, loc=self.loc(node))
        if is_const(v):
            typ, raw = self.const_type_raw(v, node, node_text(node) if node is not None else "constant")
            return N.Const(typ, raw, loc=self.loc(node))
        if isinstance(v, list):
            raise self.err(f"{node_text(node)} is a list, and a single value is needed here", node)
        if isinstance(v, LoopVar):
            raise self.err("the dmp.loop variable has no value", node,
                           "use `for i in range(N)` when the body needs an index")
        raise self.err(f"{node_text(node)} is not a value", node)

    def as_type(self, v, typ, node) -> N.Expr:
        """v as an operand of an op of type typ (Q30, INT, BITS): converts constants."""
        if is_const(v):
            if typ == N.Q30:
                if not is_number(v):
                    raise self.err("dmp.bits in Q30 arithmetic", node)
                return N.Const(N.Q30, self.q30(v, node, node_text(node)), loc=self.loc(node))
            if typ == N.INT:
                if not isinstance(v, int) or isinstance(v, bool):
                    raise self.err(f"{node_text(node)} is not an integer, but this is Int arithmetic",
                                   node, HINT_MIX)
                return N.Const(N.INT, self.const_type_raw(v, node, node_text(node))[1], loc=self.loc(node))
            if typ == N.BITS:
                raw = v.value if isinstance(v, api.Bits) else v
                if not isinstance(raw, int):
                    raise self.err("dmp.bits takes an integer mask", node)
                return N.Const(N.BITS, s32(raw), loc=self.loc(node))
        e = self.value(v, node)
        et = N.arith_type(e.type)
        if et != typ:
            raise self.err(f"{node_text(node)} is {self.tname(e.type)}, where {self.tname(typ)} is needed",
                           node, HINT_MIX if N.INT in (et, typ) else None)
        return e

    def kind_of(self, v, node):
        """Q30 / INT / BITS for a runtime operand, None for a constant."""
        if is_const(v):
            return None
        return N.arith_type(self.value(v, node).type)

    def unify(self, vals, nodes, what) -> str:
        kinds = {k for k in (self.kind_of(v, n) for v, n in zip(vals, nodes)) if k}
        if len(kinds) > 1:
            raise self.err(f"{what} mixes {' and '.join(sorted(self.tname(k) for k in kinds))}",
                           nodes[0], HINT_MIX)
        if kinds:
            return kinds.pop()
        c = vals[0]
        return N.BITS if isinstance(c, api.Bits) else N.INT if isinstance(c, int) else N.Q30

    # ------------------------------------------------------------------ expressions
    def expr(self, node):
        m = getattr(self, "ex_" + type(node).__name__, None)
        if m is None:
            raise self.unsupported_expr(node)
        return m(node)

    def unsupported_expr(self, node):
        name = type(node).__name__
        what = {"Lambda": "lambda", "Dict": "a dict", "Set": "a set", "DictComp": "a dict comprehension",
                "SetComp": "a set comprehension", "GeneratorExp": "a generator", "JoinedStr": "an f-string",
                "Await": "await", "Yield": "yield", "YieldFrom": "yield", "NamedExpr": ":=",
                "Starred": "*unpacking", "Slice": "a slice"}.get(name, name)
        hint = "index with a constant: xs[2]" if name == "Slice" else None
        return self.err(f"{what} is not supported in a DMP program", node, hint)

    def ex_Constant(self, node):
        v = node.value
        if isinstance(v, bool) or not is_number(v):
            if isinstance(v, bool):
                raise self.err(f"{v} is not a DMP value", node, "use 1 and 0 (Int) for flags")
            raise self.err(f"{v!r} is not a DMP value", node)
        return v

    def ex_Name(self, node):
        fr = self.f
        name = node.id
        if name in fr.names:
            b = fr.names[name]
            if isinstance(b, N.Var):
                if name not in fr.assigned:
                    raise self.err(f"{name} may be used before it is assigned", node,
                                   "assign it on every path first, or make it a dmp.state")
                return N.Load(b, loc=self.loc(node))
            if isinstance(b, list) and name not in fr.assigned:
                raise self.err(f"{name} may be used before it is assigned", node)
            return b
        return self.global_value(name, node)

    def global_value(self, name, node):
        return global_name(name, self.f.src.globals, lambda msg, hint: self.err(msg, node, hint))

    def ex_Attribute(self, node):
        base = self.expr(node.value)
        a = node.attr
        if isinstance(base, SensorNS):
            if a in SENSOR_GROUPS:
                return SensorGroup(SENSOR_GROUPS[a])
            if a in ("temp", "fsync"):
                self.uses.add(a)
                return N.Sensor(a, loc=self.loc(node))
            raise self.err(f"the sample has no {a!r}", node, "it has accel, gyro, temp and fsync")
        if isinstance(base, SensorGroup):
            if a in ("x", "y", "z"):
                f = base.prefix + a
                self.uses.add(f)
                return N.Sensor(f, loc=self.loc(node))
            raise self.err(f"a sensor triple has x, y and z, not {a!r}", node)
        if isinstance(base, (DmpNS, types.ModuleType)):
            return module_attr(base, a, lambda msg, hint: self.err(msg, node, hint))
        raise self.err(f"{node_text(node)}: values have no attributes", node)

    def ex_Subscript(self, node):
        base = self.expr(node.value)
        if isinstance(base, SensorGroup):
            i = self.const_index(node.slice)
            if not 0 <= i <= 2:
                raise self.err(f"a sensor triple has indices 0-2, not {i}", node.slice)
            f = base.prefix + "xyz"[i]
            self.uses.add(f)
            return N.Sensor(f, loc=self.loc(node))
        if not isinstance(base, list):
            raise self.err(f"{node_text(node.value)} is not a list", node.value)
        e = self.index(base, node)
        if isinstance(e, N.Var):
            return N.Load(e, loc=self.loc(node))
        return e

    def const_index(self, sl):
        if isinstance(sl, ast.Slice):
            raise self.unsupported_expr(sl)
        i = self.expr(sl) if not self.is_static(sl) else self.const_expr(sl)
        if not isinstance(i, int) or isinstance(i, bool):
            raise self.err("a list index must be a constant integer", sl,
                           "the DMP has no indirect addressing: unroll with `for i in range(N)`")
        return i

    def index(self, base, node, store=False):
        i = self.const_index(node.slice)
        if not -len(base) <= i < len(base):
            raise self.err(f"index {i} is outside the list (length {len(base)})", node.slice)
        return base[i]

    def ex_List(self, node):
        return [self.expr(e) for e in node.elts]

    ex_Tuple = ex_List

    def ex_ListComp(self, node):
        if len(node.generators) != 1:
            raise self.err("one `for` per comprehension", node)
        g = node.generators[0]
        if g.ifs or g.is_async or not isinstance(g.target, ast.Name):
            raise self.err("a comprehension is [expr for i in range(N)]", node)
        if not (isinstance(g.iter, ast.Call) and self.callee(g.iter.func) == ("builtin", "range")):
            raise self.err("a comprehension runs over range(N)", g.iter)
        args = [self.const_expr(a) for a in g.iter.args]
        fr = self.f
        saved = fr.names.get(g.target.id, None), g.target.id in fr.assigned
        out = []
        for i in range(*args):
            fr.names[g.target.id] = i
            fr.assigned.add(g.target.id)
            out.append(self.expr(node.elt))
        if saved[0] is None:
            fr.names.pop(g.target.id, None)
        else:
            fr.names[g.target.id] = saved[0]
        return out

    def ex_IfExp(self, node):
        c = self.cond(node.test)
        if isinstance(c, bool):
            return self.expr(node.body if c else node.orelse)
        a, b = self.expr(node.body), self.expr(node.orelse)
        if isinstance(a, list) or isinstance(b, list):
            raise self.err("a conditional expression chooses between values, not lists", node)
        t = self.unify([a, b], [node.body, node.orelse], "this conditional expression")
        ea, eb = self.as_type(a, t, node.body), self.as_type(b, t, node.orelse)
        if ea.type == eb.type:
            t = ea.type
        return N.Select(c, ea, eb, t, loc=self.loc(node))

    def ex_Compare(self, node):
        raise self.err("a comparison is not a value", node, "write `x = 1 if a > b else 0`")

    ex_BoolOp = ex_Compare

    def ex_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            raise self.err("`not` makes a condition, not a value", node, "write `x = 0 if c else 1`")
        if isinstance(node.op, ast.Invert):
            v = self.expr(node.operand)
            if is_const(v):
                return self.fold(node)
            return N.BitOp("xor", self.as_type(v, N.BITS, node.operand),
                           N.Const(N.BITS, -1), loc=self.loc(node))
        return self.arith(node)

    def ex_BinOp(self, node):
        op = node.op
        if isinstance(op, ast.Mult) and (self.is_list_node(node.left) or self.is_list_node(node.right)):
            lst, cnt = (node.left, node.right) if self.is_list_node(node.left) else (node.right, node.left)
            k = self.const_expr(cnt)
            if not isinstance(k, int) or k < 0:
                raise self.err("a list repeats a constant number of times", cnt)
            items = self.expr(lst)
            return [x for _ in range(k) for x in items]
        if isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            return self.arith(node)
        if self.is_static(node):
            return self.fold(node)
        if isinstance(op, ast.Pow):
            raise self.err("** is not supported on DMP values", node, "write x * x")
        if isinstance(op, (ast.FloorDiv, ast.Mod)):
            raise self.err(f"{'//' if isinstance(op, ast.FloorDiv) else '%'} is not supported on DMP values",
                           node, HINT_DIV if isinstance(op, ast.FloorDiv) else None)
        if isinstance(op, ast.MatMult):
            raise self.err("@ is not supported", node)
        if isinstance(op, (ast.BitAnd, ast.BitXor, ast.BitOr)):
            a, b = self.expr(node.left), self.expr(node.right)
            for v, n in ((a, node.left), (b, node.right)):
                if self.kind_of(v, n) in (N.Q30, N.INT):
                    raise self.err(f"{node_text(n)} is {self.tname(self.value(v, n).type)}: & ^ | work on "
                                   "dmp.bits values", n, "make masks with dmp.bits(0x...)")
            opn = {ast.BitAnd: "and", ast.BitXor: "xor", ast.BitOr: "or"}[type(op)]
            return N.BitOp(opn, self.as_type(a, N.BITS, node.left), self.as_type(b, N.BITS, node.right),
                           loc=self.loc(node))
        if isinstance(op, (ast.LShift, ast.RShift)):
            a = self.expr(node.left)
            k = self.const_expr(node.right) if self.is_static(node.right) else None
            if self.kind_of(a, node.left) != N.BITS:
                raise self.err("<< and >> work on dmp.bits values", node,
                               "to scale a Q30 value, multiply: x * 4, x * 0.25")
            if not isinstance(k, int) or not 0 <= k <= 31:
                raise self.err("a shift count is a constant 0-31", node.right)
            return N.Shift("shl" if isinstance(op, ast.LShift) else "shr",
                           self.as_type(a, N.BITS, node.left), k, loc=self.loc(node))
        raise self.unsupported_expr(node)

    @staticmethod
    def is_list_node(n):
        return isinstance(n, (ast.List, ast.ListComp, ast.Tuple))

    # ------------------------------------------------------------------ sums
    def arith(self, node):
        """+ - * / and unary minus: one exact Sum over the whole +/- tree."""
        if self.is_static(node):
            return self.fold(node)
        terms = []
        self.collect(node, 1, terms)
        return self.build_sum(terms, node)

    def collect(self, node, coef, terms):
        if self.is_static(node):
            v = self.fold(node)
            if not is_const(v):
                raise self.err(f"{node_text(node)} is not a number", node)
            terms.append((coef, "const", v, node))
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            if self.is_list_node(node.left) or self.is_list_node(node.right):
                raise self.err("+ on lists is not supported", node)
            self.collect(node.left, coef, terms)
            self.collect(node.right, -coef if isinstance(node.op, ast.Sub) else coef, terms)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            self.collect(node.operand, -coef if isinstance(node.op, ast.USub) else coef, terms)
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for k_node, x_node in ((node.left, node.right), (node.right, node.left)):
                if self.is_static(k_node):
                    k = self.fold(k_node)
                    if isinstance(k, int) and not isinstance(k, bool):
                        if k:
                            self.collect(x_node, coef * k, terms)
                        return
            sa, fa = self.strip_neg(node.left)
            sb, fb = self.strip_neg(node.right)
            a, b = self.factor(fa), self.factor(fb)
            terms.append((coef * sa * sb, "prod", (a, b), node))
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if not self.is_static(node.right):
                raise self.err("division by a variable: the DMP has no divide", node, HINT_DIV)
            c = self.fold(node.right)
            if not is_number(c) or c == 0:
                raise self.err(f"division by {node_text(node.right)}", node.right)
            sa, fa = self.strip_neg(node.left)
            a = self.factor(fa)
            if not isinstance(a, N.Expr) or N.arith_type(a.type) != N.Q30:
                raise self.err("/ divides Q30 values only", node, "Int and dmp.bits values do not divide")
            recip = N.Const(N.Q30, self.q30(1 / c, node.right, f"1 / {node_text(node.right)}"),
                            loc=self.loc(node.right))
            terms.append((coef * sa, "prod", (a, recip), node))
            return
        v = self.expr(node)
        terms.append((coef, "val", self.value(v, node), node))

    @staticmethod
    def strip_neg(node):
        s = 1
        while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            if isinstance(node.op, ast.USub):
                s = -s
            node = node.operand
        return s, node

    def factor(self, node) -> N.Expr:
        v = self.expr(node)
        if is_const(v):
            if not is_number(v):
                raise self.err("dmp.bits do not multiply", node)
            return N.Const(N.Q30, self.q30(v, node, node_text(node)), loc=self.loc(node))
        e = self.value(v, node)
        t = N.arith_type(e.type)
        if t == N.INT:
            raise self.err(f"{node_text(node)} is an Int: Int values only scale by a constant integer",
                           node, "n * 3 is fine; a Q30 value times an Int is not")
        if t == N.BITS:
            raise self.err("dmp.bits values do not multiply", node, "use << for powers of two")
        return e

    def build_sum(self, terms, node):
        kinds = set()
        for coef, kind, x, n in terms:
            if kind == "prod":
                kinds.add(N.Q30)
            elif kind == "val":
                kinds.add(N.arith_type(x.type))
        if N.BITS in kinds:
            raise self.err("+ and - do not apply to dmp.bits", node, "combine masks with ^ and |")
        if len(kinds) > 1:
            raise self.err("this expression mixes an Int and a Q30 value", node, HINT_MIX)
        typ = kinds.pop() if kinds else N.Q30
        out = []
        for coef, kind, x, n in terms:
            if kind == "const":
                if typ == N.INT and not isinstance(x, int):
                    raise self.err(f"{node_text(n)} is not an integer, but this is Int arithmetic", n,
                                   HINT_MIX)
                out.append(N.Term(coef, self.as_type(x, typ, n)))
            elif kind == "prod":
                out.append(N.Term(coef, N.Prod(*x)))
            else:
                out.append(N.Term(coef, x))
        out = [t for t in out if not (isinstance(t.x, N.Const) and t.x.value == 0)]
        if len(out) == 1 and out[0].coef == 1 and not isinstance(out[0].x, N.Prod):
            return out[0].x
        if not out:
            return N.Const(typ, 0, loc=self.loc(node))
        return N.Sum(typ, out, loc=self.loc(node))

    # ------------------------------------------------------------------ constants
    def is_static(self, node) -> bool:
        return self.folder.is_static(node)

    def fold(self, node):
        return self.folder.fold(node)

    def pure_path(self, node) -> bool:
        return self.folder.pure_path(node)

    def const_expr(self, node):
        if not self.is_static(node):
            raise self.err(f"{node_text(node)} must be a constant", node)
        return self.fold(node)

    # ------------------------------------------------------------------ calls
    def callee(self, func, quiet=False):
        """('builtin'|'math'|'dmp'|'helper'|'sub', what)"""
        try:
            f = self.expr(func)
        except DmpError:
            if quiet:
                return None
            raise
        if isinstance(f, Builtin):
            return ("builtin", f.name)
        if isinstance(f, ApiFn):
            return ("dmp", f.name)
        if isinstance(f, api.Sub):
            return ("sub", f)
        if is_math(f):
            return ("math", f)
        if isinstance(f, types.FunctionType):
            return ("helper", f)
        if quiet:
            return None
        raise self.err(f"{node_text(func)} is not something a DMP program can call", func)

    def ex_Call(self, node):
        return self.call(node, stmt=False)

    def call(self, node, stmt):
        c = self.callee(node.func)
        kind, what = c
        if kind == "helper" or kind == "sub":
            if node.keywords:
                raise self.err("helpers take positional arguments", node)
            args = [self.expr(a) for a in node.args]
            if kind == "helper":
                return self.inline(what, args, node)
            return self.call_sub(what, args, node)
        if kind == "math":
            if self.is_static(node):
                return self.fold(node)
            raise self.err(f"{node_text(node.func)} needs constant arguments: the DMP has no {what.__name__}",
                           node, "compute it at module level, or see examples/unit_vectors.py for a "
                           "Newton iteration" if what.__name__ in ("sqrt", "hypot") else
                           "compute it at module level")
        if kind == "builtin":
            if what == "range":
                raise self.err("range() belongs in a for loop", node)
            if what == "len":
                (a,) = node.args
                v = self.expr(a)
                if not isinstance(v, list):
                    raise self.err("len() of a list", node)
                return len(v)
            if self.is_static(node):
                return self.fold(node)
            if what == "abs":
                (a,) = self.args(node, 1)
                v = self.expr(a)
                e = self.value(v, a)
                t = N.arith_type(e.type)
                if t == N.BITS:
                    raise self.err("abs of dmp.bits", node)
                return N.Abs(e, t, loc=self.loc(node))
            if what in ("min", "max"):
                if len(node.args) < 2 or node.keywords:
                    raise self.err(f"{what}(a, b, ...) of at least two values", node)
                vals = [self.expr(a) for a in node.args]
                if any(isinstance(v, list) for v in vals):
                    raise self.err(f"{what} of values, not lists", node)
                t = self.unify(vals, node.args, what)
                if t == N.BITS:
                    raise self.err(f"{what} of dmp.bits", node)
                es = [self.as_type(v, t, a) for v, a in zip(vals, node.args)]
                acc = es[0]
                for e in es[1:]:
                    acc = N.MinMax(what, acc, e, t, loc=self.loc(node))
                return acc
            raise self.err(f"{what}() needs constant arguments", node)
        # dmp.*
        if what == "emit":
            if not stmt:
                raise self.err("dmp.emit is a statement", node)
            self.do_emit(node)
            return None
        if what == "irq":
            if not stmt:
                raise self.err("dmp.irq is a statement", node)
            if node.args or node.keywords:
                raise self.err("dmp.irq() takes no arguments", node)
            self.emit_stmt(N.Irq(self.loc(node)))
            return None
        if what in ("clamp", "assume"):
            x, lo_n, hi_n = self.args(node, 3)
            lo, hi = self.const_expr(lo_n), self.const_expr(hi_n)
            if not (is_number(lo) and is_number(hi)) or lo > hi:
                raise self.err(f"dmp.{what}(x, lo, hi) needs constant bounds with lo <= hi", node)
            v = self.expr(x)
            if isinstance(v, list):
                raise self.err(f"dmp.{what} of a value, not a list", x)
            e = self.value(v, x)
            t = N.arith_type(e.type)
            if t == N.BITS:
                raise self.err(f"dmp.{what} of dmp.bits", x)
            if what == "assume":
                if stmt:
                    self.emit_stmt(N.Check(e, float(lo), float(hi), self.loc(node)))
                    return None
                return N.Assume(e, float(lo), float(hi), loc=self.loc(node))
            lo_e, hi_e = self.as_type(lo, t, lo_n), self.as_type(hi, t, hi_n)
            return N.MinMax("min", N.MinMax("max", e, lo_e, t, loc=self.loc(node)), hi_e, t,
                            loc=self.loc(node))
        if what == "bits":
            return self.const_expr(node)
        if what == "loop":
            raise self.err("dmp.loop() belongs in a for loop: `for _ in dmp.loop(N):`", node)
        raise self.err(f"dmp.{what} cannot be called here", node)  # pragma: no cover

    def args(self, node, n):
        if node.keywords or len(node.args) != n:
            raise self.err(f"{node_text(node.func)} takes {n} argument{'s' if n > 1 else ''}", node)
        return node.args

    # ------------------------------------------------------------------ emit
    def do_emit(self, node):
        fr = self.f
        if fr.kind != "program":
            raise self.err("dmp.emit belongs in the program body, not in a helper", node,
                           "return the value and emit it in the program")
        if self.depth or self.cond_return:
            raise self.err("dmp.emit under a condition", node,
                           "v1 packets have a fixed length: emit unconditionally, and send a flag "
                           "field (1 if cond else 0) instead")
        fields = []
        for a in node.args:
            if isinstance(a, ast.Starred):
                raise self.err("emit the elements one by one: dmp.emit(w=q[0], x=q[1], ...)", a)
            fields.append((self.field_name(a), a))
        for kw in node.keywords:
            if kw.arg is None:
                raise self.err("**kwargs is not supported", kw.value)
            fields.append((kw.arg, kw.value))
        if not fields:
            raise self.err("dmp.emit() with nothing to send", node)
        out = []
        for name, a in fields:
            v = self.expr(a)
            if isinstance(v, list):
                raise self.err(f"{name} is a list: emit its elements", a,
                               f"dmp.emit({name}0={node_text(a)}[0], ...)")
            out.append((name, self.value(v, a)))
        self.emit_stmt(N.Emit(out, self.loc(node)))

    def field_name(self, a):
        if isinstance(a, ast.Name):
            return a.id
        v = None
        if self.pure_path(a):
            try:
                v = self.expr(a)
            except DmpError:
                pass
        if isinstance(v, N.Sensor):
            return FIELD_NAME[v.field]
        if isinstance(a, ast.Subscript):
            idx = []
            n = a
            while isinstance(n, ast.Subscript):
                if not self.is_static(n.slice):
                    break
                idx.append(str(self.fold(n.slice)))
                n = n.value
            if isinstance(n, ast.Name):
                return n.id + "_".join(reversed(idx))
        raise self.err(f"this packet field needs a name: dmp.emit(name={node_text(a)})", a)

    # ------------------------------------------------------------------ helpers and subs
    def inline(self, fn, args, node):
        if fn in self.calls:
            raise self.err(f"{fn.__name__} calls itself: recursion is not supported", node,
                           "helpers are inlined; write the recursion as a dmp.loop")
        src = function_source(fn)
        a = src.node.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.defaults or a.posonlyargs:
            raise self.err(f"{fn.__name__}: helpers take plain positional parameters (no defaults, "
                           "*args or keywords)", node)
        if len(args) != len(a.args):
            raise self.err(f"{fn.__name__} takes {len(a.args)} arguments, not {len(args)}", node)
        fr = Frame(src, "helper")
        for p, v, an in zip(a.args, args, node.args):
            if is_const(v) or isinstance(v, (SensorNS, SensorGroup)):
                fr.names[p.arg] = v
            elif isinstance(v, list):
                fr.names[p.arg] = v if self.is_named(an) else self.copy_list(v, p.arg, an)
            else:
                e = self.value(v, an)
                t = self.new_var(f"{fn.__name__}.{p.arg}#{self.n + 1}", "temp", e.type, an)
                self.n += 1
                self.emit_stmt(N.Assign(t, e, self.loc(an)))
                fr.names[p.arg] = t
            fr.assigned.add(p.arg)
        fr.ret_label = self.label(f"{fn.__name__}.ret")
        self.calls.append(fn)
        self.frames.append(fr)
        saved_out, saved_falls, saved_depth = self.out, self.falls, self.depth
        self.out, self.falls = [], True
        try:
            body = [s for s in src.node.body if not self.is_docstring(s, src.node)]
            self.block(body)
            if fr.ret is not None and self.falls:
                raise DmpError(f"{fn.__name__} does not return a value on every path",
                               Loc.of(src.node, src.file))
            stmts = self.out
        finally:
            self.frames.pop()
            self.calls.pop()
            self.out, self.falls, self.depth = saved_out, saved_falls, saved_depth
        if self.contains(src.node.body, ast.Return):
            if stmts and isinstance(stmts[-1], N.Break) and stmts[-1].label == fr.ret_label and \
                    not self.has_break(stmts[:-1], fr.ret_label):
                stmts = stmts[:-1]                  # a single return at the end needs no block
            else:
                stmts = [N.Block(fr.ret_label, stmts)]
        for s in stmts:
            self.emit_stmt(s)
        if fr.ret is None:
            return None
        return self.load_result(fr.ret, node)

    def has_break(self, stmts, label):
        for s in stmts:
            if isinstance(s, N.Break) and s.label == label:
                return True
            if isinstance(s, N.If) and (self.has_break(s.then, label) or self.has_break(s.els, label)):
                return True
            if isinstance(s, (N.Block, N.Loop)) and self.has_break(s.body, label):
                return True
        return False

    def load_result(self, r, node):
        if isinstance(r, list):
            return [self.load_result(x, node) for x in r]
        return N.Load(r, loc=self.loc(node))

    def call_sub(self, s, args, node):
        fn = s.fn
        if fn in self.calls:
            raise self.err(f"{fn.__name__} calls itself: recursion is not supported", node)
        vals = []
        for v, an in zip(args, node.args):
            if isinstance(v, list):
                raise self.err(f"a @dmp.sub takes values, not lists ({node_text(an)})", an,
                               "pass the elements, or make it a plain helper (inlined)")
            vals.append((self.value(v, an), an))
        sub = self.subs.get(fn.__name__)
        src = function_source(fn)
        a = src.node.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.defaults or a.posonlyargs:
            raise self.err(f"{fn.__name__}: a sub takes plain positional parameters", node)
        if len(vals) != len(a.args):
            raise self.err(f"{fn.__name__} takes {len(a.args)} arguments, not {len(vals)}", node)
        if sub is None:
            fr = Frame(src, "sub")
            params = []
            for p, (e, an) in zip(a.args, vals):
                t = N.Q30 if e.type == N.S16 else e.type
                pv = self.new_var(f"{fn.__name__}.{p.arg}", "arg", t, p)
                params.append(pv)
                fr.names[p.arg] = pv
                fr.assigned.add(p.arg)
            fr.ret_label = self.label(f"{fn.__name__}.ret")
            sub = N.SubIR(fn.__name__, params, [], [], Loc.of(src.node, src.file))
            self.subs[fn.__name__] = sub
            self.calls.append(fn)
            self.frames.append(fr)
            saved = self.out, self.falls, self.depth, self.cond_return
            self.out, self.falls, self.depth = [], True, 0
            try:
                body = [st for st in src.node.body if not self.is_docstring(st, src.node)]
                self.block(body)
                if fr.ret is not None and self.falls:
                    raise DmpError(f"{fn.__name__} does not return a value on every path",
                                   Loc.of(src.node, src.file))
                stmts = self.out
            finally:
                self.frames.pop()
                self.calls.pop()
                self.out, self.falls, self.depth, self.cond_return = saved
            if self.contains(src.node.body, ast.Return):
                stmts = [N.Block(fr.ret_label, stmts)]
            sub.body = stmts
            flat = []

            def fl(r):
                if isinstance(r, list):
                    for x in r:
                        fl(x)
                else:
                    flat.append(r)
            if fr.ret is not None:
                fl(fr.ret)
            sub.results = flat
            sub.shape = fr.ret
        for pv, (e, an) in zip(sub.params, vals):
            if N.arith_type(e.type) != pv.type:
                raise self.err(f"{fn.__name__}'s {pv.name.split('.')[-1]} is {self.tname(pv.type)} "
                               f"(from its first call), but this is {self.tname(e.type)}", an)
        outs = [self.temp(r.type, f"{fn.__name__}.out", node) for r in sub.results]
        self.emit_stmt(N.CallSub(sub, [e for e, _ in vals], outs, self.loc(node)))
        if not outs:
            return None
        it = iter(outs)

        def shape(r):
            if isinstance(r, list):
                return [shape(x) for x in r]
            return N.Load(next(it), loc=self.loc(node))
        return shape(sub.shape)

    # ------------------------------------------------------------------ conditions
    def cond(self, node):
        """A condition: an IR Cmp/BoolOp/Not, or a Python bool when it is constant."""
        if isinstance(node, ast.Compare):
            return self.compare(node)
        if isinstance(node, ast.BoolOp):
            parts = [self.cond(v) for v in node.values]
            is_and = isinstance(node.op, ast.And)
            acc = None
            for p in parts:
                if isinstance(p, bool):
                    if p != is_and:
                        return p                    # False in an `and`, True in an `or`
                    continue
                acc = p if acc is None else N.BoolOp("and" if is_and else "or", acc, p)
            return is_and if acc is None else acc
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            c = self.cond(node.operand)
            return (not c) if isinstance(c, bool) else N.Not(c)
        v = self.expr(node)
        if is_const(v):
            return bool(v.value if isinstance(v, api.Bits) else v)
        e = self.value(v, node)
        zero = N.Const(N.arith_type(e.type), 0)
        return N.Cmp("ne", e, zero, self.loc(node))

    def compare(self, node):
        left = self.expr(node.left)
        lnode = node.left
        out = []
        for i, (op, rnode) in enumerate(zip(node.ops, node.comparators)):
            if type(op) not in CMP_OPS:
                raise self.err(f"{type(op).__name__.lower()} comparisons are not supported", node,
                               "compare values with < <= > >= == !=")
            right = self.expr(rnode)
            if isinstance(left, list) or isinstance(right, list):
                raise self.err("lists do not compare", node)
            if is_const(left) and is_const(right):
                c = self.fold_cmp(type(op), left, right)
            else:
                t = self.unify([left, right], [lnode, rnode], "this comparison")
                a, b = self.as_type(left, t, lnode), self.as_type(right, t, rnode)
                if t == N.BITS and type(op) not in (ast.Eq, ast.NotEq):
                    raise self.err("dmp.bits compare with == and != only", node)
                if i < len(node.ops) - 1 and not isinstance(b, (N.Const, N.Load, N.Sensor)):
                    b = self.materialize(b, rnode)
                    right = b
                c = N.Cmp(CMP_OPS[type(op)], a, b, self.loc(node))
            out.append(c)
            left, lnode = right, rnode
        acc = None
        for c in out:
            if c is False:
                return False
            if c is True:
                continue
            acc = c if acc is None else N.BoolOp("and", acc, c)
        return True if acc is None else acc

    @staticmethod
    def fold_cmp(op, a, b):
        a = a.value if isinstance(a, api.Bits) else a
        b = b.value if isinstance(b, api.Bits) else b
        return {ast.Lt: a < b, ast.LtE: a <= b, ast.Gt: a > b, ast.GtE: a >= b, ast.Eq: a == b,
                ast.NotEq: a != b}[op]


def lower(program) -> N.ProgramIR:
    """Lower a dmp.Program (its source) to the IR.  Raises DmpError on anything not in the
    language."""
    return Lowerer(program).run()
