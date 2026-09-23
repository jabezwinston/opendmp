"""Range analysis: where can a value wrap?

Interval arithmetic over the IR, in raw words.  Because a Sum is exact and wraps once, the only
places a program can wrap are a Sum's result, an integer multiple, and abs(-2.0); a compare
never overflows.  So a warning names one of those, says where the wrapped value goes, and
suggests the fix.

What it assumes:
- sensor samples span their full int16 range;
- a state or param with a declared range stays in it (checked: storing something wider is a
  warning), and one without is inferred by running the pass until the intervals settle (a
  state that keeps growing is widened to the full range);
- `dmp.assume(x, lo, hi)` holds (the simulator checks it), and silences a wrap warning on x
  itself: it is how to state what interval arithmetic cannot see (|q| = 1, a stable filter);
- a branch on `v < c` narrows v inside it.

Int arithmetic (counters) is tracked but never warned about.  The intervals of every
expression are kept for the backend (whether MIN's `b - a` can overflow, for one).
"""
import linecache
from dataclasses import dataclass, field

from ..lang.errors import Warning_
from ..q30 import ONE
from . import nodes as N

LO, HI = -(1 << 31), (1 << 31) - 1
FULL = (LO, HI)
S16_RANGE = (LO, HI - 0xFFFF)


@dataclass
class Report:
    warnings: list = field(default_factory=list)
    intervals: dict = field(default_factory=dict)       # id(expr) -> (lo, hi) raw
    states: dict = field(default_factory=dict)          # Var -> (lo, hi) raw, at the pass start


def join(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]))


def join_env(a, b):
    if a is None:
        return None if b is None else dict(b)
    if b is None:
        return dict(a)
    out = dict(a)
    for k, v in b.items():
        out[k] = join(out.get(k), v)
    return out


def fmt(iv, typ=N.Q30):
    if typ in (N.Q30, N.S16):
        return f"[{iv[0] / ONE:.4g}, {iv[1] / ONE:.4g}]"
    return f"[{iv[0]}, {iv[1]}]"


def text_at(loc) -> str:
    if loc is None:
        return "this expression"
    if loc.line == loc.end_line:
        t = linecache.getline(loc.file, loc.line)[loc.col:loc.end_col]
        if t:
            return f"`{t}`"
    return "this expression"


class Analyser:
    WIDEN_AFTER = 6

    def __init__(self, ir: N.ProgramIR):
        self.ir = ir
        self.report = Report()
        self.warn_on = False
        self.pending: dict = {}           # (loc, kind) -> [lo, hi, where, subject, typ, hint]
        self.breaks: dict = {}

    # ------------------------------------------------------------------ driver
    def run(self) -> Report:
        start = {}
        for v in self.ir.vars:
            if v.kind in ("state", "param"):
                start[v] = self.declared(v) if v.range else (FULL if v.kind == "param" else (v.init, v.init))
        for it in range(3 * self.WIDEN_AFTER):
            out = self.pass_(dict(start))
            new = dict(start)
            changed = False
            for v in start:
                if v.kind == "state" and not v.range:
                    j = join(start[v], out.get(v, start[v]))
                    if j != start[v]:
                        changed = True
                        new[v] = FULL if it >= self.WIDEN_AFTER else j
            start = new
            if not changed:
                break
        # narrowing: from the (possibly widened) fixpoint, a state starts a pass with its initial
        # value or whatever a pass stores (a clamped state comes back from the full range)
        for _ in range(4):
            out = self.pass_(dict(start))
            new = dict(start)
            for v in start:
                if v.kind == "state" and not v.range:
                    j = join((v.init, v.init), out.get(v, start[v]))
                    new[v] = (max(j[0], start[v][0]), min(j[1], start[v][1]))
            if new == start:
                break
            start = new
        self.warn_on = True
        self.report.intervals = {}                  # the final pass: every evaluation, joined
        self.pass_(dict(start))
        self.report.states = start
        for (loc, kind), (lo, hi, where, subject, typ, hint) in self.pending.items():
            if kind == "wrap":
                msg = f"{text_at(loc)} can reach {fmt((lo, hi))}, outside [-2, 2): it wraps {where}"
            elif kind == "abs":
                msg = f"{text_at(loc)} can take abs(-2.0), which wraps to -2.0"
            else:
                msg = f"{subject} may leave its declared range {where}: this can store {fmt((lo, hi), typ)}"
            self.report.warnings.append(Warning_(msg, loc, hint))
        self.report.warnings.sort(key=lambda w: (w.loc.file, w.loc.line, w.loc.col) if w.loc else ("", 0, 0))
        return self.report

    def declared(self, v):
        lo, hi = v.range
        if v.type in (N.Q30, N.S16):
            return (max(LO, round(lo * ONE)), min(HI, round(hi * ONE)))
        return (int(lo), int(hi))

    def pass_(self, env):
        self.breaks = {}
        out = self.block(self.ir.body, env)
        return join_env(out, self.join_breaks("pass")) or env

    def warn(self, loc, kind, iv, where=None, hint=None, subject=None, typ=N.Q30):
        """One warning per (place, kind), over every inlined copy and loop iteration: the
        widest interval, the first context."""
        if not self.warn_on:
            return
        key = (loc, kind)
        cur = self.pending.get(key)
        if cur is None:
            self.pending[key] = [iv[0], iv[1], where, subject, typ, hint]
        else:
            cur[0], cur[1] = min(cur[0], iv[0]), max(cur[1], iv[1])

    # ------------------------------------------------------------------ statements
    def join_breaks(self, label):
        envs = self.breaks.pop(label, [])
        acc = None
        for e in envs:
            acc = join_env(acc, e)
        return acc

    def block(self, stmts, env):
        """The env after stmts, or None when control cannot fall through."""
        for s in stmts:
            if env is None:
                return None
            env = self.stmt(s, env)
        return env

    def stmt(self, s, env):
        if isinstance(s, N.Assign):
            v = s.var
            where = f"before it is stored in {v.name}" if v.kind in ("state", "param") else \
                f"in {display(v)}"
            iv = self.iv(s.value, env, where)
            if v.range and v.kind == "state":
                d = self.declared(v)
                if iv[0] < d[0] or iv[1] > d[1]:
                    self.warn(s.loc, "range", iv, fmt(d, v.type),
                              "clamp the value (dmp.clamp) where it is computed, widen the range, or "
                              "dmp.assume(...) what interval arithmetic cannot see",
                              v.name.split("[")[0], v.type)
            env = dict(env)
            env[v] = iv
            return env
        if isinstance(s, N.If):
            self.cond_iv(s.cond, env)
            t = self.block(s.then, self.refine(env, s.cond, True))
            e = self.block(s.els, self.refine(env, s.cond, False))
            return join_env(t, e)
        if isinstance(s, N.Block):
            out = self.block(s.body, env)
            return join_env(out, self.join_breaks(s.label))
        if isinstance(s, N.Loop):
            head, last = env, None
            n = s.count if s.count is not None else 64
            for i in range(n):
                out = self.block(s.body, head)
                if out is None:
                    break
                last = out
                new = join_env(head, out)
                if new == head:
                    break
                if i >= self.WIDEN_AFTER:
                    new = {k: (FULL if new[k] != head.get(k) else new[k]) for k in new}
                head = new
            if s.count is None:
                last = None                          # a while loop leaves only by break
            return join_env(last, self.join_breaks(s.label))
        if isinstance(s, N.Break):
            self.breaks.setdefault(s.label, []).append(env)
            return None
        if isinstance(s, N.Emit):
            for name, e in s.fields:
                self.iv(e, env, f"before it is emitted as {name}")
            return env
        if isinstance(s, N.Irq):
            return env
        if isinstance(s, N.Check):
            self.iv(s.x, env, None)
            if isinstance(s.x, N.Load):
                env = dict(env)
                env[s.x.var] = self.assumed(s.x.type, s.lo, s.hi)
            return env
        if isinstance(s, N.CallSub):
            inner = dict(env)
            for p, a in zip(s.sub.params, s.args):
                inner[p] = self.iv(a, env, f"as {s.sub.name}'s {p.name.split('.')[-1]}")
            saved = self.breaks
            self.breaks = {}
            out = self.block(s.sub.body, inner)
            self.breaks = saved
            env = dict(env)
            for o, r in zip(s.outs, s.sub.results):
                env[o] = (out or inner).get(r, FULL)
            return env
        raise TypeError(type(s).__name__)

    def assumed(self, typ, lo, hi):
        if typ in (N.Q30, N.S16):
            return (max(LO, round(lo * ONE)), min(HI, round(hi * ONE)))
        return (int(lo), int(hi))

    # ------------------------------------------------------------------ conditions
    def refine(self, env, c, truth):
        if isinstance(c, N.Not):
            return self.refine(env, c.x, not truth)
        if isinstance(c, N.BoolOp):
            if (c.op == "and") == truth:              # both parts hold
                return self.refine(self.refine(env, c.a, truth), c.b, truth)
            return env
        if not isinstance(c, N.Cmp):
            return env
        op, a, b = c.op, c.a, c.b
        if isinstance(a, N.Const) and isinstance(b, N.Load):
            a, b = b, a
            op = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le"}.get(op, op)
        if not (isinstance(a, N.Load) and isinstance(b, N.Const)):
            return env
        if not truth:
            op = {"lt": "ge", "le": "gt", "gt": "le", "ge": "lt", "eq": "ne", "ne": "eq"}[op]
        cur = env.get(a.var, FULL)
        k = b.value
        lo, hi = cur
        if op == "lt":
            hi = min(hi, k - 1)
        elif op == "le":
            hi = min(hi, k)
        elif op == "gt":
            lo = max(lo, k + 1)
        elif op == "ge":
            lo = max(lo, k)
        elif op == "eq":
            lo, hi = max(lo, k), min(hi, k)
        if lo > hi:
            lo, hi = cur                              # infeasible: leave it
        env = dict(env)
        env[a.var] = (lo, hi)
        return env

    # ------------------------------------------------------------------ values
    def iv(self, e, env, where, quiet=False):
        r = self._iv(e, env, where, quiet)
        prev = self.report.intervals.get(id(e))
        self.report.intervals[id(e)] = r if prev is None or not self.warn_on else join(prev, r)
        return r

    def cond_iv(self, c, env):
        """Evaluate a condition's operands (their intervals matter: a compare's subtrahend at
        -2.0 needs a guard, and a sum that wraps before a compare is a warning)."""
        if isinstance(c, N.Cmp):
            self.iv(c.a, env, "before the comparison")
            self.iv(c.b, env, "before the comparison")
        elif isinstance(c, N.BoolOp):
            self.cond_iv(c.a, env)
            self.cond_iv(c.b, env)
        elif isinstance(c, N.Not):
            self.cond_iv(c.x, env)

    def _iv(self, e, env, where, quiet):
        if isinstance(e, N.Const):
            return (e.value, e.value)
        if isinstance(e, N.Load):
            return env.get(e.var, FULL)
        if isinstance(e, N.Sensor):
            return (0, 1) if e.field == "fsync" else S16_RANGE
        if isinstance(e, N.Sum):
            return self.sum_iv(e, env, where, quiet)
        if isinstance(e, N.MinMax):
            a = self.iv(e.a, env, "in min/max")
            b = self.iv(e.b, env, "in min/max")
            if e.op == "min":
                return (min(a[0], b[0]), min(a[1], b[1]))
            return (max(a[0], b[0]), max(a[1], b[1]))
        if isinstance(e, N.Abs):
            lo, hi = self.iv(e.x, env, "in abs()")
            if lo >= 0:
                return (lo, hi)
            if lo == LO and e.type != N.INT:
                self.warn(e.loc, "abs", (lo, hi), hint="scale the argument into (-2, 2)")
                return FULL
            if hi <= 0:
                return (-hi, -lo)
            return (0, max(-lo, hi))
        if isinstance(e, N.Select):
            self.cond_iv(e.cond, env)
            return join(self.iv(e.a, env, where), self.iv(e.b, env, where))
        if isinstance(e, (N.BitOp, N.Shift)):
            if isinstance(e, N.BitOp):
                self.iv(e.a, env, None)
                self.iv(e.b, env, None)
            else:
                self.iv(e.x, env, None)
            return FULL
        if isinstance(e, N.Assume):
            self.iv(e.x, env, where, quiet=True)
            return self.assumed(e.x.type, e.lo, e.hi)
        raise TypeError(type(e).__name__)

    def sum_iv(self, e: N.Sum, env, where, quiet):
        lo = hi = 0
        loose = []
        for t in e.terms:
            if isinstance(t.x, N.Prod):
                a = self.iv(t.x.a, env, "before the multiply")
                b = self.iv(t.x.b, env, "before the multiply")
                ps = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
                tl, th = min(ps), max(ps)
                if same(t.x.a, t.x.b) and tl < 0:     # a square
                    tl = 0 if a[0] <= 0 <= a[1] else min(a[0] * a[0], a[1] * a[1])
            else:
                tl, th = self.iv(t.x, env, where)
                if e.type == N.Q30:
                    tl, th = tl << 30, th << 30
                if isinstance(t.x, N.Load) and t.x.var.kind in ("state", "param") and not t.x.var.range:
                    loose.append(t.x.var.name)
            if t.coef < 0:
                tl, th = th, tl
            lo += t.coef * tl
            hi += t.coef * th
        if e.type == N.Q30:
            lo, hi = lo >> 30, hi >> 30
        if lo >= LO and hi <= HI:
            return (lo, hi)
        if e.type == N.Q30 and not quiet and where is not None:
            hint = ("declare a range on " + ", ".join(f"`{n}`" for n in dict.fromkeys(loose))
                    + " (dmp.state(lo, hi)) if it is narrower, or " if loose else "") + \
                "scale the terms down (x * 0.5), or dmp.assume(x, lo, hi) what interval " \
                "arithmetic cannot see"
            self.warn(e.loc, "wrap", (lo, hi), where, hint)
        return FULL


def same(a, b) -> bool:
    return a is b or (isinstance(a, N.Load) and isinstance(b, N.Load) and a.var is b.var) or \
        (isinstance(a, N.Sensor) and isinstance(b, N.Sensor) and a.field == b.field)


def display(v: N.Var) -> str:
    """A temp's source name: `biquad.y#7` -> `y`."""
    return v.name.split("#")[0].split(".")[-1] if v.kind == "temp" else v.name


def analyse(ir: N.ProgramIR) -> Report:
    return Analyser(ir).run()
