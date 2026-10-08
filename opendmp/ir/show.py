"""The IR as readable text (`python -m opendmp ir prog.py`)."""
from ..q30 import ONE
from . import nodes as N

CMP = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}


def const(c: N.Const) -> str:
    if c.type in (N.Q30, N.S16):
        return f"{c.value / ONE:.9g}"
    if c.type == N.BITS:
        return f"bits({c.value:#010x})"
    return str(c.value)


def expr(e) -> str:
    if isinstance(e, N.Const):
        return const(e)
    if isinstance(e, N.Load):
        return e.var.name
    if isinstance(e, N.Sensor):
        return f"s.{e.field}"
    if isinstance(e, N.Sum):
        parts = []
        for t in e.terms:
            x = f"{expr(t.x.a)}*{expr(t.x.b)}" if isinstance(t.x, N.Prod) else expr(t.x)
            k = abs(t.coef)
            body = x if k == 1 else f"{k}*{x}"
            parts.append(("- " if t.coef < 0 else "+ ") + body)
        s = " ".join(parts)
        s = s[2:] if s.startswith("+ ") else "-" + s[2:]
        return f"sum[{s}]" if len(e.terms) > 1 or isinstance(e.terms[0].x, N.Prod) or e.terms[0].coef != 1 \
            else s
    if isinstance(e, N.MinMax):
        return f"{e.op}({expr(e.a)}, {expr(e.b)})"
    if isinstance(e, N.Abs):
        return f"abs({expr(e.x)})"
    if isinstance(e, N.Select):
        return f"({expr(e.a)} if {cond(e.cond)} else {expr(e.b)})"
    if isinstance(e, N.BitOp):
        return f"({expr(e.a)} {dict(and_='&', xor='^', or_='|').get(e.op + '_', e.op)} {expr(e.b)})"
    if isinstance(e, N.Shift):
        return f"({expr(e.x)} {'<<' if e.op == 'shl' else '>>'} {e.k})"
    if isinstance(e, N.Assume):
        return f"assume({expr(e.x)}, {e.lo:g}, {e.hi:g})"
    if isinstance(e, N.Cast):
        return f"{'raw' if e.type == N.INT else 'q30'}({expr(e.x)})"
    return f"<{type(e).__name__}>"


def cond(c) -> str:
    if isinstance(c, N.Cmp):
        return f"{expr(c.a)} {CMP[c.op]} {expr(c.b)}"
    if isinstance(c, N.BoolOp):
        return f"({cond(c.a)} {c.op} {cond(c.b)})"
    if isinstance(c, N.Not):
        return f"not {cond(c.x)}"
    return str(c)


def stmts(body, ind="    ") -> list[str]:
    out = []
    for s in body:
        if isinstance(s, N.Assign):
            out.append(f"{ind}{s.var.name} = {expr(s.value)}")
        elif isinstance(s, N.If):
            out.append(f"{ind}if {cond(s.cond)}:")
            out += stmts(s.then, ind + "    ") or [ind + "    pass"]
            if s.els:
                out.append(f"{ind}else:")
                out += stmts(s.els, ind + "    ")
        elif isinstance(s, N.Block):
            out.append(f"{ind}block {s.label}:")
            out += stmts(s.body, ind + "    ")
        elif isinstance(s, N.Loop):
            out.append(f"{ind}loop {s.label} x{s.count if s.count is not None else 'inf'}:")
            out += stmts(s.body, ind + "    ")
        elif isinstance(s, N.Break):
            out.append(f"{ind}break {s.label}")
        elif isinstance(s, N.Emit):
            out.append(f"{ind}emit " + ", ".join(f"{n}={expr(e)}" for n, e in s.fields))
        elif isinstance(s, N.Irq):
            out.append(f"{ind}irq")
        elif isinstance(s, N.Check):
            out.append(f"{ind}assume {expr(s.x)} in [{s.lo:g}, {s.hi:g}]")
        elif isinstance(s, N.CallSub):
            outs = ", ".join(v.name for v in s.outs)
            out.append(f"{ind}{outs + ' = ' if outs else ''}call {s.sub.name}("
                       + ", ".join(expr(a) for a in s.args) + ")")
        else:
            out.append(f"{ind}<{type(s).__name__}>")
    return out


def program(ir: N.ProgramIR) -> str:
    lines = [f"program {ir.name}  rate {ir.opts['rate']:g} Hz  gyro {ir.opts['gyro_fs']} dps  "
             f"accel {ir.opts['accel_fs']} g"]
    for v in ir.vars:
        if v.kind in ("state", "param"):
            init = f"{v.init / ONE:.9g}" if v.type == N.Q30 else str(v.init)
            rng = f"  in [{v.range[0]:g}, {v.range[1]:g}]" if v.range else ""
            lines.append(f"  {v.kind:5s} {v.name}: {v.type} = {init}{rng}")
    lines.append("  packet " + ", ".join(f"{f.name}:{f.type}@{f.offset}" for f in ir.packet)
                 + f"  ({ir.packet_size} B)")
    for s in ir.subs.values():
        lines.append(f"  sub {s.name}(" + ", ".join(f"{p.name}:{p.type}" for p in s.params) + ") -> "
                     + ", ".join(r.name for r in s.results))
        lines += stmts(s.body, "      ")
    lines.append("  pass:")
    lines += stmts(ir.body, "    ")
    return "\n".join(lines)
