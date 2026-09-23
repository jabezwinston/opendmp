"""Three-address code clean-up, before layout: copy propagation, then dead-store elimination.

Inlining and lists leave chains of plain copies (`ret = y; y[0] = ret; emit y[0]`) and stores
that are overwritten before anything reads them (`y = [0.0, 0.0, 0.0]`).  Both are removed
here, over the real control-flow graph (labels, conditional and backward branches), so a value
is propagated only where every path agrees, and a store is dropped only where no path reads it.

Only temporaries are rewritten or dropped: states must hold their value at the end of the pass,
params are the host's, constants and the sample page are read-only.
"""
from . import tac as T

UNIVERSE = None                   # "every copy": the start value of a forward must-analysis


def uses(op) -> list:
    if isinstance(op, T.Alu):
        return [op.a, op.b]
    if isinstance(op, T.MacRun):
        return [w for _, a, b in op.terms for w in (a, b)]
    if isinstance(op, T.Mov):
        # a lane MOV writes half the word: the other half is read through
        return [op.src] + ([op.dst] if op.lane is not None and op.dst is not T.FIFO else [])
    if isinstance(op, T.IncDec):
        return [op.dst]
    if isinstance(op, (T.Cmp, T.EqTest)):
        return [op.a, op.b]
    if isinstance(op, T.Call):
        return list(op.reads)
    return []


def defs(op) -> list:
    if isinstance(op, T.Call):
        return list(op.writes)
    if isinstance(op, T.Cmp) or isinstance(op, T.EqTest):
        return [op.scratch]
    d = getattr(op, "dst", None)
    return [] if d is None or d is T.FIFO else [d]


def removable(op) -> bool:
    """An op whose only effect is its store (Cmp/EqTest also set F; IncDec on a counter too)."""
    return isinstance(op, (T.Alu, T.MacRun, T.Mov, T.Clr))


def successors(ops) -> list:
    where = {op.name: i for i, op in enumerate(ops) if isinstance(op, T.Label)}
    succ = []
    for i, op in enumerate(ops):
        if isinstance(op, T.Branch):
            succ.append([where[op.label]])
        elif isinstance(op, (T.End, T.Ret)):
            succ.append([])
        elif isinstance(op, T.BranchIf):
            succ.append([i + 1, where[op.label]])
        else:
            succ.append([i + 1] if i + 1 < len(ops) else [])
    return succ


def is_temp(w) -> bool:
    return w.kind == "temp"


def liveness(ops, succ):
    """live_out[i]: the words some path from op i on reads before writing."""
    n = len(ops)
    live_in = [set() for _ in range(n)]
    live_out = [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for i in range(n - 1, -1, -1):
            out = set()
            for s in succ[i]:
                out |= live_in[s]
            inn = (out - set(defs(ops[i]))) | set(uses(ops[i]))
            if out != live_out[i] or inn != live_in[i]:
                live_out[i], live_in[i] = out, inn
                changed = True
    return live_out


def meet(sets):
    """The intersection, where UNIVERSE (not yet computed) is the identity."""
    res = UNIVERSE
    for s in sets:
        if s is not UNIVERSE:
            res = s if res is UNIVERSE else res & s
    return res


def available_copies(ops, succ):
    """in_[i]: the copies t := s (plain MOVs into a temp) valid on every path into op i."""
    n = len(ops)
    preds = [[] for _ in range(n)]
    for i, ss in enumerate(succ):
        for s in ss:
            preds[s].append(i)
    in_ = [UNIVERSE] * n
    out = [UNIVERSE] * n
    changed = True
    while changed:
        changed = False
        for i in range(n):
            root = i == 0 or (isinstance(ops[i], T.Label) and ops[i].entry)
            cur = frozenset() if root else meet(out[p] for p in preds[i])
            if cur is UNIVERSE:
                continue                        # not reached (yet)
            op = ops[i]
            killed = set(defs(op))
            o = frozenset(c for c in cur if c[0] not in killed and c[1] not in killed)
            if isinstance(op, T.Mov) and op.lane is None and not op.swap and op.dst is not T.FIFO \
                    and is_temp(op.dst) and op.src is not op.dst:
                o |= {(op.dst, op.src)}
            if cur != in_[i] or o != out[i]:
                in_[i], out[i] = cur, o
                changed = True
    return in_


def propagate(ops) -> bool:
    succ = successors(ops)
    avail = available_copies(ops, succ)
    changed = False
    for i, op in enumerate(ops):
        if avail[i] is UNIVERSE or not avail[i]:
            continue
        copy = dict(avail[i])

        def sub(w, forbid=None):
            seen = set()
            while w in copy and w not in seen:
                seen.add(w)
                nxt = copy[w]
                if nxt is forbid:
                    break
                w = nxt
            return w

        if isinstance(op, T.Alu):
            a, b = sub(op.a), sub(op.b)
            if (a, b) != (op.a, op.b):
                op.a, op.b, changed = a, b, True
        elif isinstance(op, T.MacRun):
            # a term may not come to read the run's own destination, which every op overwrites
            terms = [(neg, sub(a, op.dst), sub(b, op.dst)) for neg, a, b in op.terms]
            if terms != op.terms:
                op.terms, changed = terms, True
        elif isinstance(op, T.Mov):
            s = sub(op.src)
            if s is not op.src and not (op.lane is None and not op.swap and s is op.dst):
                op.src, changed = s, True
        elif isinstance(op, (T.Cmp, T.EqTest)):
            a, b = sub(op.a), sub(op.b)
            if (a, b) != (op.a, op.b):
                op.a, op.b, changed = a, b, True
    return changed


def regions(ops) -> dict:
    """Routine entry label -> the indices of its ops (entry label to its Ret)."""
    out, cur = {}, None
    for i, op in enumerate(ops):
        if isinstance(op, T.Label) and op.entry:
            cur = op.name
            out[cur] = []
        if cur is not None:
            out[cur].append(i)
            if isinstance(op, T.Ret):
                cur = None
    return out


def summarise_calls(ops):
    """Fill each Call's reads/writes with what its routine, and what that calls, may touch."""
    reg = regions(ops)
    rw = {name: (set(), set()) for name in reg}
    for _ in range(len(reg) + 1):
        for name, idx in reg.items():
            r, w = rw[name]
            for i in idx:
                op = ops[i]
                if isinstance(op, T.Call):
                    r |= rw[op.label][0]
                    w |= rw[op.label][1]
                else:
                    r.update(uses(op))
                    w.update(defs(op))
    for op in ops:
        if isinstance(op, T.Call):
            op.reads, op.writes = frozenset(rw[op.label][0]), frozenset(rw[op.label][1])
    # a word a routine shares with its callers must keep its stores on both sides
    inside = {i for idx in reg.values() for i in idx}
    in_words = {w for i in inside for w in uses(ops[i]) + defs(ops[i])}
    out_words = {w for i, op in enumerate(ops) if i not in inside and not isinstance(op, T.Call)
                 for w in uses(op) + defs(op)}
    return in_words & out_words


def dead_stores(ops, shared=frozenset()) -> list:
    while True:
        succ = successors(ops)
        live = liveness(ops, succ)
        keep = []
        dropped = False
        for i, op in enumerate(ops):
            if removable(op):
                d = defs(op)
                if d and is_temp(d[0]) and d[0] not in live[i] and d[0] not in shared:
                    dropped = True
                    continue
                if isinstance(op, T.Mov) and op.lane is None and not op.swap and op.src is op.dst:
                    dropped = True
                    continue
            keep.append(op)
        ops = keep
        if not dropped:
            return ops


def optimize(ops: list) -> list:
    shared = summarise_calls(ops)
    for _ in range(10):
        if not propagate(ops):
            break
    return dead_stores(ops, shared)
