"""Run v27 (pilot): exact merging of tours on their union graph. Helper module embedded in notebook 29.

The union graph of a set of tours has the n nodes and every edge that at least one of the tours uses. A tour of the
union graph can combine parts of the given tours. This module finds the shortest tour of the union graph exactly,
optionally after adding extra edges, by branch and cut with SCIP (through PySCIPOpt):

  * Reduction. Two rules hold for every tour of the graph and are applied until nothing changes: a node with two
    remaining edges uses both (the edges are fixed), and a node with two fixed edges uses no other edge (its other
    edges are removed). An edge that would close a path of fixed edges into a cycle of fewer than n nodes is
    removed as well.
  * Branch and cut. One binary variable per remaining free edge, one degree equation per node with fewer than two
    fixed edges, and subtour elimination constraints added by a constraint handler: whenever a solution of the LP
    relaxation (at the thresholds 1e-6, 0.3, 0.5 and 0.7) or an integral solution falls apart into several
    connected components, the handler adds x(delta(S)) >= 2 for every component S that violates it. The fixed paths
    are contracted, so a component is a set of paths.
  * Start. The shortest given tour is passed to SCIP as its first solution, so the result is never longer.

Extra edges are taken from the graph of the BetaTour model and must be absent from the union: either the edges with
the highest predicted probability or, as a control with the same number, the shortest ones.

The notebook runs every solve in its own process through the command line of this module (``solve IN OUT``) with a
wall-clock deadline. Pure NumPy apart from PySCIPOpt.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from itertools import permutations

import numpy as np

VERSION = "v27-unionx-1"
THRESHOLDS = (1e-6, 0.3, 0.5, 0.7)


# ------------------------------------------------------------------------------------------------------ edges
def keys_of(a, b, n):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return np.minimum(a, b) * int(n) + np.maximum(a, b)


def tour_keys(tour):
    """Sorted keys of the n undirected edges of a tour (a permutation of 0..n-1)."""
    t = np.asarray(tour, dtype=np.int64)
    return np.unique(keys_of(t, np.roll(t, -1), len(t)))


def union_keys(tours, n):
    return np.unique(np.concatenate([keys_of(np.asarray(t), np.roll(np.asarray(t), -1), n) for t in tours]))


def edge_length(xy, u, v, ew="EUC_2D"):
    """Integer TSPLIB lengths, computed as tsplab.edge_len does."""
    d = np.sqrt(((np.asarray(xy, dtype=np.float64)[u] - np.asarray(xy, dtype=np.float64)[v]) ** 2).sum(-1))
    if ew == "EUC_2D":
        return np.floor(d + 0.5).astype(np.int64)
    if ew == "CEIL_2D":
        return np.ceil(d).astype(np.int64)
    raise ValueError(f"unsupported EDGE_WEIGHT_TYPE {ew}")


def tour_length(xy, tour, ew="EUC_2D"):
    t = np.asarray(tour, dtype=np.int64)
    return int(edge_length(xy, t, np.roll(t, -1), ew).sum())


def is_permutation(tour, n):
    t = np.asarray(tour)
    return t.shape == (n,) and len(np.unique(t)) == n and t.min() == 0 and t.max() == n - 1


# --------------------------------------------------------------------------------------------------- reduction
def reduce_graph(n, keys):
    """Fixed and removed edges that hold for every tour of the graph with the given edge keys.

    Returns status per edge (0 free, 1 fixed, 2 removed), the number of fixed edges per node, the root of the fixed
    path of every node and whether the fixed edges already form a tour. Raises ValueError if the graph has no tour
    by these rules (it cannot happen for the union of tours)."""
    keys = np.asarray(keys, dtype=np.int64)
    eu, ev = keys // n, keys % n
    m = len(keys)
    adj = [[] for _ in range(n)]
    for e in range(m):
        adj[eu[e]].append(e)
        adj[ev[e]].append(e)
    status = np.zeros(m, dtype=np.int8)
    deg = np.array([len(a) for a in adj], dtype=np.int64)
    fdeg = np.zeros(n, dtype=np.int64)
    other = np.arange(n, dtype=np.int64)          # for an end of a fixed path: the other end
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)
    closed = False
    queue = deque(range(n))

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    def remove(e):
        status[e] = 2
        for x in (eu[e], ev[e]):
            deg[x] -= 1
            queue.append(int(x))

    def fix(e):
        nonlocal closed
        a, b = int(eu[e]), int(ev[e])
        if fdeg[a] >= 2 or fdeg[b] >= 2:
            raise ValueError("no tour: a node would get three fixed edges")
        status[e] = 1
        oa, ob = int(other[a]), int(other[b])
        ra, rb = find(a), find(b)
        if ra == rb:
            if size[ra] != n:
                raise ValueError("no tour: the fixed edges close a subtour")
            closed = True
        else:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
            other[oa], other[ob] = ob, oa
            if size[ra] < n:
                for e2 in adj[oa]:
                    if status[e2] == 0 and (eu[e2] == ob or ev[e2] == ob):
                        remove(e2)
        fdeg[a] += 1
        fdeg[b] += 1
        queue.extend((a, b, oa, ob))

    while queue:
        v = queue.popleft()
        if fdeg[v] == 2:
            for e in adj[v]:
                if status[e] == 0:
                    remove(e)
        elif deg[v] == 2:
            for e in adj[v]:
                if status[e] == 0:
                    fix(e)
        if deg[v] < 2:
            raise ValueError("no tour: a node has fewer than two edges")
    roots = np.array([find(v) for v in range(n)], dtype=np.int64)
    return dict(status=status, fdeg=fdeg, roots=roots, closed=bool(closed))


def tour_from_edges(n, keys, chosen):
    """The tour formed by the chosen edges (boolean mask over keys); raises if they do not form one tour."""
    keys = np.asarray(keys, dtype=np.int64)[np.asarray(chosen, dtype=bool)]
    nb = np.full((n, 2), -1, dtype=np.int64)
    count = np.zeros(n, dtype=np.int64)
    for k in keys:
        a, b = int(k // n), int(k % n)
        if count[a] >= 2 or count[b] >= 2:
            raise ValueError("a node has more than two chosen edges")
        nb[a, count[a]] = b
        nb[b, count[b]] = a
        count[a] += 1
        count[b] += 1
    if not np.all(count == 2):
        raise ValueError("the chosen edges are not 2-regular")
    tour = np.empty(n, dtype=np.int64)
    prev, cur = -1, 0
    for i in range(n):
        tour[i] = cur
        nxt = nb[cur, 0] if nb[cur, 0] != prev else nb[cur, 1]
        prev, cur = cur, nxt
    if cur != 0 or len(np.unique(tour)) != n:
        raise ValueError("the chosen edges form more than one cycle")
    return tour


# ------------------------------------------------------------------------------------------------ branch and cut
def _scip():
    from pyscipopt import Conshdlr, Model, SCIP_RESULT, quicksum
    return Conshdlr, Model, SCIP_RESULT, quicksum


def solve_exact(xy, keys, start_tour, time_limit, ew="EUC_2D"):
    """Shortest tour of the graph with the given edge keys, by branch and cut with SCIP within time_limit seconds.

    start_tour must be a tour of the graph. Returns (tour, info); the tour is the best one found (the start tour if
    nothing shorter was found in time) and info["optimal"] says whether SCIP proved it shortest."""
    t_begin = time.perf_counter()
    xy = np.asarray(xy)
    n = len(xy)
    keys = np.asarray(keys, dtype=np.int64)
    assert np.all(np.diff(keys) > 0), "edge keys must be sorted and unique"
    start = np.asarray(start_tour, dtype=np.int64)
    start_keys = tour_keys(start)
    assert is_permutation(start, n) and np.isin(start_keys, keys).all(), "the start tour is not a tour of the graph"
    length = edge_length(xy, keys // n, keys % n, ew)
    red = reduce_graph(n, keys)
    status, fdeg, roots = red["status"], red["fdeg"], red["roots"]
    info = dict(version=VERSION, nodes=int(n), edges=int(len(keys)), fixed=int((status == 1).sum()),
                removed=int((status == 2).sum()), free=int((status == 0).sum()), core=int((fdeg < 2).sum()),
                start_length=int(tour_length(xy, start, ew)), start_has_all_fixed=None,
                start_avoids_removed=None)
    pos_of = {int(k): i for i, k in enumerate(keys)}
    in_start = np.zeros(len(keys), dtype=bool)
    in_start[[pos_of[int(k)] for k in start_keys]] = True
    # Every tour of the graph holds the fixed edges and none of the removed ones; the start tour must too.
    info["start_has_all_fixed"] = bool(np.all(in_start[status == 1]))
    info["start_avoids_removed"] = bool(not np.any(in_start[status == 2]))
    assert info["start_has_all_fixed"] and info["start_avoids_removed"], "reduction contradicts the start tour"
    info["reduce_s"] = round(time.perf_counter() - t_begin, 4)
    free = np.flatnonzero(status == 0)
    if red["closed"] or len(free) == 0:
        tour = tour_from_edges(n, keys, status == 1)
        info.update(status="fixed", optimal=True, solve_s=0.0, bnb_nodes=0, cuts=0, gap=0.0,
                    dual_bound=None, length=int(tour_length(xy, tour, ew)), time_to_best_s=0.0,
                    solve_start_epoch=None)
        return tour, info
    Conshdlr, Model, SCIP_RESULT, quicksum = _scip()
    fu, fv = keys[free] // n, keys[free] % n
    core = np.flatnonzero(fdeg < 2)
    _, sup = np.unique(roots[core], return_inverse=True)
    sup_of = np.full(n, -1, dtype=np.int64)
    sup_of[core] = sup
    ea, eb = sup_of[fu], sup_of[fv]
    assert np.all(ea >= 0) and np.all(eb >= 0)
    nsup = int(sup.max()) + 1

    class Subtour(Conshdlr):
        def components(self, vals, thr):
            parent = np.arange(nsup)

            def find(x):
                r = x
                while parent[r] != r:
                    r = parent[r]
                while parent[x] != r:
                    parent[x], x = r, parent[x]
                return r
            for j in np.flatnonzero(vals > thr):
                ra, rb = find(ea[j]), find(eb[j])
                if ra != rb:
                    parent[ra] = rb
            return np.array([find(i) for i in range(nsup)])

        def values(self, solution):
            return np.array([self.model.getSolVal(solution, v) for v in xvars])

        def connected(self, vals):
            return len(np.unique(self.components(vals, 0.5))) == 1

        def cut(self, vals, thresholds, force):
            if self.tvars is None:
                self.tvars = [self.model.getTransformedVar(v) for v in xvars]
            added = 0
            for thr in thresholds:
                lab = self.components(vals, thr)
                labels = np.unique(lab)
                if len(labels) == 1:
                    continue
                la, lb = lab[ea], lab[eb]
                for label in labels:
                    cross = np.flatnonzero((la == label) != (lb == label))
                    if vals[cross].sum() >= 2.0 - 1e-6:
                        continue
                    row = self.model.createEmptyRowUnspec(name="sec", lhs=2.0, rhs=None, local=False,
                                                          modifiable=False, removable=True)
                    self.model.cacheRowExtensions(row)
                    for j in cross:
                        self.model.addVarToRow(row, self.tvars[j], 1.0)
                    self.model.flushRowExtensions(row)
                    self.model.addCut(row, forcecut=force)
                    self.model.releaseRow(row)
                    added += 1
                if added:
                    break
            self.ncuts += added
            return added

        def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason, completely, **kw):
            ok = self.connected(self.values(solution))
            return {"result": SCIP_RESULT.FEASIBLE if ok else SCIP_RESULT.INFEASIBLE}

        def consenfolp(self, constraints, nusefulconss, solinfeasible):
            vals = self.values(None)
            if self.connected(vals):
                return {"result": SCIP_RESULT.FEASIBLE}
            added = self.cut(vals, (0.5,), True)
            return {"result": SCIP_RESULT.SEPARATED if added else SCIP_RESULT.INFEASIBLE}

        def consenfops(self, constraints, nusefulconss, solinfeasible, objinfeasible):
            vals = self.values(None)
            return {"result": SCIP_RESULT.FEASIBLE if self.connected(vals) else SCIP_RESULT.SOLVELP}

        def conssepalp(self, constraints, nusefulconss):
            added = self.cut(self.values(None), THRESHOLDS, False)
            return {"result": SCIP_RESULT.SEPARATED if added else SCIP_RESULT.DIDNOTFIND}

        def conslock(self, constraint, locktype, nlockspos, nlocksneg):
            for v in xvars:
                self.model.addVarLocks(v, nlockspos + nlocksneg, nlockspos + nlocksneg)

    model = Model()
    model.hideOutput()
    xvars = [model.addVar(vtype="B", obj=float(length[j])) for j in free]
    incident = [[] for _ in range(n)]
    for i in range(len(free)):
        incident[fu[i]].append(xvars[i])
        incident[fv[i]].append(xvars[i])
    for v in core:
        model.addCons(quicksum(incident[v]) == 2 - int(fdeg[v]))
    handler = Subtour()
    handler.tvars, handler.ncuts = None, 0
    model.includeConshdlr(handler, "subtour", "subtour elimination on the contracted union graph", sepapriority=1,
                          enfopriority=-1, chckpriority=-1, sepafreq=1, propfreq=-1, eagerfreq=-1, maxprerounds=0,
                          delaysepa=False, delayprop=False, needscons=True)
    model.addPyCons(model.createCons(handler, "subtour", initial=False, separate=True, enforce=True, check=True,
                                     propagate=False))
    first = model.createSol()
    for i, j in enumerate(free):
        model.setSolVal(first, xvars[i], 1.0 if in_start[j] else 0.0)
    accepted = model.addSol(first, free=True)
    info["build_s"] = round(time.perf_counter() - t_begin - info["reduce_s"], 4)
    left = float(time_limit) - (time.perf_counter() - t_begin)
    if left <= 0.05:
        info.update(status="no time", optimal=False, solve_s=0.0, bnb_nodes=0, cuts=0, gap=None, dual_bound=None,
                    length=info["start_length"], time_to_best_s=None, start_accepted=bool(accepted),
                    solve_start_epoch=None)
        return start.copy(), info
    model.setParam("limits/time", left)
    info["solve_start_epoch"] = time.time()
    t_solve = time.perf_counter()
    model.optimize()
    solve_s = time.perf_counter() - t_solve
    best = model.getBestSol()
    chosen = (status == 1).copy()
    values = np.array([model.getSolVal(best, v) for v in xvars])
    chosen[free[values > 0.5]] = True
    tour = tour_from_edges(n, keys, chosen)
    scip_status = model.getStatus()
    forced_length = int(length[status == 1].sum())
    info.update(status=str(scip_status), optimal=bool(scip_status == "optimal"), solve_s=round(solve_s, 4),
                bnb_nodes=int(model.getNNodes()), cuts=int(handler.ncuts), gap=float(model.getGap()),
                dual_bound=float(model.getDualbound()) + forced_length, length=int(tour_length(xy, tour, ew)),
                objective=float(model.getObjVal()) + forced_length, time_to_best_s=float(model.getSolTime(best)),
                start_accepted=bool(accepted), scip_version=str(model.version()))
    assert info["length"] == round(info["objective"]), "the tour length differs from the SCIP objective"
    assert info["length"] <= info["start_length"], "the result is longer than the start tour"
    return tour, info


# ------------------------------------------------------------------------------------------------- extra edges
def probability_table(prob, nb):
    """Undirected edge keys of the model graph and their probability (maximum over the two directions); as
    marking.probability_table of run v23. prob[i, j] belongs to the edge from i to its neighbour nb[i, j]."""
    nb = np.asarray(nb, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    assert nb.ndim == 2 and prob.shape == nb.shape and np.all(nb >= 0)
    n, k = nb.shape
    keys = keys_of(np.repeat(np.arange(n, dtype=np.int64), k), nb.reshape(-1), n)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    unique, start = np.unique(keys, return_index=True)
    return unique, np.maximum.reduceat(prob.reshape(-1)[order], start)


def extra_edges(table, union, count, by, xy=None, ew="EUC_2D"):
    """`count` edges of the model graph that are absent from the union: by="probability" takes the most probable
    ones, by="length" the shortest ones; ties are broken by the edge key. Returns sorted keys."""
    keys, prob = table
    n = None if xy is None else len(xy)
    novel = ~np.isin(keys, union)
    cand, p = keys[novel], prob[novel]
    if by == "probability":
        order = np.lexsort((cand, -p))
    elif by == "length":
        assert xy is not None
        order = np.lexsort((cand, edge_length(xy, cand // n, cand % n, ew)))
    else:
        raise ValueError(by)
    return np.sort(cand[order[:int(count)]])


# ------------------------------------------------------------------------------------------------- small checks
def brute_force(xy, keys, ew="EUC_2D"):
    """Shortest tour of a small graph by enumeration (node 0 first, one direction); None if there is none."""
    n = len(xy)
    allowed = set(np.asarray(keys, dtype=np.int64).tolist())
    best = None
    for rest in permutations(range(1, n)):
        if rest[0] > rest[-1]:
            continue
        t = (0,) + rest
        if all(min(t[i], t[(i + 1) % n]) * n + max(t[i], t[(i + 1) % n]) in allowed for i in range(n)):
            length = tour_length(xy, t, ew)
            if best is None or length < best[0]:
                best = (length, np.array(t, dtype=np.int64))
    return best


# ------------------------------------------------------------------------------------------------- command line
def main(argv):
    """python unionx_v27.py solve IN.npz OUT.npz

    IN holds xy, keys, start (tour), deadline (epoch seconds) and ew; OUT gets the tour and the info as JSON."""
    assert len(argv) == 3 and argv[0] == "solve", main.__doc__
    with np.load(argv[1], allow_pickle=False) as z:
        xy, keys, start = z["xy"], z["keys"], z["start"]
        deadline, ew = float(z["deadline"]), str(z["ew"])
    tour, info = solve_exact(xy, keys, start, deadline - time.time(), ew)
    info["finished_epoch"] = time.time()
    tmp = argv[2] + ".tmp.npz"
    np.savez_compressed(tmp, tour=tour.astype(np.int32), info=np.array(json.dumps(info)))
    import os
    os.replace(tmp, argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
