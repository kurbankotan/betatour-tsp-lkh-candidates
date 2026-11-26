#!/usr/bin/env python3
"""A stand-in for Concorde's command line, for local tests of the notebook only (n <= 150). It is not Concorde.

It solves the subtour LP over the complete graph (HiGHS; subtour cuts from the connected components and from a
minimum cut of the support graph), prints the summary lines that Concorde prints ('LP Value', 'Exact lower bound',
'Final lower bound', 'Optimal Solution', 'Number of bbnodes', 'Total Running Time'), writes the k lowest reduced
costs of every node to p.rcn for -z k ('n m' header, then 'u v rc'), and, without -B, writes an optimal tour (HiGHS
MILP with subtour cuts) to the -o file in Concorde's format. Options it does not use (-x, -s, -E) are accepted.
"""
import sys
import time

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

USAGE = """Usage: concorde [-see below-] [dat_file]
   -B    do not branch
   -E f  full edge file (must contain initial edge set)
   -o f  output file name (for optimal tour)
   -s #  random seed
   -x    delete files on completion (sav pul mas)
   -z #  dump the #-lowest reduced cost edges to file xxx.rcn
(test stand-in, not Concorde)"""


def read_tsp(path):
    xy, reading = [], False
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        if s.startswith("NODE_COORD_SECTION"):
            reading = True
            continue
        if s.startswith("EOF"):
            break
        if reading:
            _, x, y = s.split()[:3]
            xy.append((float(x), float(y)))
    return np.array(xy)


def distances(xy):
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    return np.floor(d + 0.5).astype(np.int64)


def violated_cuts(n, iu, ju, x):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for e in np.flatnonzero(x > 1e-9):
        g.add_edge(int(iu[e]), int(ju[e]), weight=float(x[e]))
    comps = [sorted(c) for c in nx.connected_components(g)]
    if len(comps) > 1:
        return comps
    value, (a, b) = nx.stoer_wagner(g)
    return [sorted(a if len(a) <= len(b) else b)] if value < 2.0 - 1e-7 else []


def subtour_lp(D):
    n = len(D)
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    c = D[iu, ju].astype(np.float64)
    A_eq = np.zeros((n, m))
    A_eq[iu, np.arange(m)] = 1.0
    A_eq[ju, np.arange(m)] = 1.0
    cuts, values = [], []
    while True:
        A_ub = np.array(cuts) if cuts else None
        b_ub = -2.0 * np.ones(len(cuts)) if cuts else None
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=2.0 * np.ones(n), bounds=(0.0, 1.0), method="highs")
        assert res.status == 0, res.message
        values.append(float(res.fun))
        new = violated_cuts(n, iu, ju, res.x)
        if not new or len(cuts) > 400:
            break
        for S in new:
            inside = np.zeros(n, dtype=bool)
            inside[S] = True
            cuts.append(-(inside[iu] != inside[ju]).astype(np.float64))
    rc = c - A_eq.T @ res.eqlin.marginals
    if cuts:
        rc = rc - np.array(cuts).T @ res.ineqlin.marginals
    return values, iu, ju, rc


def optimal_tour(D):
    n = len(D)
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    c = D[iu, ju].astype(np.float64)
    A_eq = np.zeros((n, m))
    A_eq[iu, np.arange(m)] = 1.0
    A_eq[ju, np.arange(m)] = 1.0
    cons = [LinearConstraint(A_eq, 2.0, 2.0)]
    while True:
        res = milp(c, constraints=cons, integrality=np.ones(m), bounds=Bounds(0, 1))
        assert res.status == 0, res.message
        x = np.round(res.x)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from((int(iu[e]), int(ju[e])) for e in np.flatnonzero(x > 0.5))
        comps = [sorted(s) for s in nx.connected_components(g)]
        if len(comps) == 1:
            break
        for S in comps:
            inside = np.zeros(n, dtype=bool)
            inside[S] = True
            cons.append(LinearConstraint((inside[iu] != inside[ju]).astype(np.float64)[None, :], 2.0, np.inf))
    tour, prev, cur = [0], None, 0
    for _ in range(n - 1):
        nb = list(g.neighbors(cur))
        nxt = nb[0] if nb[0] != prev else nb[1]
        tour.append(nxt)
        prev, cur = cur, nxt
    return np.array(tour), int(round(res.fun))


def main(argv):
    if not argv:
        print(USAGE)
        return 1
    flags, i = {}, 0
    while i < len(argv) - 1:
        a = argv[i]
        if a in ("-B", "-x"):
            flags[a] = True
            i += 1
        else:
            flags[a] = argv[i + 1]
            i += 2
    problem = argv[-1]
    start = time.time()
    D = distances(read_tsp(problem))
    n = len(D)
    values, iu, ju, rc = subtour_lp(D)
    for j, v in enumerate(values, 1):
        print(f"  LP Value {j:2d}: {v:.6f}  (0.00 seconds)")
    lb = values[-1]                 # the LP is over the complete graph, so its value is the exact bound
    print(f"Exact lower bound: {lb:.6f}")
    print("DIFF: 0.000000")
    tour, opt = optimal_tour(D)
    print(f"Final lower bound {lb:.6f}, upper bound {float(opt):.6f}")
    if "-z" in flags:
        k = int(flags["-z"])
        lines = []
        for v in range(n):
            at = np.flatnonzero((iu == v) | (ju == v))
            for e in at[np.argsort(rc[at], kind="stable")[:k]]:
                w = ju[e] if iu[e] == v else iu[e]
                lines.append(f"{v} {w} {rc[e]:.6f}")
        with open(problem.rsplit(".", 1)[0] + ".rcn", "w") as f:
            f.write(f"{n} {len(lines)}\n" + "\n".join(lines) + "\n")
    if "-B" not in flags:
        print(f"Optimal Solution: {float(opt):.2f}")
        print("Number of bbnodes: 1")
        if "-o" in flags:
            with open(flags["-o"], "w") as f:
                f.write(f"{n}\n" + " ".join(map(str, tour.tolist())) + "\n")
    print(f"Total Running Time: {time.time() - start:.2f} (seconds)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
