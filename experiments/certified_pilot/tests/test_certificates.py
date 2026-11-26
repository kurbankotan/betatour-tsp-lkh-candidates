"""Unit tests of certpilot.py against brute force (small instances).

    python tests/test_certificates.py --src REPO/src

1. C1: the 1-tree bound W equals tsplab.held_karp_bound; every alpha-value equals the cost of the minimum 1-tree
   that contains the edge minus the minimum 1-tree, found by brute force; every tour that uses an edge costs at least
   W + alpha (all tours enumerated).
2. C2: reduced costs of a subtour LP (all subtour constraints, solved with HiGHS): every tour that uses an edge costs
   at least LB + rc, c2_checks passes, reduced costs without the cut duals fail the dual-consistency check, and a
   dump file in Concorde's format is read back with per-node thresholds that bound the unlisted reduced costs.
3. certified_k and edge_coverage on a hand-made example.
"""
import argparse
import itertools
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import certpilot as C  # noqa: E402


def all_tours(n):
    for perm in itertools.permutations(range(1, n)):
        if perm[0] < perm[-1]:
            yield (0,) + perm


def tour_cost(D, t):
    t = np.asarray(t)
    return int(D[t, np.roll(t, -1)].sum())


def mst_with_forced(Cm, nodes, forced=None):
    """Kruskal on the given nodes; a forced edge is taken first."""
    nodes = list(nodes)
    parent = {v: v for v in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = sorted((int(Cm[i, j]), i, j) for i, j in itertools.combinations(nodes, 2))
    total, used = 0, 0
    if forced is not None:
        i, j = forced
        parent[find(i)] = find(j)
        total += int(Cm[i, j])
        used = 1
    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            total += w
            used += 1
    assert used == len(nodes) - 1
    return total


def test_c1(T, rng, n=8, trials=9):
    for trial in range(trials):
        xy = rng.integers(0, 1000, size=(n, 2))
        D = C.distance_matrix(xy)
        # zero, small and large penalties; with large ones many transformed costs are negative
        scale = [0, 300, 300, 300, 200000, 200000, 200000, 5000000, 5000000][trial]
        pi = rng.integers(-scale, scale + 1, size=n)
        assert trial < 4 or (D * 100 + pi[:, None] + pi[None, :]).min() < 0 or scale < 200000
        tree = C.one_tree(D, pi)
        hk = T.held_karp_bound(xy, pi=pi, precision=100.0)["bound"] * 100.0
        assert abs(hk - tree["W"]) < 1e-6, (hk, tree["W"])
        Cm = D * 100 + pi[:, None] + pi[None, :]
        s = tree["special"]
        others = [v for v in range(n) if v != s]
        base = mst_with_forced(Cm, others)
        row = sorted(int(Cm[s, j]) for j in others)
        L1 = base + row[0] + row[1]
        assert L1 == tree["L1"], (L1, tree["L1"])
        for i, j in itertools.combinations(range(n), 2):
            if s in (i, j):
                k = j if i == s else i
                rest = sorted(int(Cm[s, m]) for m in others if m != k)
                with_e = base + int(Cm[s, k]) + rest[0]
            else:
                with_e = mst_with_forced(Cm, others, forced=(i, j)) + row[0] + row[1]
            assert with_e - L1 == tree["alpha"][i, j], (i, j, with_e - L1, tree["alpha"][i, j])
        best = {}
        for t in all_tours(n):
            cost = tour_cost(D, t) * 100
            for a, b in zip(t, t[1:] + t[:1]):
                key = (min(a, b), max(a, b))
                best[key] = min(best.get(key, 1 << 62), cost)
        for (i, j), cost in best.items():
            assert cost >= tree["W"] + tree["alpha"][i, j], (i, j)
        opt = min(tour_cost(D, t) for t in all_tours(n))
        remaining = C.c1_remaining(tree, opt)
        for t in all_tours(n):
            if tour_cost(D, t) == opt:
                assert np.isin(C.tour_keys(t, n), remaining).all(), "C1 eliminated an optimal edge"
    print("C1: 1-tree bound, alpha-values and the elimination rule agree with brute force")


def subtour_lp(D):
    """The subtour LP over the complete graph with every subtour constraint (n small): value, reduced costs (with the
    cut duals), node duals and reduced costs without the cut duals."""
    from scipy.optimize import linprog
    n = len(D)
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    c = D[iu, ju].astype(np.float64)
    A_eq = np.zeros((n, m))
    A_eq[iu, np.arange(m)] = 1.0
    A_eq[ju, np.arange(m)] = 1.0
    rows = []
    for size in range(2, n // 2 + 1):
        for S in itertools.combinations(range(n), size):
            inside = np.zeros(n, dtype=bool)
            inside[list(S)] = True
            rows.append(-(inside[iu] != inside[ju]).astype(np.float64))
    A_ub = np.array(rows)
    res = linprog(c, A_ub=A_ub, b_ub=-2.0 * np.ones(len(rows)), A_eq=A_eq, b_eq=2.0 * np.ones(n),
                  bounds=(0.0, 1.0), method="highs")
    assert res.status == 0
    y_eq, y_ub = res.eqlin.marginals, res.ineqlin.marginals
    rc = c - A_eq.T @ y_eq - A_ub.T @ y_ub
    rc_nodes_only = c - A_eq.T @ y_eq
    return float(res.fun), iu * n + ju, rc, rc_nodes_only


def test_c2(rng, n=9):
    with_cuts, detected = 0, 0
    for _ in range(8):
        xy = rng.integers(0, 1000, size=(n, 2))
        D = C.distance_matrix(xy)
        lb, keys, rc, rc_nodes = subtour_lp(D)
        tours = list(all_tours(n))
        costs = np.array([tour_cost(D, t) for t in tours])
        opt = int(costs.min())
        opt_tour = tours[int(costs.argmin())]
        for t, cost in zip(tours, costs):
            pos, found = C.lookup(keys, C.tour_keys(t, n))
            assert found.all()
            assert (cost >= lb + rc[pos] - 1e-6).all(), "a tour is shorter than LB + rc of one of its edges"
        check = C.c2_checks(keys, rc, lb, n, opt, opt_tour, tol=1e-6)
        assert check["passed"], check
        if np.abs(rc - rc_nodes).max() > 1e-6:          # some subtour cut has a positive dual
            with_cuts += 1
            bad = C.c2_checks(keys, rc_nodes, lb, n, opt, opt_tour, tol=1e-6)
            detected += int(not bad["passed"])
        # A per-node dump of the k lowest reduced costs, in Concorde's file format, read back.
        k = 3
        a, b = keys // n, keys % n
        lines = []
        for v in range(n):
            at = np.flatnonzero((a == v) | (b == v))
            for e in at[np.argsort(rc[at], kind="stable")[:k]]:
                lines.append(f"{v} {b[e] if a[e] == v else a[e]} {rc[e]:.6f}")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.rcn"
            path.write_text(f"{n} {len(lines)}\n" + "\n".join(lines) + "\n")
            dump = C.read_rcn(path, n)
        thr = C.node_thresholds(dump["keys"], dump["rc"], n, k)
        listed = np.isin(keys, dump["keys"])
        ua, ub_ = keys[~listed] // n, keys[~listed] % n
        assert (rc[~listed] >= np.maximum(thr[ua], thr[ub_]) - 1e-6).all(), "threshold above an unlisted rc"
        rem, unresolved = C.c2_remaining(dump["keys"], dump["rc"], lb, opt, n, 1e-6, thr=thr)
        full_rem, _ = C.c2_remaining(keys, rc, lb, opt, n, 1e-6)
        assert np.isin(rem, full_rem).all()
        resolved = ~unresolved
        missing = full_rem[~np.isin(full_rem, rem)]
        assert not (resolved[missing // n] & resolved[missing % n]).any(), "a remaining edge was lost"
        check2 = C.c2_checks(dump["keys"], dump["rc"], lb, n, opt, opt_tour, tol=1e-5, thr=thr,
                             sum_negative=float(np.clip(-dump["rc"], 0, None).sum()))
        assert check2["lb_le_opt"] and check2["opt_edges_eliminated"] == 0, check2
    assert with_cuts > 0 and detected > 0, (with_cuts, detected)
    print(f"C2: subtour-LP reduced costs bound every tour; checks pass; reduced costs without the cut duals failed "
          f"the checks on {detected} of {with_cuts} instances with active cuts; per-node dump and thresholds agree")


def test_lists():
    n = 5
    remaining = np.array([0 * n + 1, 0 * n + 2, 1 * n + 2, 2 * n + 3, 3 * n + 4, 1 * n + 4], dtype=np.int64)
    ranking = np.array([[1, 2, 3, 4], [0, 2, 4, 3], [3, 0, 1, 4], [2, 4, 0, 1], [3, 1, 0, 2]])
    k = C.certified_k(ranking, np.sort(remaining), n)
    assert k.tolist() == [2, 3, 3, 2, 2], k
    k2 = C.certified_k(ranking[:, :2], np.sort(remaining), n)
    assert k2.tolist() == [2, 3, 3, 2, 2], k2
    cov = C.edge_coverage(ranking[:, :1], np.sort(remaining), n)
    assert abs(cov - 3 / 6) < 1e-12, cov
    print("certified_k and edge_coverage: hand-made example passes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the repository's src folder (tsplab.py)")
    args = ap.parse_args()
    sys.path.insert(0, args.src)
    import tsplab as T
    rng = np.random.default_rng(2026)
    test_c1(T, rng)
    test_c2(rng)
    test_lists()
    print("all tests passed")


if __name__ == "__main__":
    main()
