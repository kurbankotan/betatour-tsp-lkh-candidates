"""Tests of unionx.py (run v27). Pure NumPy and PySCIPOpt; no LKH-3 or GA-EAX needed.

    python experiments/v27_union_exact/tests/test_unionx.py

1. The reduction never contradicts a tour of the graph: on small random graphs (unions of random tours plus random
   extra edges) every tour found by enumeration holds all fixed edges and no removed edge.
2. The branch and cut finds the shortest tour of the graph: same length as enumeration on small random graphs, with
   status optimal, a valid tour and only edges of the graph.
3. On a union of six 2-opt tours of 400 nodes the result is a valid tour of the union, not longer than the shortest
   input tour, proven optimal, and the fixed edges are in every input tour.
4. A time limit of zero returns the start tour; the command line writes the same result as the function.
5. Extra edges are absent from the union, come from the model graph, have the requested number and order.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import unionx as U  # noqa: E402


def two_opt(xy, tour):
    """Plain first-improvement 2-opt until no improving move (small n only)."""
    t = list(tour)
    n = len(t)
    d = lambda a, b: int(U.edge_length(xy, np.array([a]), np.array([b]))[0])
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n if i else n - 1):
                a, b, c, e = t[i], t[i + 1], t[j], t[(j + 1) % n]
                if d(a, c) + d(b, e) < d(a, b) + d(c, e):
                    t[i + 1:j + 1] = reversed(t[i + 1:j + 1])
                    improved = True
    return np.array(t, dtype=np.int64)


def all_tours(n, keys):
    allowed = set(keys.tolist())
    from itertools import permutations
    out = []
    for rest in permutations(range(1, n)):
        if rest[0] > rest[-1]:
            continue
        t = (0,) + rest
        if all(min(t[i], t[(i + 1) % n]) * n + max(t[i], t[(i + 1) % n]) in allowed for i in range(n)):
            out.append(np.array(t))
    return out


def test_reduction_and_exact_small():
    rng = np.random.default_rng(27)
    checked = 0
    for trial in range(60):
        n = int(rng.integers(6, 10))
        xy = rng.integers(0, 1000, size=(n, 2))
        tours = [rng.permutation(n) for _ in range(int(rng.integers(1, 4)))]
        keys = U.union_keys(tours, n)
        extra = rng.integers(0, n, size=(int(rng.integers(0, 4)), 2))
        extra = extra[extra[:, 0] != extra[:, 1]]
        keys = np.unique(np.concatenate([keys, U.keys_of(extra[:, 0], extra[:, 1], n)])) if len(extra) else keys
        red = U.reduce_graph(n, keys)
        tours_of_graph = all_tours(n, keys)
        assert tours_of_graph, "a union of tours always has a tour"
        for t in tours_of_graph:
            pos = np.searchsorted(keys, U.tour_keys(t))
            mask = np.zeros(len(keys), dtype=bool)
            mask[pos] = True
            assert np.all(mask[red["status"] == 1]), (trial, "a tour misses a fixed edge")
            assert not np.any(mask[red["status"] == 2]), (trial, "a tour uses a removed edge")
        start = min(tours, key=lambda t: U.tour_length(xy, t))
        tour, info = U.solve_exact(xy, keys, start, 30.0)
        best = U.brute_force(xy, keys)
        assert U.is_permutation(tour, n) and np.isin(U.tour_keys(tour), keys).all()
        assert info["optimal"] and info["length"] == best[0] == U.tour_length(xy, tour), (trial, info, best[0])
        checked += 1
    print("small graphs: reduction and exact solve agree with enumeration on", checked, "graphs")


def test_union_of_two_opt_tours():
    rng = np.random.default_rng(2701)
    n = 400
    xy = rng.integers(0, 100000, size=(n, 2))
    tours = [two_opt(xy, rng.permutation(n)) for _ in range(6)]
    lengths = [U.tour_length(xy, t) for t in tours]
    keys = U.union_keys(tours, n)
    red = U.reduce_graph(n, keys)
    for t in tours:
        mask = np.isin(keys, U.tour_keys(t))
        assert np.all(mask[red["status"] == 1]) and not np.any(mask[red["status"] == 2])
    start = tours[int(np.argmin(lengths))]
    t0 = time.perf_counter()
    tour, info = U.solve_exact(xy, keys, start, 120.0)
    assert U.is_permutation(tour, n) and np.isin(U.tour_keys(tour), keys).all()
    assert info["optimal"] and info["length"] <= min(lengths) and info["length"] == U.tour_length(xy, tour)
    print(f"union of six 2-opt tours (n={n}, {len(keys)} edges, core {info['core']}): {min(lengths)} -> "
          f"{info['length']} in {time.perf_counter() - t0:.1f} s, {info['bnb_nodes']} nodes, {info['cuts']} cuts")
    return xy, keys, start, info


def test_time_limit_and_cli(xy, keys, start, info):
    tour, quick = U.solve_exact(xy, keys, start, 0.0)
    assert quick["status"] == "no time" and np.array_equal(tour, start)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        np.savez(src, xy=xy, keys=keys, start=start, deadline=np.float64(time.time() + 120.0), ew=np.array("EUC_2D"))
        done = subprocess.run([sys.executable, str(HERE.parent / "unionx.py"), "solve", str(src), str(dst)],
                              capture_output=True, text=True, timeout=300)
        assert done.returncode == 0, done.stderr[-2000:]
        with np.load(dst, allow_pickle=False) as z:
            tour_cli, info_cli = z["tour"], json.loads(str(z["info"]))
    assert info_cli["optimal"] and info_cli["length"] == info["length"] == U.tour_length(xy, tour_cli)
    print("time limit 0 returns the start tour; the command line gives the same length", info_cli["length"])


def test_extra_edges():
    rng = np.random.default_rng(5)
    n, k = 50, 8
    xy = rng.integers(0, 1000, size=(n, 2))
    d = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    nb = np.argsort(d, axis=1)[:, 1:k + 1]
    prob = rng.random((n, k))
    table = U.probability_table(prob, nb)
    # maximum over the two directions
    keys, p = table
    for key, value in zip(keys[:20], p[:20]):
        a, b = int(key // n), int(key % n)
        both = [prob[a][list(nb[a]).index(b)] if b in nb[a] else -1, prob[b][list(nb[b]).index(a)] if a in nb[b] else -1]
        assert np.isclose(value, max(both))
    union = U.union_keys([rng.permutation(n) for _ in range(2)], n)
    top = U.extra_edges(table, union, 10, "probability")
    near = U.extra_edges(table, union, 10, "length", xy=xy)
    assert len(top) == len(near) == 10 and not np.isin(top, union).any() and not np.isin(near, union).any()
    assert np.isin(top, keys).all() and np.isin(near, keys).all()
    novel = ~np.isin(keys, union)
    assert p[np.isin(keys, top)].min() >= np.sort(p[novel])[::-1][9] - 1e-12
    lengths = U.edge_length(xy, keys // n, keys % n)
    assert lengths[np.isin(keys, near)].max() <= np.sort(lengths[novel])[9]
    print("extra edges: novel, from the model graph, requested number and order")


if __name__ == "__main__":
    test_reduction_and_exact_small()
    args = test_union_of_two_opt_tours()
    test_time_limit_and_cli(*args)
    test_extra_edges()
    print("all tests passed")
