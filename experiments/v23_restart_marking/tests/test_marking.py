"""Tests of the run-v23 module: edge keys, ensemble counts, edge probabilities and the alpha-values of a marking;
with --lkh also the markings on the candidate file of a real LKH-3 run and on the output of the model.

    python tests/test_marking.py --src REPO/src [--lkh /path/to/LKH --checkpoint REPO/checkpoints/betatour_v2.pt]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "v21_alpha_encoding"))


def brute_count(cand, tours):
    n = len(cand)
    sets = [{(min(a, b), max(a, b)) for a, b in zip(t.tolist(), np.roll(t, -1).tolist())} for t in tours]
    return np.array([[0 if j < 0 else sum((min(i, j), max(i, j)) in s for s in sets) for j in row]
                     for i, row in enumerate(cand.tolist())]).reshape(n, -1)


def brute_probability(cand, prob, nb):
    best = {}
    for i, (row, values) in enumerate(zip(nb.tolist(), prob.tolist())):
        for j, p in zip(row, values):
            key = (min(i, j), max(i, j))
            best[key] = max(best.get(key, 0.0), p)
    return np.array([[0.0 if j < 0 else best.get((min(i, j), max(i, j)), 0.0) for j in row]
                     for i, row in enumerate(cand.tolist())])


def unit_tests(A, K):
    rng = np.random.default_rng(5)
    a, b = rng.integers(0, 50, 200), rng.integers(0, 50, 200)
    assert np.array_equal(K.edge_keys(a, b, 50), A.edge_keys(a, b, 50))

    # four nodes; the tours 0-1-2-3 and 0-2-1-3 share the edges 1-2 and 0-3
    cand = np.array([[1, 2, 3], [0, 2, -1], [1, 3, 0], [2, 0, -1]])
    assert K.entry_keys(cand).tolist() == [[1, 2, 3], [1, 6, -1], [6, 11, 2], [11, 3, -1]]
    t1, t2 = np.array([0, 1, 2, 3]), np.array([0, 2, 1, 3])
    assert K.tour_edge_keys(t1).tolist() == [1, 3, 6, 11] and K.tour_edge_keys(t2).tolist() == [2, 3, 6, 7]
    count = K.ensemble_count(cand, [t1, t2])
    assert count.tolist() == [[1, 1, 2], [1, 2, 0], [2, 1, 1], [1, 2, 0]]
    assert np.array_equal(count, brute_count(cand, [t1, t2]))
    assert K.ensemble_count(cand, []).sum() == 0
    assert K.key_count([1, 3, 6, 11, 7, -1], [t1, t2]).tolist() == [1, 2, 2, 1, 1, 0]

    # directed probabilities: the edge 0-1 is held in both directions (0.2 and 0.9), 2-3 in one (0.7)
    nb = np.array([[1, 2], [0, 2], [3, 1], [0, 1]])
    prob = np.array([[0.2, 0.4], [0.9, 0.1], [0.7, 0.3], [0.6, 0.5]])
    keys, values = K.probability_table(prob, nb)
    assert keys.tolist() == [1, 2, 3, 6, 7, 11] and values.tolist() == [0.9, 0.4, 0.6, 0.3, 0.5, 0.7]
    p = K.entry_probability(cand, (keys, values))
    assert p.tolist() == [[0.9, 0.4, 0.6], [0.9, 0.3, 0.0], [0.3, 0.7, 0.4], [0.7, 0.6, 0.0]]
    assert np.array_equal(p, brute_probability(cand, prob, nb))
    # with the first graph column only, the graph is {0-1, 0-3, 2-3}: the edges 0-2 and 1-3 get probability 0
    small = K.probability_table(prob[:, :1], nb[:, :1])
    assert small[0].tolist() == [1, 3, 11] and small[1].tolist() == [0.9, 0.6, 0.7]
    assert K.entry_probability(np.array([[2, 1], [3, 0], [0, 3], [1, 2]]), small).tolist() \
        == [[0.0, 0.9], [0.0, 0.9], [0.0, 0.7], [0.0, 0.7]]

    assert K.key_probability([11, 5, -1, 1], (keys, values)).tolist() == [0.7, 0.0, 0.0, 0.9]

    # threshold that trusts a given number of the edges of a tour
    tour_p = np.array([0.9, 0.2, 0.7, 0.7, 0.0, 0.5])
    assert [K.matched_threshold(tour_p, k) for k in range(7)] == [float("inf"), 0.9, 0.7, 0.7, 0.5, 0.2, 0.0]
    assert [int(K.trusted_by_probability(tour_p, K.matched_threshold(tour_p, k)).sum()) for k in range(7)] \
        == [0, 1, 3, 3, 4, 5, 5]                     # equal probabilities together; outside the graph never
    assert K.trusted_by_probability(np.array([[0.0, 0.3], [0.31, 1.0]]), 0.3).tolist() == [[False, True], [True, True]]

    # alpha-values of a marking
    trusted = np.array([[True, False, True], [False, True, True], [False, False, False], [True, True, True]])
    assert K.mark_alpha(cand, trusted).tolist() == [[0, 200, 0], [100, 0, 300], [100, 200, 300], [0, 0, 300]]
    native = np.array([[0, 5, 9], [0, 0, -1], [0, 0, 7], [0, 9, -1]])
    assert K.mark_alpha(cand, trusted, rest=native).tolist() == [[0, 5, 0], [1, 0, 1], [1, 1, 7], [0, 0, 1]]
    zero = (cand >= 0) & (native == 0)
    same = K.mark_alpha(cand, zero, rest=native)                # LKH's own marks give back LKH's own alpha-values
    assert np.array_equal(same[cand >= 0], native[cand >= 0])
    tree_keys = A.tree_edge_keys(cand, native)
    tree, is_tree = A.tree_alpha(cand, tree_keys)                # the tree encoding of run v21 is one marking
    assert np.array_equal(K.mark_alpha(cand, is_tree), tree)

    # random lists, tours and graphs against the direct computation
    for n, k in ((30, 4), (400, 6)):
        cand = np.stack([rng.choice(np.delete(np.arange(n), i), size=k, replace=False) for i in range(n)])
        cand[rng.random(cand.shape) < 0.1] = -1
        tours = [rng.permutation(n) for _ in range(5)]
        count = K.ensemble_count(cand, tours)
        assert np.array_equal(count, brute_count(cand, tours)) and count.max() <= 5 and np.all(count[cand < 0] == 0)
        nb = np.stack([rng.choice(np.delete(np.arange(n), i), size=7, replace=False) for i in range(n)])
        prob = rng.random(nb.shape).astype(np.float32)
        p = K.entry_probability(cand, K.probability_table(prob, nb))
        assert np.allclose(p, brute_probability(cand, prob, nb), rtol=0, atol=0) and np.all(p[cand < 0] == 0)
        own = K.edge_keys(tours[0], np.roll(tours[0], -1), n)
        assert np.array_equal(K.key_count(own, tours), [sum(k in set(K.tour_edge_keys(t).tolist()) for t in tours)
                                                         for k in own.tolist()]) and K.key_count(own, tours).min() >= 1
        own_p = K.key_probability(own, K.probability_table(prob, nb))
        for keep in (0, 1, n // 3, n):
            tau = K.matched_threshold(own_p, keep)
            got = int(K.trusted_by_probability(own_p, tau).sum())
            assert min(keep, int((own_p > 0).sum())) <= got <= keep + 2
        for threshold in (1, 3, 5):
            alpha = K.mark_alpha(cand, count >= threshold)
            assert np.array_equal(alpha == 0, (count >= threshold) & (cand >= 0)) and alpha.dtype == np.int64
            kept = A.kept_share(cand, alpha, tours[0])           # walking direction mean, as the notebook reports it
            succ = np.empty(n, dtype=int)
            succ[tours[0]] = np.roll(tours[0], -1)
            fwd = np.mean([(cand[i] == succ[i])[alpha[i] == 0].any() for i in range(n)])
            bwd = np.mean([(cand[succ[i]] == i)[alpha[succ[i]] == 0].any() for i in range(n)])
            assert abs(kept - (fwd + bwd) / 2) < 1e-12
    print("unit tests passed")


def lkh_tests(T, A, K, lkh, checkpoint):
    import torch
    import tsp_model as M
    torch.set_num_threads(1)
    model, _ = M.load_model(checkpoint, torch.device("cpu"))
    with tempfile.TemporaryDirectory() as tmp:
        for dist, n, gen in (("uniform", 1500, 51), ("clustered", 1500, 52)):
            xy = T.to_int_coords(T.gen_instance(n, dist, np.random.default_rng(gen)))
            firsts = [A.task_first_trial(lkh, tmp, seed, xy, k=5, timeout=120) for seed in range(101, 107)]
            tours = np.stack([f["tour"] for f in firsts])
            first = firsts[0]
            cand, alpha, dad = first["cand"], first["alpha"], first["dad"]
            present = cand >= 0
            zero = present & (alpha == 0)
            fixed = {"MAX_TRIALS": 12}

            # LKH's own marks written by mark_alpha: the same file, hence the same search
            own = A.task_guided(lkh, tmp, 600, 101, xy, cand, alpha=alpha, dad=dad, init_tour=first["tour"],
                                extra_params=fixed)
            again = K.mark_alpha(cand, zero, rest=alpha)
            assert np.array_equal(again[present], alpha[present])
            same = A.task_guided(lkh, tmp, 600, 101, xy, cand, alpha=again, dad=dad, init_tour=first["tour"],
                                 extra_params=fixed)
            assert [c for _, c, _ in same["events"]] == [c for _, c, _ in own["events"]]
            assert same["cost_check"] == own["cost_check"]

            # count marking: every tour edge in the list is kept when one tour is enough, fewer with more tours
            count = K.ensemble_count(cand, tours)
            assert np.array_equal(count, brute_count(cand, tours))
            in_list = A.kept_share(cand, np.zeros_like(alpha), first["tour"])
            kept = [A.kept_share(cand, K.mark_alpha(cand, count >= c, rest=alpha), first["tour"]) for c in (1, 4, 5, 6)]
            assert abs(kept[0] - in_list) < 1e-12 and kept[0] > kept[1] > kept[2] > kept[3] > 0.3 and in_list > 0.95

            # model marking on LKH's lists and on the mixed lists
            succ = np.stack([T.tour_succ_pred(t)[0] for t in tours], axis=1).astype(np.int32)
            pred = M.predict_tourfold(model, xy, 16, torch.device("cpu"), ens_succ=succ,
                                      ens_cost=np.array([f["cost"] for f in firsts]))
            table = K.probability_table(pred["prob"], pred["NB"])
            assert len(table[0]) == len(np.unique(table[0])) and np.all((table[1] >= 0) & (table[1] <= 1))
            p = K.entry_probability(cand, table)
            assert np.array_equal(p, brute_probability(cand, np.asarray(pred["prob"], dtype=np.float64), pred["NB"]))
            beta = T.mix_candidates(pred["rank"], cand, k_first=3, k=5, fill=pred["rank"][:, 3:]).astype(np.int64)
            p_beta = K.entry_probability(beta, table)
            assert np.all(p_beta >= 0) and (p_beta[:, 0] > 0).mean() > 0.99      # the model's first choice is in its graph
            # the threshold of the model marking: as many edges of the first tour as "in at least 5 of 6 tours"
            own = K.edge_keys(first["tour"], np.roll(first["tour"], -1), n)
            own_count, own_p = K.key_count(own, tours), K.key_probability(own, table)
            keep = int((own_count >= 5).sum())
            tau = K.matched_threshold(own_p, keep)
            by_model = K.trusted_by_probability(own_p, tau)
            assert (own_p > 0).mean() > 0.999 and 0.5 < tau < 1.0 and keep <= int(by_model.sum()) <= keep + 3
            shares = {"count_beta": A.kept_share(beta, K.mark_alpha(beta, K.ensemble_count(beta, tours) >= 5),
                                                 first["tour"])}
            for name, lists, prob, rest in (("pop", cand, p, alpha), ("beta", beta, p_beta, None)):
                marked = K.mark_alpha(lists, K.trusted_by_probability(prob, tau), rest=rest)
                shares[name] = A.kept_share(lists, marked, first["tour"])
                assert 0.5 < shares[name] <= keep / n + 3 / n
                run = A.task_guided(lkh, tmp, 600, 101, xy, lists, alpha=marked, dad=dad, init_tour=first["tour"],
                                    extra_params=fixed)
                assert run["returncode"] == 0 and len(run["trial_times"]) == 12
                assert run["cost_check"] is not None and run["cost_check"] <= first["cost"]
            native = A.kept_share(cand, alpha, first["tour"])
            assert abs(shares["beta"] - shares["count_beta"]) < 0.01          # equal exposure on the mixed lists
            print(dist, n, "kept share: LKH's marks", round(native, 3), "| model marks on LKH's lists",
                  round(shares["pop"], 3), "| on the mixed lists", round(shares["beta"], 3),
                  "| count marks on the mixed lists", round(shares["count_beta"], 3), "| threshold", round(tau, 4),
                  "| first-tour edges trusted by both", round(float((by_model & (own_count >= 5)).mean()), 3),
                  "| count marks on LKH's lists (1, 4, 5, 6 of 6 tours)", [round(v, 3) for v in kept])
    print("LKH tests passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of tsplab.py and tsp_model.py")
    ap.add_argument("--lkh", default=None, help="LKH-3.0.14 binary; without it only the unit tests run")
    ap.add_argument("--checkpoint", default=None, help="betatour_v2.pt (needed with --lkh)")
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import tsplab as T
    import alphaenc as A
    import marking as K
    unit_tests(A, K)
    if a.lkh:
        assert a.checkpoint, "--checkpoint is needed with --lkh"
        lkh_tests(T, A, K, a.lkh, a.checkpoint)
