"""Tests of the run-v21 module: encodings, kept-edge share, trace parsing and statistics; with --lkh also the task
functions and the self-check against a real LKH-3 binary.

    python tests/test_alphaenc.py --src REPO/src [--lkh /path/to/LKH]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

LOG = """PARAMETER_FILE = run.par
Creating POPMUSIC candidate set ...
1: Initial cost:  84665213, Time: 0.04 sec.
1: Improved cost: 79534588, Time: 0.09 sec.
Cost.min = 79367486, Cost.avg = 80155989.58, Cost.max = 80976241
Cand.min = 2, Cand.avg = 7.1, Cand.max = 16
Preprocessing time = 6.47 sec.
Cost = 72079716, Time = 0.67 sec. +
#1000: Time = 0.68 sec.
Cost = 72079058, Time = 0.68 sec.
* 1: Cost = 72079058, Time = 0.71 sec.
  2: Cost = 72080111, Time = 0.90 sec.
Cost = 72078000, Time = 1.02 sec.
* 3: Cost = 72078000, Time = 1.10 sec.
  4: Cost = 72079999, Gap = 0.0100%, Time = 1.31 sec.
Run 1: Cost = 72078000, Time = 1.31 sec.
Cost.min = 72078000, Cost.avg = 72078000.00, Cost.max = 72078000
"""


def unit_tests(T, A):
    # edge keys and the two encodings
    assert A.edge_keys([3, 1], [1, 3], 10).tolist() == [13, 13]
    cand_native = np.array([[1, 2, 3], [0, 2, -1], [1, 3, 0], [2, 0, -1]])
    alpha_native = np.array([[0, 5, 9], [0, 0, -1], [0, 0, 7], [0, 9, -1]])
    keys = A.tree_edge_keys(cand_native, alpha_native)                       # tree edges: 0-1, 1-2, 2-3
    assert keys.tolist() == [1, 6, 11]
    assert A.rank_alpha(cand_native).tolist() == [[0, 100, 200]] * 4
    mixed = np.array([[3, 1, 2], [2, 3, 0], [0, 3, 1], [1, 2, -1]])
    alpha, is_tree = A.tree_alpha(mixed, keys)
    assert alpha.tolist() == [[100, 0, 300], [0, 200, 0], [100, 0, 0], [100, 0, 300]]
    assert is_tree.tolist() == [[False, True, False], [True, False, True], [False, True, True], [False, True, False]]
    same_alpha, same_tree = A.tree_alpha(cand_native, keys)                  # the rule reproduces the native zeros
    assert np.array_equal(same_tree, (cand_native >= 0) & (alpha_native == 0))
    assert np.array_equal(same_alpha == 0, same_tree)

    # share of tour edges that the restart can keep
    tour = np.array([0, 1, 2, 3])
    assert A.kept_share(cand_native, alpha_native, tour) == (3 / 4 + 3 / 4) / 2      # the edge 3-0 has alpha 9
    assert A.kept_share(cand_native, A.rank_alpha(cand_native), tour) == (3 / 4 + 1 / 4) / 2
    assert A.kept_share(cand_native, np.full_like(alpha_native, 100), tour) == 0.0
    ring = np.array([[1, 3], [2, 0], [3, 1], [0, 2]])
    assert A.kept_share(ring, np.zeros_like(ring), tour) == 1.0
    assert A.kept_share(ring, np.zeros_like(ring), tour[::-1].copy()) == 1.0
    rng = np.random.default_rng(3)
    for n in (20, 300):
        perm = rng.permutation(n)
        cand = np.stack([rng.choice(np.delete(np.arange(n), i), size=5, replace=False) for i in range(n)])
        a = rng.integers(0, 3, size=cand.shape) * 50
        succ = np.empty(n, dtype=int)
        succ[perm] = np.roll(perm, -1)
        fwd = np.mean([any(cand[i, j] == succ[i] and a[i, j] == 0 for j in range(5)) for i in range(n)])
        bwd = np.mean([any(cand[succ[i], j] == i and a[succ[i], j] == 0 for j in range(5)) for i in range(n)])
        assert abs(A.kept_share(cand, a, perm) - (fwd + bwd) / 2) < 1e-12

    # trace parsing at TRACE_LEVEL = 2
    assert A.trial_times(LOG) == [0.71, 0.90, 1.10, 1.31]
    assert A.trial_costs(LOG) == [(1, 72079058), (2, 72080111), (3, 72078000), (4, 72079999)]
    assert A.trial_times(LOG + "  5: Cost = 7207") == [0.71, 0.90, 1.10, 1.31]           # cut-off last line
    assert A.trial_times("") == [] and A.trial_times(None) == []
    assert A.trial_times(LOG.replace("  2: Cost", "  7: Cost")) == [0.71]               # consistent prefix only
    assert A.trials_within([0.71, 0.90, 1.10, 1.31], offset=10.0, budget=11.1) == [0, 0, 0, 3]
    assert A.trials_within([1.0, 2.0, 3.0, 4.0], offset=0.0, budget=4.0) == [1, 2, 3, 4]
    assert [e[1] for e in T.parse_lkh_output(LOG)["events"]] == [72079058, 72078000]      # improvements only
    assert T.parse_lkh_output(LOG)["cost_min"] == 79367486       # POPMUSIC's line: why the module does not return it

    # statistics
    assert A.sign_test_p(0, 0) == 1.0 and A.sign_test_p(5, 5) == 1.0
    assert abs(A.sign_test_p(18, 6) - 2 * 190051 / 2 ** 24) < 1e-15            # 190051 = sum of C(24, j), j <= 6
    try:
        from scipy.stats import binomtest
        for w, l in [(1, 0), (0, 6), (18, 6), (17, 7), (9, 9), (3, 20), (24, 0)]:
            assert abs(A.sign_test_p(w, l) - binomtest(w, w + l, 0.5).pvalue) < 1e-12, (w, l)
    except ImportError:
        print("scipy not available: the comparison with scipy.stats.binomtest was skipped")
    assert A.holm([0.01, 0.04]) == [0.02, 0.04] and A.holm([0.04, 0.01]) == [0.04, 0.02]
    assert A.holm([0.03, 0.04]) == [0.06, 0.06] and A.holm([0.6, 0.7]) == [1.0, 1.0]
    values = np.array([-0.02, -0.01, 0.0, 0.01, -0.03, -0.005])
    s1, s2 = A.paired_stats(values, 11), A.paired_stats(values, 11)
    assert s1 == s2 and (s1["win"], s1["tie"], s1["loss"]) == (4, 1, 1)
    assert s1["ci95_low_pct"] <= s1["mean_delta_pct"] <= s1["ci95_high_pct"]
    assert abs(s1["mean_delta_pct"] - values.mean()) < 1e-15 and s1["sign_p"] == A.sign_test_p(4, 1)

    # candidate file with alpha-values and fathers: write and read back
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "x.cand")
        dad = np.array([-1, 0, 1, 2])
        T.write_candidates(path, cand_native, alpha_native, dad)
        back = T.read_candidates(path, k_pad=3)
        assert np.array_equal(back["cand"], cand_native) and np.array_equal(back["dad"], dad)
        assert np.array_equal(back["alpha"][cand_native >= 0], alpha_native[cand_native >= 0])
        T.write_candidates(path, cand_native)                                # the writer of runs v1-v8
        back = T.read_candidates(path, k_pad=3)
        assert np.array_equal(back["dad"], np.full(4, -1))
        assert np.array_equal(back["alpha"][cand_native >= 0], A.rank_alpha(cand_native)[cand_native >= 0])
    print("unit tests passed")


def lkh_tests(T, A, lkh):
    with tempfile.TemporaryDirectory() as tmp:
        for dist, n, gen in (("uniform", 1500, 41), ("clustered", 1500, 42)):
            xy = T.to_int_coords(T.gen_instance(n, dist, np.random.default_rng(gen)))
            first = A.task_first_trial(lkh, tmp, 101, xy, k=5, timeout=120)
            old = T.task_first_trial(lkh, tmp, 101, xy, k=5, timeout=120)
            assert np.array_equal(first["tour"], old["tour"]) and np.array_equal(first["cand"], old["cand"])
            assert first["cost"] == old["cost"]                              # TRACE_LEVEL does not change the trial
            assert len(first["events"]) == 1 and first["events"][0][1] == int(first["cost"])
            cand, alpha, dad = first["cand"], first["alpha"], first["dad"]
            present = cand >= 0
            assert cand.shape == (n, 5) == alpha.shape and dad.shape == (n,) and int((dad >= 0).sum()) == n - 1
            assert np.all(np.diff(np.where(present, alpha, 2**62), axis=1) >= 0)          # ascending alpha
            zero = present & (alpha == 0)
            assert 1.99 <= zero.sum() / n <= 2.05                            # the minimum 1-tree and rare ties
            _, is_tree = A.tree_alpha(cand, A.tree_edge_keys(cand, alpha))
            assert np.array_equal(is_tree, zero)
            native = A.kept_share(cand, alpha, first["tour"])
            rank = A.kept_share(cand, A.rank_alpha(cand), first["tour"])
            assert 0.6 < native < 0.9 and 0.35 < rank < 0.55 and native > rank + 0.2

            # the guided task with the old writer equals tsplab's task (fixed number of trials)
            fixed = {"MAX_TRIALS": 12}
            mine = A.task_guided(lkh, tmp, 600, 101, xy, cand, init_tour=first["tour"], extra_params=fixed)
            theirs = T.task_guided(lkh, tmp, 600, 101, xy, cand, init_tour=first["tour"], extra_params=fixed)
            assert [c for _, c, _ in mine["events"]] == [c for _, c, _ in theirs["events"]]
            assert mine["cost_check"] == theirs["cost_check"] and mine["returncode"] == 0
            assert len(mine["trial_times"]) == 12 and mine["trial_times"] == sorted(mine["trial_times"])
            # LKH's own alpha-values and fathers: a different file for LKH, the same or a different search
            own = A.task_guided(lkh, tmp, 600, 101, xy, cand, alpha=alpha, dad=dad, init_tour=first["tour"],
                                extra_params=fixed)
            assert own["returncode"] == 0 and len(own["trial_times"]) == 12

            # the uninterrupted task: the same improvements as tsplab's, and one end time per trial
            params = {"CANDIDATE_SET_TYPE": "POPMUSIC", "SUBGRADIENT": "NO", "MAX_CANDIDATES": 5, "MAX_TRIALS": 15}
            mine = A.task_anytime(lkh, tmp, 600, 101, xy=xy, params=params)
            theirs = T.task_anytime(lkh, tmp, 600, 101, xy=xy, params=params)
            assert [c for _, c, _ in mine["events"]] == [c for _, c, _ in theirs["events"]]
            assert mine["events"][0][1] == int(first["cost"])                 # the shared first trial
            assert len(mine["trial_times"]) == 15 and mine["trial_times"] == sorted(mine["trial_times"])
            assert set(mine) == {"events", "pre_time", "wall", "returncode", "timed_out", "t_start", "trial_times"}

            check = A.encoding_selfcheck(lkh, tmp, xy, 101, trials=20)
            diff = check["first_trial_with_a_different_cost"]
            assert check["native_file_equals_lkh_file"] and diff["native_dad"] is None
            assert check["tree_rule_matches_lkh"] and diff["rank_dad"] is not None and diff["rank_dad"] >= 2
            assert check["kept_share_native"] > check["kept_share_rank"]
            print(dist, n, "kept share", round(native, 3), round(rank, 3), "first differing trial", diff)
    print("LKH tests passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of tsplab.py")
    ap.add_argument("--lkh", default=None, help="LKH-3.0.14 binary; without it only the unit tests run")
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import tsplab as T
    import alphaenc as A
    unit_tests(T, A)
    if a.lkh:
        lkh_tests(T, A, a.lkh)
