"""Run v21: alpha-encoding control. Helper functions embedded in notebook 23.

A candidate file of LKH-3 holds, per node, the father of the node in the minimum 1-tree (Dad) and the candidates
with their alpha-values. LKH-3 uses both fields after it has read the file:

  * Every trial after the first starts from a tour built by a random walk (ChooseInitialTour.c). The walk keeps an
    edge of the best or next best tour only if that edge is a candidate whose alpha-value is zero (case C). With
    CANDIDATE_SET_TYPE = POPMUSIC and SUBGRADIENT = NO, LKH gives alpha = 0 to the edges of its minimum 1-tree, about
    two candidates per node.
  * Until the first tour of a run is recorded, the macro Near(a, b) is true for the edges of the minimum 1-tree
    (a is the father of b or b the father of a); the first trial does not break these edges first.

The candidate files of runs v1-v8 gave alpha = 0 to the first candidate of each node only (alpha = 0, 100, 200, ...)
and wrote no father, so their restarted arms kept fewer edges of the best tour at every trial.

This module holds
  * the two encodings compared in run v21 ("rank": 0, 100, 200, ..., no father; "tree": alpha = 0 exactly on the
    candidates that LKH's own lists of the same seed mark with alpha = 0, and the fathers of those lists),
  * LKH task functions that mirror tsplab.task_first_trial, task_anytime and task_guided, return the alpha-values
    and fathers of LKH's own lists and count the completed trials (TRACE_LEVEL = 2),
  * the share of a tour's edges that LKH's restart keeps under a given candidate file,
  * a self-check that compares LKH's search under its own candidate file and under rewritten files, and
  * the paired statistics of the analysis.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from math import comb

import numpy as np

import tsplab as T

VERSION = "v21-alphaenc-3"

# One line per completed trial at TRACE_LEVEL = 2: "* 12: Cost = 123, Time = 1.23 sec." (new best tour of the run)
# or "  13: Cost = 124, Time = 1.31 sec." (no improvement). LKH reports its own CPU time since the search began.
TRIAL_RX = re.compile(r"^(\*?)[ \t]*(\d+):[ \t]*Cost[ \t]*=[ \t]*(-?\d+)[^\n]*?Time[ \t]*=[ \t]*([\d.]+)[ \t]*sec", re.M)


# ----------------------------------------------------------------------------- encodings
def edge_keys(a, b, n):
    """Undirected edge keys min * n + max for 0-based node arrays a and b."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return np.minimum(a, b) * int(n) + np.maximum(a, b)


def tree_edge_keys(cand_native, alpha_native):
    """Keys of the edges that LKH's own lists mark with alpha = 0 (the edges of its minimum 1-tree).

    cand_native: (n, K) 0-based neighbour ids, -1 padded; alpha_native: (n, K) alpha-values written by LKH."""
    cand_native = np.asarray(cand_native, dtype=np.int64)
    alpha_native = np.asarray(alpha_native, dtype=np.int64)
    assert cand_native.shape == alpha_native.shape and cand_native.ndim == 2
    n = cand_native.shape[0]
    rows = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], cand_native.shape)
    mask = (cand_native >= 0) & (alpha_native == 0)
    return np.unique(edge_keys(rows[mask], cand_native[mask], n))


def rank_alpha(cand):
    """The encoding of runs v1-v8: alpha = 0, 100, 200, ... in list order (tsplab.write_candidates default)."""
    cand = np.asarray(cand)
    return np.broadcast_to(100 * np.arange(cand.shape[1], dtype=np.int64)[None, :], cand.shape).copy()


def tree_alpha(cand, tree_keys):
    """Tree encoding: alpha = 0 for the entries of `cand` whose edge is in `tree_keys`, and 100, 200, ... by list
    position for the other entries. LKH sorts each list by alpha and then by length when it reads the file, so the
    alpha = 0 entries move to the front and the other entries keep their order. Returns (alpha, is_tree)."""
    cand = np.asarray(cand, dtype=np.int64)
    n, k = cand.shape
    rows = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], cand.shape)
    valid = cand >= 0
    keys = edge_keys(rows, np.where(valid, cand, 0), n)
    is_tree = valid & np.isin(keys, np.asarray(tree_keys, dtype=np.int64))
    alpha = np.where(is_tree, 0, 100 * (np.arange(k, dtype=np.int64)[None, :] + 1))
    return alpha.astype(np.int64), is_tree


def kept_share(cand, alpha, tour):
    """Share of the edges of `tour` that LKH's restart can keep under a candidate file (cand, alpha): the edge (i, j)
    is kept when the walk reaches i and j is a candidate of i with alpha = 0. Mean over both walking directions."""
    cand = np.asarray(cand, dtype=np.int64)
    alpha = np.asarray(alpha, dtype=np.int64)
    tour = np.asarray(tour, dtype=np.int64)
    n = len(tour)
    assert cand.shape[0] == n and cand.shape == alpha.shape
    succ = np.empty(n, dtype=np.int64)
    pred = np.empty(n, dtype=np.int64)
    succ[tour] = np.roll(tour, -1)
    pred[tour] = np.roll(tour, 1)
    zero = (cand >= 0) & (alpha == 0)
    forward = ((cand == succ[:, None]) & zero).any(1).mean()
    backward = ((cand == pred[:, None]) & zero).any(1).mean()
    return float((forward + backward) / 2.0)


# ----------------------------------------------------------------------------- LKH tasks
def trial_times(stdout):
    """End times (LKH's clock, seconds since its search began) of the trials printed at TRACE_LEVEL = 2."""
    rows = [(int(k), float(t)) for _, k, _, t in TRIAL_RX.findall(stdout or "")]
    times = []
    for index, (k, t) in enumerate(rows, 1):
        if k != index:          # a truncated or foreign line: keep the consistent prefix only
            break
        times.append(t)
    return times


def trial_costs(stdout):
    """(trial number, cost) of every trial printed at TRACE_LEVEL = 2, in order."""
    return [(int(k), int(c)) for _, k, c, _ in TRIAL_RX.findall(stdout or "")]


def trials_within(times, offset, budget, fractions=(0.25, 0.5, 0.75, 1.0)):
    """Number of completed trials whose absolute end time (offset + LKH time) is within each fraction of the budget."""
    absolute = np.asarray(times, dtype=np.float64) + float(offset)
    return [int((absolute <= f * budget + 1e-9).sum()) for f in fractions]


def task_first_trial(lkh_bin, workroot, seed, xy, problem_file=None, ew="EUC_2D", k=T.CAND_K, timeout=None):
    """tsplab.task_first_trial (ascent-free POPMUSIC lists and one trial) at TRACE_LEVEL = 2 that also returns the
    alpha-values and the fathers (0-based, -1 for the root) of the run's own candidate lists, as LKH wrote them to
    its CANDIDATE_FILE. tsplab reads each list in ascending alpha order, which is the order of LKH's file."""
    p = {"CANDIDATE_SET_TYPE": "POPMUSIC", "SUBGRADIENT": "NO", "MAX_TRIALS": 1, "MAX_CANDIDATES": int(k),
         "SEED": int(seed), "TRACE_LEVEL": 2}
    res = T.lkh_solve(lkh_bin, xy=None if problem_file else xy, problem_file=problem_file, ew=ew, params=p,
                      workroot=workroot, want_cands=True, timeout=timeout)
    n = len(xy)
    if not T.is_valid_tour(res["tour"], n):
        raise RuntimeError(f"first trial gave no valid tour (rc={res['returncode']}): {res['stdout'][-500:]}")
    if "cand" not in res:
        raise RuntimeError("LKH did not write the candidate file: " + res["stdout"][-500:])
    cand, alpha, dad = res["cand"]["cand"], res["cand"]["alpha"], np.asarray(res["cand"]["dad"], dtype=np.int64)
    if cand.shape != (n, k) or alpha.shape != (n, k) or dad.shape != (n,):
        raise RuntimeError(f"unexpected candidate list shape {cand.shape}")       # no silent truncation to k
    return {"tour": res["tour"].astype(np.int32), "cost": float(T.tour_length(xy, res["tour"], ew)),
            "cand": cand.astype(np.int32), "alpha": alpha.astype(np.int64), "dad": dad, "events": res["events"],
            "pre_time": res["pre_time"], "wall": res["wall"], "t_start": res["t_start"]}


def task_anytime(lkh_bin, workroot, budget, seed, xy=None, problem_file=None, ew="EUC_2D", params=None,
                 timeout=None):
    """tsplab.task_anytime with TRACE_LEVEL = 2; also returns the end time of every completed trial. At this trace
    level POPMUSIC prints its own "Cost.min" and "Cand.min" lines first, so tsplab's cost_min and cand_stats would
    describe the POPMUSIC sample tours; they are not returned."""
    p = {"TIME_LIMIT": float(budget), "MAX_TRIALS": 100_000_000, "SEED": int(seed), "TRACE_LEVEL": 2}
    p.update(params or {})
    res = T.lkh_solve(lkh_bin, xy=xy, problem_file=problem_file, ew=ew, params=p, workroot=workroot,
                      timeout=budget * 3 + 600 if timeout is None else timeout, keep_stdout=True)
    out = {k: res.get(k) for k in ("events", "pre_time", "wall", "returncode", "timed_out", "t_start")}
    out["trial_times"] = trial_times(res.get("stdout"))
    return out


def task_guided(lkh_bin, workroot, budget, seed, xy, cand, alpha=None, dad=None, init_tour=None, problem_file=None,
                ew="EUC_2D", extra_params=None, timeout=None):
    """tsplab.task_guided without fixed edges (candidate file with the given alpha-values and fathers, zero
    penalties, initial tour, no ascent) with TRACE_LEVEL = 2; also returns the end time of every completed trial.
    alpha = None writes 0, 100, 200, ... and dad = None writes no father, as in runs v1-v8."""
    os.makedirs(workroot or tempfile.gettempdir(), exist_ok=True)
    wd = tempfile.mkdtemp(dir=workroot, prefix="g_")
    try:
        n = len(xy)
        pf = problem_file
        if pf is None:
            pf = os.path.join(wd, "p.tsp")
            T.write_tsp(pf, xy, ew=ew)
        cf, pif = os.path.join(wd, "g.cand"), os.path.join(wd, "g.pi")
        T.write_candidates(cf, cand, alpha, dad)
        T.write_pi(pif, np.zeros(n))
        params = {"SUBGRADIENT": "NO", "TIME_LIMIT": float(budget), "MAX_TRIALS": 100_000_000,
                  "SEED": int(seed), "MAX_CANDIDATES": int(np.asarray(cand).shape[1]), "TRACE_LEVEL": 2}
        if init_tour is not None:
            tp = os.path.join(wd, "init.tour")
            T.write_tour(tp, init_tour)
            params["INITIAL_TOUR_FILE"] = tp
        params.update(extra_params or {})
        params["CANDIDATE_FILE"], params["PI_FILE"] = cf, pif
        res = T.run_lkh(lkh_bin, pf, params, os.path.join(wd, "run"),
                        timeout=budget * 3 + 600 if timeout is None else timeout, keep_stdout=True)
        ok = T.is_valid_tour(res["tour"], n)
        out = {k: res.get(k) for k in ("events", "pre_time", "wall", "returncode", "timed_out", "t_start")}
        out["cost_check"] = float(T.tour_length(xy, res["tour"], ew)) if ok else None
        out["trial_times"] = trial_times(res.get("stdout"))
        if not ok:
            out["stdout_tail"] = (res.get("stdout") or "")[-800:]
        return out
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def encoding_selfcheck(lkh_bin, workroot, xy, seed, trials=25, k=T.CAND_K, ew="EUC_2D", timeout=300):
    """Compares, on the machine of the run, LKH's search under the candidate file it wrote itself with its search
    under four rewritten files: LKH's own alpha-values or 0, 100, 200, ..., each with and without the fathers.
    All continuations start from the tour of the first trial, have a fixed number of trials and no time limit, so
    their traces are deterministic. Expected from the source code:
      * own alpha-values with fathers: the same cost at every trial as under LKH's own file;
      * 0, 100, 200, ... with fathers: a different search from the second trial on (restart walk, case C);
      * without fathers: a difference, if any, can start in the first trial (Near() before a tour is recorded).
    Also checks that the tree rule reproduces the entries that LKH marks with alpha = 0."""
    os.makedirs(workroot or tempfile.gettempdir(), exist_ok=True)
    wd = tempfile.mkdtemp(dir=workroot, prefix="enc_")
    try:
        n = len(xy)
        pf, own = os.path.join(wd, "p.tsp"), os.path.join(wd, "own.cand")
        T.write_tsp(pf, xy, ew=ew)
        first = T.run_lkh(lkh_bin, pf, {"CANDIDATE_SET_TYPE": "POPMUSIC", "SUBGRADIENT": "NO", "MAX_TRIALS": 1,
                                        "MAX_CANDIDATES": int(k), "SEED": int(seed), "TRACE_LEVEL": 2,
                                        "CANDIDATE_FILE": own}, os.path.join(wd, "first"), timeout=timeout)
        if first["returncode"] != 0 or not T.is_valid_tour(first["tour"], n) or not os.path.exists(own):
            raise RuntimeError("self-check: the first trial failed: " + (first.get("stdout") or "")[-500:])
        lists = T.read_candidates(own, k_pad=k)
        cand, alpha, dad = lists["cand"], lists["alpha"], lists["dad"]
        if cand.shape != (n, k):
            raise RuntimeError(f"self-check: unexpected candidate list shape {cand.shape}")
        pif, tp = os.path.join(wd, "zero.pi"), os.path.join(wd, "first.tour")
        T.write_pi(pif, np.zeros(n))
        T.write_tour(tp, first["tour"])
        variants = {"native_dad": (alpha, dad), "native_nodad": (alpha, None),
                    "rank_dad": (rank_alpha(cand), dad), "rank_nodad": (rank_alpha(cand), None)}
        files = {"lkh_file": own}
        for name, (a, d) in variants.items():
            files[name] = os.path.join(wd, name + ".cand")
            T.write_candidates(files[name], cand, a, d)
        traces = {}
        for name, cf in files.items():
            res = T.run_lkh(lkh_bin, pf, {"SUBGRADIENT": "NO", "MAX_TRIALS": int(trials), "MAX_CANDIDATES": int(k),
                                          "SEED": int(seed), "TRACE_LEVEL": 2, "INITIAL_TOUR_FILE": tp,
                                          "CANDIDATE_FILE": cf, "PI_FILE": pif}, os.path.join(wd, name),
                            timeout=timeout, keep_stdout=True)
            traces[name] = trial_costs(res.get("stdout"))
            if res["returncode"] != 0 or [k_ for k_, _ in traces[name]] != list(range(1, int(trials) + 1)):
                raise RuntimeError(f"self-check: continuation '{name}' failed: " + (res.get("stdout") or "")[-500:])

        def first_difference(name):
            trials_ = [k_ for (k_, a), (_, b) in zip(traces[name], traces["lkh_file"]) if a != b]
            return int(trials_[0]) if trials_ else None

        valid = cand >= 0
        _, is_tree = tree_alpha(cand, tree_edge_keys(cand, alpha))
        return {"nodes": int(n), "trials": int(trials), "seed": int(seed),
                "native_file_equals_lkh_file": traces["native_dad"] == traces["lkh_file"],
                "first_trial_with_a_different_cost": {name: first_difference(name) for name in variants},
                "tree_rule_matches_lkh": bool(np.array_equal(is_tree, valid & (alpha == 0))),
                "candidates_per_node": float(valid.sum() / n),
                "zero_alpha_per_node": float((valid & (alpha == 0)).sum() / n),
                "nodes_with_a_father": int((dad >= 0).sum()),
                "kept_share_native": kept_share(cand, alpha, first["tour"]),
                "kept_share_rank": kept_share(cand, rank_alpha(cand), first["tour"]),
                "best_cost": {name: int(min(c for _, c in trace)) for name, trace in traces.items()}}
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# ----------------------------------------------------------------------------- statistics
def sign_test_p(wins, losses):
    """Exact two-sided sign test."""
    n, k = int(wins) + int(losses), min(int(wins), int(losses))
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * sum(comb(n, j) for j in range(k + 1)) / 2.0 ** n))


def paired_stats(values, bootstrap_seed, replicates=5000):
    """Paired percentage differences (first minus second arm): wins are negative values (the first arm is shorter)."""
    x = np.asarray(values, dtype=np.float64)
    assert x.ndim == 1 and len(x) and np.isfinite(x).all()
    wins, losses = int((x < -1e-12).sum()), int((x > 1e-12).sum())
    rng = np.random.default_rng(bootstrap_seed)
    means = x[rng.integers(0, len(x), size=(replicates, len(x)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return dict(instances=int(len(x)), win=wins, tie=int(len(x) - wins - losses), loss=losses,
                mean_delta_pct=float(x.mean()), median_delta_pct=float(np.median(x)),
                ci95_low_pct=float(low), ci95_high_pct=float(high), sign_p=sign_test_p(wins, losses))


def holm(p_values):
    """Holm-adjusted p-values in the input order."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, j in enumerate(order):
        running = max(running, (len(p) - rank) * p[j])
        adjusted[j] = min(1.0, running)
    return [float(v) for v in adjusted]
