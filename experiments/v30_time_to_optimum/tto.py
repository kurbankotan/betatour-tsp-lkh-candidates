"""Run v30: time to the proven optimum with different candidate lists. Helper functions embedded in notebook 33.

LKH-3.0.14 stops a run when its tour reaches OPTIMUM (STOP_AT_OPTIMUM = YES) or when its CPU clock, counted from the
start of the program, reaches TOTAL_TIME_LIMIT. The time to the optimum of a run is LKH-3's Time.total; for the
restarted arms the pause before the restart is added. This module holds

  * the first trial of a seed (ascent-free POPMUSIC-5, one trial), with its lists, alpha-values, fathers and CPU time;
  * the candidate files of the restarted arms (pop_r, beta, count, mix, union) and the time to build them;
  * one LKH-3 run that stops at the optimum or at the cap, with or without a candidate file and an initial tour;
  * the time to the optimum, PAR scores, sign tests with Holm correction and bootstrap intervals.

The candidate-file encodings come from run v21 (alphaenc.py, unchanged).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time

import numpy as np

import tsplab as T

try:
    import alphaenc_v21 as A        # the notebook writes run v21's module under this name
except ImportError:                 # pragma: no cover - repository layout (tests)
    import alphaenc as A

VERSION = "v30-tto-1"
ARMS = ("pop", "alpha", "pop_r", "beta", "count", "mix", "union")
UNINTERRUPTED = {"pop": {"CANDIDATE_SET_TYPE": "POPMUSIC", "SUBGRADIENT": "NO"},
                 "alpha": {"CANDIDATE_SET_TYPE": "ALPHA", "SUBGRADIENT": "YES"}}
RESTARTED = ("pop_r", "beta", "count", "mix", "union")
MODEL_ARMS = ("beta", "mix", "union")
MAX_TRIALS = 100_000_000


# ----------------------------------------------------------------------------------------------------- LKH runs
def first_trial(lkh_bin, workroot, seed, xy, problem_file, ew, optimum, k=5, timeout=900):
    """One ascent-free POPMUSIC-k trial (the first trial of `pop` with the same seed). Returns the tour, its length,
    the run's own lists with LKH-3's alpha-values and fathers, and LKH-3's CPU time (Time.total)."""
    params = {"CANDIDATE_SET_TYPE": "POPMUSIC", "SUBGRADIENT": "NO", "MAX_TRIALS": 1, "MAX_CANDIDATES": int(k),
              "SEED": int(seed), "TRACE_LEVEL": 1, "OPTIMUM": int(optimum), "STOP_AT_OPTIMUM": "YES"}
    res = T.lkh_solve(lkh_bin, problem_file=problem_file, ew=ew, params=params, workroot=workroot, want_cands=True,
                      timeout=timeout)
    n = len(xy)
    if res["returncode"] != 0 or not T.is_valid_tour(res["tour"], n):
        raise RuntimeError(f"first trial failed (seed {seed}, rc={res['returncode']}): {res['stdout'][-600:]}")
    if "cand" not in res or res["time_total"] is None:
        raise RuntimeError("first trial: no candidate file or no Time.total: " + res["stdout"][-600:])
    cand, alpha, dad = res["cand"]["cand"], res["cand"]["alpha"], np.asarray(res["cand"]["dad"], dtype=np.int64)
    if cand.shape != (n, k) or alpha.shape != (n, k) or dad.shape != (n,):
        raise RuntimeError(f"first trial: unexpected list shape {cand.shape}")
    return {"tour": np.asarray(res["tour"], dtype=np.int32), "cost": float(T.tour_length(xy, res["tour"], ew)),
            "cand": cand.astype(np.int32), "alpha": alpha.astype(np.int64), "dad": dad,
            "time_total": float(res["time_total"]), "wall": float(res["wall"]), "pre_time": res["pre_time"]}


def lkh_run(lkh_bin, workroot, problem_file, xy, ew, seed, optimum, cpu_limit, params=None, files=None,
            init_tour=None, timeout=None, max_trials=MAX_TRIALS, return_tour=False):
    """One LKH-3 run that stops at the optimum or when LKH-3's CPU clock reaches cpu_limit (TOTAL_TIME_LIMIT).
    files = (cand, alpha, dad) gives a candidate file with zero penalties and no ascent (as in run v21); init_tour
    starts the first trial from that tour. Returns the recomputed length of the output tour, whether it is the
    optimum, LKH-3's Time.total (CPU seconds), the wall time and the number of trials (and the tour if asked)."""
    n = len(xy)
    os.makedirs(workroot, exist_ok=True)
    wd = tempfile.mkdtemp(dir=workroot, prefix="r_")
    try:
        p = dict(params or {})
        if files is not None:
            cand, alpha, dad = files
            cand = np.asarray(cand)
            cf, pif = os.path.join(wd, "g.cand"), os.path.join(wd, "g.pi")
            T.write_candidates(cf, cand, alpha, dad)
            T.write_pi(pif, np.zeros(n))
            p.update(CANDIDATE_FILE=cf, PI_FILE=pif, SUBGRADIENT="NO", MAX_CANDIDATES=int(cand.shape[1]))
        else:
            p.setdefault("MAX_CANDIDATES", 5)
        if init_tour is not None:
            tp = os.path.join(wd, "init.tour")
            T.write_tour(tp, init_tour)
            p["INITIAL_TOUR_FILE"] = tp
        limit = max(float(cpu_limit), 0.05)
        p.update(RUNS=1, MAX_TRIALS=int(max_trials), SEED=int(seed), OPTIMUM=int(optimum), STOP_AT_OPTIMUM="YES",
                 TOTAL_TIME_LIMIT=round(limit, 3), TRACE_LEVEL=1)
        res = T.run_lkh(lkh_bin, problem_file, p, os.path.join(wd, "run"),
                        timeout=timeout if timeout is not None else 3.0 * limit + 120.0, keep_stdout=True)
        ok = T.is_valid_tour(res["tour"], n)
        cost = float(T.tour_length(xy, res["tour"], ew)) if ok else None
        stdout = res.get("stdout") or ""
        out = {"cost": cost, "found": bool(ok and cost == float(optimum)), "shorter": bool(ok and cost < optimum),
               "time_total": None if res["time_total"] is None else float(res["time_total"]),
               "wall": float(res["wall"]), "trials": res["trials"], "pre_time": res["pre_time"],
               "returncode": res["returncode"], "timed_out": bool(res["timed_out"]), "cost_min": res["cost_min"],
               "new_optimum": "New OPTIMUM" in stdout, "cpu_limit": limit}
        if not ok or res["returncode"] != 0 or res["time_total"] is None:
            out["stdout_tail"] = stdout[-1500:] + (res.get("stderr") or "")[-500:]
        if return_tour:
            out["tour"] = np.asarray(res["tour"], dtype=np.int64) if ok else None
        return out
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ candidate files
def _valid(cand, n, k_max):
    cand = np.asarray(cand)
    assert cand.ndim == 2 and cand.shape[0] == n and cand.shape[1] <= k_max, cand.shape
    assert np.all((cand >= -1) & (cand < n)), "invalid candidate"
    assert not np.any(cand == np.arange(n)[:, None]), "a node is its own candidate"
    for i in range(cand.shape[1]):
        for j in range(i + 1, cand.shape[1]):
            assert not np.any((cand[:, i] == cand[:, j]) & (cand[:, i] >= 0)), "repeated candidate"
    assert np.all((cand >= 0).sum(1) >= 2), "a node with fewer than two candidates"


def arm_lists(firsts, prediction, k=5, k_union=10, k_learned=3):
    """Candidate files (cand, alpha, dad) of the restarted arms for every seed, the time to build each arm's lists
    (wall seconds; the inference of the model is not included) and the long rankings (BetaTour and counting).

    pop_r : the seed's own POPMUSIC-k lists with LKH-3's alpha-values and fathers (its own file).
    beta  : BetaTour's top k; count: top k by counting over the first-trial tours on BetaTour's graph (ties by
            length); mix: BetaTour's top k_learned, then the seed's POPMUSIC-k, then BetaTour's next ones (runs v21
            to v23); union: BetaTour's top k, then the seed's POPMUSIC-k entries not yet listed (up to k_union).
            All four in the tree encoding of run v21 (alpha = 0 on the edges of the seed's own minimum 1-tree) with
            the fathers of the seed's own file."""
    W = len(firsts)
    n = len(firsts[0]["tour"])
    rank = np.asarray(prediction["rank"], dtype=np.int64)
    NB = np.asarray(prediction["NB"], dtype=np.int64)
    D = np.asarray(prediction["D"], dtype=np.float64)
    assert rank.shape == NB.shape == D.shape and rank.shape[0] == n and rank.shape[1] >= k_union
    start = time.perf_counter()
    succ = np.stack([T.tour_succ_pred(f["tour"])[0] for f in firsts], axis=1)
    tree_keys = [A.tree_edge_keys(f["cand"], f["alpha"]) for f in firsts]
    common = time.perf_counter() - start

    lists, build = {arm: [] for arm in RESTARTED}, {}
    start = time.perf_counter()
    for f in firsts:
        lists["pop_r"].append((f["cand"].astype(np.int64), f["alpha"].astype(np.int64), f["dad"]))
    build["pop_r"] = time.perf_counter() - start

    start = time.perf_counter()
    top = rank[:, :k]
    for j, f in enumerate(firsts):
        lists["beta"].append((top, A.tree_alpha(top, tree_keys[j])[0], f["dad"]))
    build["beta"] = common + time.perf_counter() - start

    start = time.perf_counter()
    count_rank = T.frequency_candidates(NB, D, T.ensemble_frequency(NB, succ), k=NB.shape[1]).astype(np.int64)
    ctop = count_rank[:, :k]
    for j, f in enumerate(firsts):
        lists["count"].append((ctop, A.tree_alpha(ctop, tree_keys[j])[0], f["dad"]))
    build["count"] = common + time.perf_counter() - start

    start = time.perf_counter()
    for j, f in enumerate(firsts):
        mix = T.mix_candidates(rank, f["cand"], k_first=k_learned, k=k, fill=rank[:, k_learned:]).astype(np.int64)
        lists["mix"].append((mix, A.tree_alpha(mix, tree_keys[j])[0], f["dad"]))
    build["mix"] = common + time.perf_counter() - start

    start = time.perf_counter()
    for j, f in enumerate(firsts):
        uni = T.mix_candidates(rank, f["cand"], k_first=k, k=k_union).astype(np.int64)
        lists["union"].append((uni, A.tree_alpha(uni, tree_keys[j])[0], f["dad"]))
    build["union"] = common + time.perf_counter() - start

    for arm in RESTARTED:
        assert len(lists[arm]) == W
        for cand, alpha, dad in lists[arm]:
            _valid(cand, n, k_union if arm == "union" else k)
            assert alpha.shape == cand.shape and dad.shape == (n,)
    return lists, {arm: float(v) for arm, v in build.items()}, rank, count_rank


def pauses(first_cpu, model_s, build_s):
    """Pause of each restarted arm: the longest CPU time of the first trials (they run in parallel), plus the
    inference of the model for the arms that use it, plus the time to build the arm's lists."""
    longest = float(np.max(first_cpu))
    return {arm: longest + (float(model_s) if arm in MODEL_ARMS else 0.0) + float(build_s[arm]) for arm in RESTARTED}


def missed_optimal_edges(cand, opt_tour):
    """Edges of the optimal tour that are in neither end's list."""
    cand = np.asarray(cand)
    return int(T.candidate_coverage(cand, opt_tour, ks=[cand.shape[1]])[cand.shape[1]]["missing"])


# ------------------------------------------------------------------------------------------------------ scores
def time_to_optimum(record, cap):
    """Seconds to the optimum of one run record, or inf when the run did not reach it within the cap."""
    if record.get("first_found"):
        t = float(record["first_cpu"])
    elif record.get("found") and record.get("time_total") is not None:
        t = float(record.get("pause", 0.0)) + float(record["time_total"])
    else:
        return float("inf")
    return t if t <= cap + 1e-9 else float("inf")


def par(times, cap, factor=10.0):
    t = np.asarray(times, dtype=np.float64)
    return float(np.where(np.isfinite(t), t, factor * cap).mean())


def holm(pvalues):
    """Holm-adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def compare(first, second, tol=0.0, rel_tol=1e-9):
    """Paired comparison of two arrays (lower is better): wins of the first, ties, losses, exact two-sided sign p.
    Differences of at most tol (a number or one value per pair) count as ties."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    tie = np.isclose(a, b, rtol=rel_tol, atol=1e-9) | (np.abs(a - b) <= np.asarray(tol, dtype=np.float64))
    wins, losses = int(((a < b) & ~tie).sum()), int(((a > b) & ~tie).sum())
    return {"wins": wins, "ties": int(tie.sum()), "losses": losses, "p": A.sign_test_p(wins, losses)}


def ratio_interval(first, second, seed, replicates=5000):
    """Geometric mean of first/second over instances and a 95% percentile interval (bootstrap over instances)."""
    r = np.log(np.asarray(first, dtype=np.float64) / np.asarray(second, dtype=np.float64))
    rng = np.random.default_rng(seed)
    boot = r[rng.integers(0, len(r), size=(replicates, len(r)))].mean(1)
    return {"ratio": float(np.exp(r.mean())), "low": float(np.exp(np.quantile(boot, 0.025))),
            "high": float(np.exp(np.quantile(boot, 0.975)))}
