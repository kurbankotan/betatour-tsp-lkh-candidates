"""Run v12: how much of the candidate deficit of POPMUSIC lists does LKH-3 repair by itself during the search?

Protocol v1, Stage 1 (mechanism diagnosis). For every instance with a reference tour, six native, ascent-free
POPMUSIC-5 LKH-3.0.14 processes (seeds 101-106) run in parallel for the instance's wall-clock budget, as the
uninterrupted baseline of runs v4-v8. Each process writes its initial candidate lists C0 (CANDIDATE_FILE) and rewrites
OUTPUT_TOUR_FILE after every new best tour; a poller copies every completed rewrite. Six separate single-trial runs
with the same seeds give the six first-trial tours, the M = 6 ensemble of the BetaTour and counting top-5 lists, as in
runs v4 and v5.

Within a run, LKH-3's AdjustCandidateSet appends the edges of every new best tour to the candidate lists (they are
removed again only when the run ends), so the live candidate set at time t is C0 together with every edge of every best
tour found by t. The analysis counts, per 1,000 nodes, the reference-tour edges missing from C0, from the BetaTour and
counting lists and from this dynamic set, and how many of the edges missing from C0 the search itself reached.

The poller can miss a best tour that LKH overwrote within one polling interval, so the dynamic set measured here is a
lower bound of LKH's live candidate set.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from math import comb
from pathlib import Path
from typing import Sequence

import numpy as np

VERSION = "v12-selfcorrect-1"
FRACTIONS = (0.25, 0.5, 0.75, 1.0)


# ----------------------------------------------------------------------------------------------- small helpers
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def xy_sha256(xy) -> str:
    a = np.ascontiguousarray(np.asarray(xy, dtype=np.float64))
    return sha256_bytes(a.tobytes())


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test; ties are excluded."""
    n, k = wins + losses, min(wins, losses)
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


# -------------------------------------------------------------------------------------------- edges and costs
def tour_keys(tour) -> np.ndarray:
    """Sorted unique undirected edge keys min*n+max of a tour (0-based node ids)."""
    t = np.asarray(tour, dtype=np.int64)
    n = len(t)
    a, b = t, np.roll(t, -1)
    return np.unique(np.minimum(a, b) * n + np.maximum(a, b))


def list_keys(cand) -> np.ndarray:
    """Sorted unique undirected edge keys of candidate lists (n, K), -1 padded. An edge counts as a candidate if
    either endpoint lists the other, as in tsplab.candidate_coverage."""
    c = np.asarray(cand, dtype=np.int64)
    n = c.shape[0]
    src = np.repeat(np.arange(n, dtype=np.int64), c.shape[1])
    dst = c.ravel()
    ok = (dst >= 0) & (dst != src)
    return np.unique(np.minimum(src[ok], dst[ok]) * n + np.maximum(src[ok], dst[ok]))


def checked_cost(T, xy, tour, ew: str) -> int:
    """Tour length with tsplab and an independent recomputation; raises on an invalid tour or a disagreement."""
    tour = np.asarray(tour, dtype=np.int64)
    if not T.is_valid_tour(tour, len(xy)):
        raise ValueError("invalid Hamilton tour")
    library = int(round(T.tour_length(xy, tour, ew)))
    a = np.asarray(xy, dtype=np.float64)[tour]
    b = np.asarray(xy, dtype=np.float64)[np.roll(tour, -1)]
    d = np.sqrt(((a - b) ** 2).sum(1))
    if ew == "EUC_2D":
        independent = int(np.floor(d + 0.5).sum())
    elif ew == "CEIL_2D":
        independent = int(np.ceil(d).sum())
    else:
        raise ValueError(f"unsupported EDGE_WEIGHT_TYPE {ew}")
    if independent != library:
        raise AssertionError(f"tour costs disagree: {independent} != {library}")
    return library


def parse_tour_bytes(data: bytes) -> np.ndarray:
    """0-based tour of an LKH TSPLIB tour file; raises if the tour section is incomplete."""
    _head, marker, body = data.decode("ascii").partition("TOUR_SECTION")
    if not marker:
        raise ValueError("tour file has no TOUR_SECTION")
    nodes = []
    for token in body.split():
        if token == "-1":
            return np.asarray(nodes, dtype=np.int64) - 1
        nodes.append(int(token))
    raise ValueError("tour section is not terminated by -1")


def tour_file_complete(data: bytes) -> bool:
    tail = data.rstrip()
    return b"TOUR_SECTION" in data and (tail.endswith(b"EOF") or tail.endswith(b"-1"))


def parse_trace(text: str) -> list[tuple[int, int, float]]:
    """(trial, cost, LKH time) of every '* trial: Cost = ...' line printed with TRACE_LEVEL = 1."""
    import re
    rx = re.compile(r"^\*\s*(\d+):\s*Cost\s*=\s*(-?\d+)[^\n]*?Time\s*=\s*([\d.]+)\s*sec", re.M)
    return [(int(a), int(b), float(c)) for a, b, c in rx.findall(text)]


# ------------------------------------------------------------------------------------------------ LKH runs
def _base_params(problem: Path, seed: int, k: int) -> dict:
    return {"PROBLEM_FILE": str(Path(problem).resolve()), "CANDIDATE_SET_TYPE": "POPMUSIC",
            "MAX_CANDIDATES": int(k), "SUBGRADIENT": "NO", "SEED": int(seed), "RUNS": 1}


def _stop(processes) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_first_trials(T, lkh: str, problem: Path, xy, ew: str, work: Path, seeds: Sequence[int],
                     k: int = 5, timeout_s: float = 900.0) -> dict:
    """The first trial of each seed's ascent-free POPMUSIC run (MAX_TRIALS = 1), all seeds in parallel. With the same
    seed this is the first trial of the corresponding full run. Returns tours, costs and candidate lists."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    processes, streams = [], []
    start = time.monotonic()
    try:
        for seed in seeds:
            base = work / f"seed_{seed}"
            base.mkdir()
            params = _base_params(problem, seed, k)
            params.update(MAX_TRIALS=1, TRACE_LEVEL=1, CANDIDATE_FILE=str(base / "c0.cand"),
                          OUTPUT_TOUR_FILE=str(base / "first.tour"))
            T.write_par(str(base / "run.par"), params)
            out = (base / "stdout.txt").open("w")
            err = (base / "stderr.txt").open("w")
            streams += [out, err]
            processes.append(subprocess.Popen([str(lkh), str((base / "run.par").resolve())], cwd=base,
                                              stdout=out, stderr=err, stdin=subprocess.DEVNULL))
        while any(p.poll() is None for p in processes):
            if time.monotonic() - start > timeout_s:
                raise TimeoutError("first trials exceeded their time limit")
            time.sleep(0.05)
    finally:
        _stop(processes)
        for s in streams:
            s.close()
    wall = time.monotonic() - start
    tours, costs, cands = [], [], []
    for seed, process in zip(seeds, processes):
        base = work / f"seed_{seed}"
        if process.returncode != 0:
            raise RuntimeError(f"first trial of seed {seed} failed (rc={process.returncode}): "
                               + (base / "stdout.txt").read_text()[-600:])
        tour = parse_tour_bytes((base / "first.tour").read_bytes())
        tours.append(tour.astype(np.int32))
        costs.append(checked_cost(T, xy, tour, ew))
        cands.append(T.read_candidates(str(base / "c0.cand"), k_pad=k)["cand"][:, :k].astype(np.int32))
    return {"tours": np.stack(tours), "costs": np.asarray(costs, dtype=np.int64), "cands": np.stack(cands),
            "wall_s": wall}


def run_native_pop(T, lkh: str, problem: Path, xy, ew: str, work: Path, seeds: Sequence[int], budget_s: float,
                   k: int = 5, poll_s: float = 0.02) -> dict:
    """Six uninterrupted ascent-free POPMUSIC-k processes under one monotonic wall-clock deadline (start-up,
    candidate construction and search included). Every completed rewrite of OUTPUT_TOUR_FILE is copied with its
    poll time; tours are parsed only after the deadline, so polling costs the search almost nothing."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    processes, streams = [], []
    captured = {seed: [] for seed in seeds}
    seen = {}

    def observe(now: float) -> None:
        for seed in seeds:
            path = work / f"seed_{seed}" / "best.tour"
            try:
                stamp = path.stat().st_mtime_ns
                if stamp == seen.get(seed):
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if not tour_file_complete(data):
                continue                                   # still being written; the next poll sees a new mtime
            seen[seed] = stamp
            if not captured[seed] or captured[seed][-1][1] != data:
                captured[seed].append((now, data))

    start = time.monotonic()
    deadline = start + budget_s
    try:
        for seed in seeds:
            base = work / f"seed_{seed}"
            base.mkdir()
            params = _base_params(problem, seed, k)
            params.update(MAX_TRIALS=100_000_000, TIME_LIMIT=max(10.0, 10.0 * budget_s), TRACE_LEVEL=1,
                          STOP_AT_OPTIMUM="NO", CANDIDATE_FILE=str(base / "c0.cand"),
                          OUTPUT_TOUR_FILE=str(base / "best.tour"))
            T.write_par(str(base / "run.par"), params)
            out = (base / "stdout.txt").open("w")
            err = (base / "stderr.txt").open("w")
            streams += [out, err]
            processes.append(subprocess.Popen([str(lkh), str((base / "run.par").resolve())], cwd=base,
                                              stdout=out, stderr=err, stdin=subprocess.DEVNULL))
        while time.monotonic() < deadline and any(p.poll() is None for p in processes):
            observe(time.monotonic() - start)
            time.sleep(poll_s)
    finally:
        _stop(processes)
        for s in streams:
            s.close()
    wall = time.monotonic() - start
    observe(min(wall, budget_s))                           # a rewrite completed just before the stop
    seeds_out = {}
    for seed in seeds:
        base = work / f"seed_{seed}"
        c0_path = base / "c0.cand"
        if not c0_path.exists():
            raise RuntimeError(f"LKH wrote no candidate file for seed {seed}: "
                               + (base / "stdout.txt").read_text()[-600:])
        seeds_out[seed] = {"captures": captured[seed],
                           "c0": T.read_candidates(str(c0_path), k_pad=k)["cand"][:, :k].astype(np.int32),
                           "trace": parse_trace((base / "stdout.txt").read_text(errors="replace"))}
    return {"seeds": seeds_out, "wall_s": wall, "budget_s": budget_s}


def decode_captures(T, xy, ew: str, captures, budget_s: float) -> tuple[list, list]:
    """Valid, strictly improving (elapsed_s, cost, tour) records within the budget; flags for skipped files."""
    out, flags = [], []
    for elapsed, data in captures:
        if elapsed > budget_s + 1e-9:
            flags.append("capture_after_deadline_skipped")
            continue
        try:
            tour = parse_tour_bytes(data)
            cost = checked_cost(T, xy, tour, ew)
        except (ValueError, AssertionError, UnicodeDecodeError):
            flags.append("invalid_capture_skipped")
            continue
        if out and cost >= out[-1][1]:
            flags.append("non_improving_capture_skipped")
            continue
        out.append((float(elapsed), int(cost), tour.astype(np.int32)))
    return out, sorted(set(flags))


# ------------------------------------------------------------------------------------------------ lists
def model_and_count_lists(M, T, model, device, xy, first_tours, first_costs, k_graph: int = 16, k_list: int = 5):
    """BetaTour and counting top-k lists from the first-trial ensemble on the same extended graph (runs v4/v5)."""
    succ = np.stack([T.tour_succ_pred(t)[0] for t in first_tours], axis=1).astype(np.int32)
    t0 = time.perf_counter()
    pred = M.predict_tourfold(model, xy, k_graph, device, ens_succ=succ,
                              ens_cost=np.asarray(first_costs, dtype=np.float64))
    t_model = time.perf_counter() - t0
    beta = np.asarray(pred["rank"][:, :k_list], dtype=np.int32)
    xyf = np.asarray(xy, dtype=np.float64)
    span = float((xyf.max(0) - xyf.min(0)).max()) or 1.0
    xyn = (xyf - xyf.min(0)) / span
    nb, d = M.knn_np(xyn, k_graph)
    extra = int(model.cfg.get("graph_extra", 0))
    if extra:
        nb, d, _ = M.extend_graph(xyn, nb, succ, extra)
    if not np.array_equal(nb, pred["NB"]):
        raise AssertionError("counting and model graphs differ")
    freq = T.ensemble_frequency(nb, succ)
    count = np.asarray(T.frequency_candidates(nb, d, freq, k=k_list), dtype=np.int32)
    return {"beta": beta, "count": count, "model_s": float(t_model), "model_total_s": float(pred["t_total"])}


# ------------------------------------------------------------------------------------------------ analysis
def per_thousand(count, n: int) -> float:
    return 1000.0 * float(count) / float(n)


def analyse_seed(n: int, ref_keys: np.ndarray, c0, captures, beta_keys: np.ndarray, budget_s: float,
                 fractions=FRACTIONS) -> dict:
    """Edge accounting for one seed. captures: strictly improving (elapsed_s, cost, tour)."""
    k0 = list_keys(c0)
    r0 = np.setdiff1d(ref_keys, k0, assume_unique=True)              # reference edges missing from C0
    rc = np.intersect1d(ref_keys, k0, assume_unique=True)            # reference edges present in C0
    out = {"missing_c0": int(len(r0)), "present_c0": int(len(rc)), "captures": len(captures)}
    union = np.zeros(0, dtype=np.int64)
    pos, best = 0, None
    for f in fractions:
        limit = f * budget_s
        while pos < len(captures) and captures[pos][0] <= limit + 1e-9:
            union = np.union1d(union, tour_keys(captures[pos][2]))
            best = captures[pos]
            pos += 1
        tag = f"{int(round(100 * f))}"
        reached = np.intersect1d(r0, union, assume_unique=True)
        out[f"missing_dynamic_{tag}"] = int(len(r0) - len(reached))
        out[f"recovered_{tag}"] = int(len(reached))
        if best is None:
            out[f"best_cost_{tag}"] = None
            out[f"ref_edges_in_best_{tag}"] = None
        else:
            bk = tour_keys(best[2])
            out[f"best_cost_{tag}"] = int(best[1])
            out[f"ref_edges_in_best_{tag}"] = int(len(np.intersect1d(ref_keys, bk, assume_unique=True)))
    if best is None:
        raise RuntimeError("no valid tour was captured within the budget")
    final_keys = tour_keys(best[2])
    out["r0_in_final"] = int(len(np.intersect1d(r0, final_keys, assume_unique=True)))
    out["rc_in_final"] = int(len(np.intersect1d(rc, final_keys, assume_unique=True)))
    r0_beta = np.intersect1d(r0, beta_keys, assume_unique=True)
    out["r0_in_beta"] = int(len(r0_beta))
    out["r0_beta_not_reached"] = int(len(np.setdiff1d(r0_beta, union, assume_unique=True)))
    out["r0_reached_not_beta"] = int(len(np.setdiff1d(np.intersect1d(r0, union, assume_unique=True), beta_keys,
                                                      assume_unique=True)))
    out["dynamic_set_size"] = int(len(np.union1d(k0, union)))
    out["c0_size"] = int(len(k0))
    out["final_cost"] = int(best[1])
    out["final_elapsed_s"] = float(best[0])
    out["first_capture_cost"] = int(captures[0][1])
    out["first_capture_s"] = float(captures[0][0])
    return out, union


def analyse_instance(T, inst: dict, ref_tour, first: dict, native: dict, lists: dict, budget_s: float,
                     fractions=FRACTIONS) -> tuple[dict, dict]:
    """Per-instance record (JSON) and audit arrays (NPZ)."""
    xy, ew, n = inst["xy"], inst["ew"], len(inst["xy"])
    ref_tour = np.asarray(ref_tour, dtype=np.int64)
    ref_cost = checked_cost(T, xy, ref_tour, ew)
    ref = tour_keys(ref_tour)
    beta_keys, count_keys = list_keys(lists["beta"]), list_keys(lists["count"])
    seeds = sorted(native["seeds"])
    rows, flags, audit = [], {}, {"ref_tour": ref_tour.astype(np.int32), "beta": lists["beta"],
                                  "count": lists["count"], "first_tours": first["tours"]}
    for m, seed in enumerate(seeds):
        raw = native["seeds"][seed]
        caps, seed_flags = decode_captures(T, xy, ew, raw["captures"], budget_s)
        if not caps:
            raise RuntimeError(f"seed {seed}: no valid tour within the budget")
        if not np.array_equal(raw["c0"], first["cands"][m]):
            seed_flags.append("c0_differs_from_first_trial_run")
        first_cost = int(first["costs"][m])
        if caps[0][1] < first_cost:
            # The first trial was overwritten within one polling interval: its tour (identical by seed) enters the
            # dynamic set at the time of the first capture.
            seed_flags.append("first_trial_not_captured")
            caps = [(caps[0][0], first_cost, first["tours"][m])] + caps
        elif caps[0][1] > first_cost:
            seed_flags.append("first_capture_worse_than_first_trial_run")
        row, union = analyse_seed(n, ref, raw["c0"], caps, beta_keys, budget_s, fractions)
        row["seed"] = int(seed)
        row["trace_improvements"] = len(raw["trace"])
        row["trace_last_trial"] = int(raw["trace"][-1][0]) if raw["trace"] else None
        rows.append(row)
        if seed_flags:
            flags[str(seed)] = sorted(set(seed_flags))
        audit[f"c0_seed_{seed}"] = raw["c0"]
        audit[f"final_tour_seed_{seed}"] = caps[-1][2]
        audit[f"union_keys_seed_{seed}"] = union.astype(np.int64)
    per = lambda key: float(np.mean([per_thousand(r[key], n) for r in rows]))
    summary = {"name": inst["name"], "group": inst["group"], "n": n, "ew": ew, "budget_s": budget_s,
               "reference_cost": ref_cost, "reference_source": inst.get("reference_source"),
               "missing_beta": int(len(np.setdiff1d(ref, beta_keys, assume_unique=True))),
               "missing_count": int(len(np.setdiff1d(ref, count_keys, assume_unique=True))),
               "seed_rows": rows, "flags": flags,
               "best_of_six_cost": int(min(r["final_cost"] for r in rows)),
               "first_trial_costs": [int(c) for c in first["costs"]],
               "first_trial_wall_s": float(first["wall_s"]), "native_wall_s": float(native["wall_s"]),
               "model_s": float(lists["model_s"])}
    summary["per1000"] = {
        "missing_c0": per("missing_c0"),
        "missing_beta": per_thousand(summary["missing_beta"], n),
        "missing_count": per_thousand(summary["missing_count"], n),
        **{f"missing_dynamic_{int(round(100 * f))}": per(f"missing_dynamic_{int(round(100 * f))}")
           for f in fractions},
        "r0_beta_not_reached": per("r0_beta_not_reached"),
        "r0_reached_not_beta": per("r0_reached_not_beta"),
    }
    r0 = sum(r["missing_c0"] for r in rows)
    rc = sum(r["present_c0"] for r in rows)
    summary["rates"] = {
        **{f"recovered_share_{int(round(100 * f))}":
           (sum(r[f"recovered_{int(round(100 * f))}"] for r in rows) / r0 if r0 else None) for f in fractions},
        "r0_in_final_share": sum(r["r0_in_final"] for r in rows) / r0 if r0 else None,
        "rc_in_final_share": sum(r["rc_in_final"] for r in rows) / rc if rc else None,
        "final_gap_to_reference_pct": float(np.mean([100.0 * (r["final_cost"] - ref_cost) / ref_cost
                                                      for r in rows])),
    }
    return summary, audit


def summary_tables(records: list[dict]):
    """Instance table, group table and the prespecified primary comparison."""
    import pandas as pd
    inst = pd.DataFrame([{"name": r["name"], "group": r["group"], "n": r["n"], **r["per1000"], **r["rates"]}
                         for r in records])
    order = ["uni1k", "clu1k", "uni10k", "clu10k", "tsplib"]
    cols = [c for c in inst.columns if c not in ("name", "group", "n")]
    grp = inst.groupby("group")[cols].mean().reindex([g for g in order if g in set(inst["group"])])
    grp.loc["all"] = inst[cols].mean()
    d = inst["missing_dynamic_100"] - inst["missing_beta"]
    wins, ties, losses = int((d < -1e-12).sum()), int((d.abs() <= 1e-12).sum()), int((d > 1e-12).sum())
    primary = {"comparison": "dynamic set at the budget vs BetaTour top-5 (missing reference edges per 1,000 nodes)",
               "instances": int(len(d)), "dynamic_fewer": wins, "ties": ties, "dynamic_more": losses,
               "sign_test_p": sign_test_p(wins, losses),
               "mean_dynamic_100": float(inst["missing_dynamic_100"].mean()),
               "mean_beta": float(inst["missing_beta"].mean()),
               "h1_supported": bool(wins > losses and sign_test_p(wins, losses) < 0.05)}
    return inst, grp, primary
