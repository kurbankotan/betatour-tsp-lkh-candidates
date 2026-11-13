"""BetaTour v9 live-LKH pilot helpers.

This module defines interventions and validates their actual tour endpoints.
The frozen TourFold checkpoint supplies reference-edge scores; whether these
interventions improve LKH is an empirical question for the pilot.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from adaptive_policy import SparseAddConfig, propose_sparse_add, rerank_scored_pop
from coordinator import Snapshot, noop_actions, read_tour, run_six


LIVE_ARMS = ("sham", "beta_promote", "beta_sparse", "frequency_same_gate")
NATIVE_ARMS = ("native_pop5", "native_pop6")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_array(snapshots: Sequence[Snapshot], field: str) -> np.ndarray:
    rows = [getattr(snapshot, field) for snapshot in snapshots]
    n = len(rows[0])
    if any(len(item) != n for item in rows):
        raise ValueError("Candidate snapshots have different dimensions")
    width = max(len(row) for seed_rows in rows for row in seed_rows)
    if width < 1 or width >= n:
        raise ValueError(f"Invalid candidate width {width}")
    out = np.full((len(rows), n, width), -1, dtype=np.int32)
    for m, seed_rows in enumerate(rows):
        for i, row in enumerate(seed_rows):
            if row:
                out[m, i, :len(row)] = [to - 1 for to, _ in row]
    return out


def first_tour_arrays(snapshots: Sequence[Snapshot], xy: np.ndarray, T) -> tuple[np.ndarray, np.ndarray]:
    successors, costs = [], []
    for snapshot in snapshots:
        tour = np.asarray(snapshot.first_tour, dtype=np.int32) - 1
        if not T.is_valid_tour(tour, len(xy)):
            raise ValueError(f"Invalid first tour from seed {snapshot.seed}")
        successors.append(T.tour_succ_pred(tour)[0])
        costs.append(T.tour_length(xy, tour, "EUC_2D"))
    return np.stack(successors, axis=1).astype(np.int32), np.asarray(costs, dtype=np.float64)


def _save_npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _actions_from_destinations(destinations: np.ndarray, slot: int, alpha: int) -> list[list[tuple[int, int, int]]]:
    return [[(int(to) + 1, slot, alpha) if to >= 0 else (0, 0, 500)
             for to in row] for row in destinations]


def make_action_callback(kind: str, xy: np.ndarray, model, device, T, M,
                         audit_dir: Path, graph_k: int = 16,
                         sparse_fraction: float = 0.25):
    if kind not in LIVE_ARMS:
        raise ValueError(f"Unknown live arm {kind}")
    audit_dir.mkdir(parents=True, exist_ok=True)

    def callback(snapshots: list[Snapshot]) -> list[list[tuple[int, int, int]]]:
        if len(snapshots) != 6:
            raise ValueError("TourFold requires six first tours")
        succ, costs = first_tour_arrays(snapshots, xy, T)
        live = candidate_array(snapshots, "live_candidates")
        native = candidate_array(snapshots, "native_candidates")
        prediction = M.predict_tourfold(model, xy, graph_k, device,
                                        ens_succ=succ, ens_cost=costs)
        graph = np.asarray(prediction["NB"], dtype=np.int32)
        prob = np.asarray(prediction["prob"], dtype=np.float32)
        if kind == "sham":
            destinations = np.full((6, len(xy)), -1, dtype=np.int32)
            actions = [noop_actions(len(xy)) for _ in range(6)]
        elif kind == "beta_promote":
            reordered = rerank_scored_pop(live, graph, prob)
            destinations = np.where(reordered[:, :, 0] != live[:, :, 0],
                                    reordered[:, :, 0], -1).astype(np.int32)
            actions = _actions_from_destinations(destinations, slot=1, alpha=0)
        else:
            sparse = propose_sparse_add(
                live, graph, prob,
                SparseAddConfig(max_fraction=sparse_fraction,
                                min_scored_pop=2, min_margin=0.0),
            )
            destinations = sparse.destination.copy()
            if kind == "frequency_same_gate":
                # Keep Beta's node-level gate and intervention count, while
                # replacing its edge choice with ensemble frequency.
                frequency = T.ensemble_frequency(graph, succ)
                frequency_rank = T.frequency_candidates(
                    graph, prediction["D"], frequency, k=graph.shape[1])
                n = len(xy)
                for seed in range(6):
                    existing = {
                        (min(i, int(v)), max(i, int(v)))
                        for i, row in enumerate(live[seed]) for v in row if v >= 0
                    }
                    for i in np.flatnonzero(destinations[seed] >= 0):
                        choice = -1
                        for v in frequency_rank[i]:
                            v = int(v)
                            if (min(i, v), max(i, v)) not in existing:
                                choice = v
                                break
                        destinations[seed, i] = choice
            # Slot 3 tests whether a sparse proposal can be searched early
            # while retaining every native candidate edge in the live set.
            actions = _actions_from_destinations(destinations, slot=3, alpha=500)
        _save_npz_atomic(audit_dir / "policy_arrays.npz", first_successors=succ,
                         first_costs=costs, native_candidates=native,
                         live_candidates=live, model_graph=graph,
                         model_probability=prob, chosen_destinations=destinations)
        (audit_dir / "policy_meta.json").write_text(json.dumps({
            "arm": kind,
            "actions_per_seed": [int(np.sum(row >= 0)) for row in destinations],
            "model_inference_s": float(prediction["t_total"]),
            "sparse_fraction": sparse_fraction if kind in ("beta_sparse", "frequency_same_gate") else None,
        }, indent=2) + "\n")
        return actions

    return callback


def _read_best(path: Path, xy: np.ndarray, T) -> int:
    # The hook writes a compact atomic integer list; native LKH writes a
    # TSPLIB TOUR_SECTION file.  Recompute costs from either exact tour.
    first_token = path.read_text().split(maxsplit=1)[0]
    if first_token.isdigit():
        tour = np.asarray(read_tour(path), dtype=np.int32) - 1
    else:
        tour = T.read_tour(path).astype(np.int32)
    if not T.is_valid_tour(tour, len(xy)):
        raise ValueError(f"Invalid best tour {path}")
    return int(T.tour_length(xy, tour, "EUC_2D"))


def run_live_arm(kind: str, patched_binary: Path, problem: Path, xy: np.ndarray,
                 work_dir: Path, seeds: Sequence[int], budget_s: float,
                 model, device, T, M, *, max_trials: int = 100_000_000,
                 sparse_fraction: float = 0.25) -> dict:
    callback = make_action_callback(kind, xy, model, device, T, M,
                                    work_dir.parent / (work_dir.name + "_audit"),
                                    sparse_fraction=sparse_fraction)
    result = run_six(patched_binary, problem, work_dir, seeds, budget_s,
                     callback, max_trials=max_trials)
    costs = []
    for run in result["runs"]:
        if run["best_tour"] is None:
            raise RuntimeError(f"No best tour in {kind}, seed {run['seed']}")
        costs.append(_read_best(Path(run["best_tour"]), xy, T))
    if not result["barrier_completed"]:
        raise RuntimeError(f"First-tour barrier incomplete in {kind}")
    if any(run["applied_stats"] is None for run in result["runs"]):
        raise RuntimeError(f"One or more hook actions were not applied in {kind}")
    result.update(arm=kind, seed_costs=costs, best_cost=min(costs),
                  policy_meta=str(work_dir.parent / (work_dir.name + "_audit")),
                  binary_sha256=sha256_file(patched_binary))
    return result


def run_native_arm(kind: str, native_binary: Path, problem: Path, xy: np.ndarray,
                   work_dir: Path, seeds: Sequence[int], budget_s: float, T,
                   *, max_trials: int = 100_000_000) -> dict:
    if kind not in NATIVE_ARMS or len(seeds) != 6 or work_dir.exists():
        raise ValueError("Invalid native arm, seed panel, or work directory")
    work_dir.mkdir(parents=True)
    k = 5 if kind == "native_pop5" else 6
    processes, streams = [], []
    start = time.monotonic()
    deadline = start + budget_s
    try:
        for seed in seeds:
            base = work_dir / f"seed_{seed}"
            base.mkdir()
            params = {
                "PROBLEM_FILE": str(problem.resolve()),
                "CANDIDATE_SET_TYPE": "POPMUSIC", "SUBGRADIENT": "NO",
                "MAX_CANDIDATES": k, "SEED": seed, "RUNS": 1,
                "MAX_TRIALS": max_trials, "TIME_LIMIT": max(10.0, 10.0 * budget_s),
                "TRACE_LEVEL": 0, "STOP_AT_OPTIMUM": "NO",
                "OUTPUT_TOUR_FILE": str(base / "best.tour"),
            }
            par = base / "run.par"
            T.write_par(par, params)
            stdout = (base / "stdout.txt").open("w")
            stderr = (base / "stderr.txt").open("w")
            streams.extend((stdout, stderr))
            env = dict(os.environ)
            env.pop("BETATOUR_HOOK_DIR", None)
            processes.append(subprocess.Popen(
                [str(native_binary.resolve()), str(par.resolve())], cwd=base,
                stdout=stdout, stderr=stderr, env=env))
        while any(process.poll() is None for process in processes) and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in streams:
            stream.close()
    costs = []
    runs = []
    for seed, process in zip(seeds, processes):
        best = work_dir / f"seed_{seed}" / "best.tour"
        if not best.exists():
            raise RuntimeError(f"No best tour in {kind}, seed {seed}")
        cost = _read_best(best, xy, T)
        costs.append(cost)
        runs.append({"seed": seed, "best_tour": str(best),
                     "returncode": process.returncode, "cost": cost})
    result = {"arm": kind, "seeds": list(seeds), "wall_budget_s": budget_s,
              "wall_elapsed_s": time.monotonic() - start, "seed_costs": costs,
              "best_cost": min(costs), "runs": runs,
              "binary_sha256": sha256_file(native_binary)}
    (work_dir / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
