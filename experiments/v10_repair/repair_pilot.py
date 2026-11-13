"""Six-seed, wall-clock-charged BetaTour compressed-repair pilot runner."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from compressed_repair import repair_once
from coordinator import Snapshot, noop_actions, read_tour, run_six
from repair_engine import tour_cost


REPAIR_ARMS = ("repair_random", "repair_longest", "repair_disagreement", "repair_betatour")
LIVE_ARMS = ("sham",) + REPAIR_ARMS


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_tour(path: Path, tour: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(str(len(tour)) + "\n" + "\n".join(map(str, tour + 1)) + "\n")
    os.replace(temporary, path)


def _hash_tour(tour: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(tour, dtype=np.int32).tobytes()).hexdigest()


def _model_edge_risk(graph: np.ndarray, probability: np.ndarray):
    graph = np.asarray(graph, dtype=np.int32)
    probability = np.asarray(probability, dtype=np.float32)
    if graph.shape != probability.shape:
        raise ValueError("model graph/probability shape mismatch")

    def risk_by_position(tour: np.ndarray) -> np.ndarray:
        tour = np.asarray(tour, dtype=np.int32)
        successor = np.roll(tour, -1)
        forward = np.max(np.where(graph[tour] == successor[:, None],
                                  probability[tour], 0.0), axis=1)
        backward = np.max(np.where(graph[successor] == tour[:, None],
                                   probability[successor], 0.0), axis=1)
        return (1.0 - np.maximum(forward, backward)).astype(np.float64)

    return risk_by_position


def _valid_cost(T, xy: np.ndarray, tour: np.ndarray) -> int:
    if not T.is_valid_tour(tour, len(xy)):
        raise ValueError("invalid Hamilton tour")
    a = tour_cost(xy, tour)
    b = int(T.tour_length(xy, tour, "EUC_2D"))
    if a != b:
        raise AssertionError(f"independent EUC_2D checks disagree: {a} != {b}")
    return a


def run_repair_arm(
    kind: str,
    patched_binary: Path,
    sidecar_binary: Path,
    problem: Path,
    xy: np.ndarray,
    work_dir: Path,
    seeds: list[int],
    budget_s: float,
    model,
    device,
    T,
    M,
    *,
    attempts_per_seed: int = 8,
    cut_count: int = 3,
    window_size: int = 256,
    sidecar_timeout_s: float = 2.0,
    sidecar_trials: int = 100,
    repair_fraction_cap: float = 0.30,
    barrier_fraction: float = 0.35,
) -> dict:
    if kind not in LIVE_ARMS:
        raise ValueError(f"unknown live repair arm: {kind}")
    if len(seeds) != 6 or len(set(seeds)) != 6:
        raise ValueError("exactly six unique seeds required")
    if attempts_per_seed < 1 or cut_count < 3 or window_size < 2:
        raise ValueError("invalid repair schedule")
    if not 0 < repair_fraction_cap < 0.8 or not 0 <= barrier_fraction < 0.8 or budget_s <= 0:
        raise ValueError("invalid wall budget or repair cap")
    xy = np.asarray(xy)
    strategy = "model" if kind == "repair_betatour" else kind.removeprefix("repair_")
    outer_start = time.monotonic()
    outer_deadline = outer_start + budget_s
    audit_path = work_dir / "repair_meta.json"
    callback_meta: dict = {}

    def callback(snapshots: list[Snapshot]):
        if len(snapshots) != 6:
            raise ValueError("incomplete first-tour barrier")
        first = [np.asarray(s.first_tour, dtype=np.int32) - 1 for s in snapshots]
        first_costs = [_valid_cost(T, xy, t) for t in first]
        barrier_seen = time.monotonic() - outer_start
        successors = np.stack([T.tour_succ_pred(t)[0] for t in first], axis=1).astype(np.int32)
        prediction = M.predict_tourfold(model, xy, 16, device,
                                        ens_succ=successors,
                                        ens_cost=np.asarray(first_costs, dtype=np.float64))
        repair_stop = min(time.monotonic() + budget_s * repair_fraction_cap,
                          outer_deadline - 5.0)
        model_risk = _model_edge_risk(prediction["NB"], prediction["prob"])
        callback_meta.update({
            "barrier_costs": first_costs,
            "barrier_tour_hashes": [_hash_tour(t) for t in first],
            "barrier_wall_from_start_s": barrier_seen,
            "model_inference_s": float(prediction["t_total"]),
            "attempts": {str(seed): [] for seed in seeds},
        })
        current = [t.copy() for t in first]
        rngs = [np.random.default_rng(8901 + seed) for seed in seeds]
        repair_start = time.monotonic()
        if kind != "sham":
            # Round-robin keeps seed effort balanced if the cap is approached.
            for round_index in range(attempts_per_seed):
                for index, seed in enumerate(seeds):
                    remaining = repair_stop - time.monotonic()
                    if remaining < 0.5:
                        raise RuntimeError("repair wall cap reached before all matched attempts")
                    attempt = repair_once(
                        xy, current[index], sidecar_binary, work_dir / "sidecar_tmp",
                        strategy=strategy,
                        ensemble_tours=first,
                        model_scores=model_risk if kind == "repair_betatour" else None,
                        rng=rngs[index], cut_count=cut_count, window_size=window_size,
                        timeout_s=min(sidecar_timeout_s, max(0.1, remaining - 0.25)),
                        max_trials=sidecar_trials,
                    )
                    if attempt.after > attempt.before:
                        raise AssertionError("repair worsened a tour")
                    current[index] = attempt.tour
                    callback_meta["attempts"][str(seed)].append({
                        "round": round_index,
                        "cuts": list(attempt.cuts),
                        "before": attempt.before,
                        "after": attempt.after,
                        "reduced_n": attempt.reduced_n,
                        "free_n": attempt.free_n,
                        "elapsed_s": attempt.elapsed_s,
                        "solver_returncode": attempt.solver_returncode,
                        "solver_timed_out": attempt.solver_timed_out,
                        "improved": attempt.improved,
                        "end_wall_from_start_s": time.monotonic() - outer_start,
                    })
        callback_meta["repair_wall_s"] = time.monotonic() - repair_start
        callback_meta["repair_costs"] = [_valid_cost(T, xy, t) for t in current]
        callback_meta["completed_attempts_per_seed"] = [len(callback_meta["attempts"][str(s)]) for s in seeds]
        for seed, tour in zip(seeds, current):
            _atomic_tour(work_dir / f"seed_{seed}" / "repair.tour", tour)
        _atomic_json(audit_path, callback_meta)
        if time.monotonic() >= outer_deadline - 2.0:
            raise RuntimeError("no time remains to release LKH action barrier")
        return [noop_actions(len(xy)) for _ in seeds]

    result = run_six(patched_binary, problem, work_dir, seeds, budget_s,
                     callback, extra_params={"MAX_CANDIDATES": 6},
                     barrier_name="barrier.tour",
                     hook_after_s=budget_s * barrier_fraction)
    if not result["barrier_completed"] or not audit_path.exists():
        raise RuntimeError("repair barrier or audit incomplete")
    if any(run["applied_stats"] is None or run["best_tour"] is None for run in result["runs"]):
        raise RuntimeError("one or more LKH processes did not resume")

    lkh_costs: list[int] = []
    repair_costs: list[int] = []
    hybrid_costs: list[int] = []
    for seed in seeds:
        seed_dir = work_dir / f"seed_{seed}"
        lkh_tour = np.asarray(read_tour(seed_dir / "best.tour"), dtype=np.int32) - 1
        repair_tour = np.asarray(read_tour(seed_dir / "repair.tour"), dtype=np.int32) - 1
        lkh_cost = _valid_cost(T, xy, lkh_tour)
        repair_cost = _valid_cost(T, xy, repair_tour)
        lkh_costs.append(lkh_cost)
        repair_costs.append(repair_cost)
        hybrid_costs.append(min(lkh_cost, repair_cost))
    result.update(
        arm=kind,
        barrier_costs=callback_meta["barrier_costs"],
        barrier_tour_hashes=callback_meta["barrier_tour_hashes"],
        lkh_costs=lkh_costs,
        repair_costs=repair_costs,
        seed_costs=hybrid_costs,
        best_cost=min(hybrid_costs),
        model_inference_s=callback_meta["model_inference_s"],
        repair_wall_s=callback_meta["repair_wall_s"],
        barrier_wall_from_start_s=callback_meta["barrier_wall_from_start_s"],
        completed_attempts_per_seed=callback_meta["completed_attempts_per_seed"],
        wall_elapsed_s=result["wall_elapsed_s"],
    )
    _atomic_json(work_dir / "hybrid_result.json", {
        key: result[key] for key in (
            "arm", "barrier_costs", "barrier_tour_hashes", "lkh_costs", "repair_costs",
            "seed_costs", "best_cost", "model_inference_s", "repair_wall_s",
            "barrier_wall_from_start_s",
            "completed_attempts_per_seed", "wall_elapsed_s")
    })
    return result
