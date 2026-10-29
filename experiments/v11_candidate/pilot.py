"""Wall-clock v11 POP6-residual candidate pilot helpers.

All guided arms start from six fresh POPMUSIC-6 LKH processes and pause at the
same completed first trial. The unchanged processes resume after their one
candidate action per selected source. Source/candidate snapshots are hashed so
the notebook can reject a comparison when the starting search states differ.
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

from coordinator import read_candidates, read_tour, run_six
from policy import choose_actions


GUIDED_ARMS = ("sham", "beta_add", "frequency_add", "random_add")
NATIVE_ARMS = ("native_pop6", "native_pop7")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_tour_hash(tour: np.ndarray) -> str:
    t = np.asarray(tour, dtype=np.int32)
    nxt = np.roll(t, -1)
    edges = np.column_stack((np.minimum(t, nxt), np.maximum(t, nxt)))
    edges = np.ascontiguousarray(edges[np.lexsort((edges[:, 1], edges[:, 0]))])
    return hashlib.sha256(edges.tobytes()).hexdigest()


def candidate_hash(rows: list[list[tuple[int, int]]]) -> str:
    """Hash full directed candidate contents, including alpha and row order."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(len(row).to_bytes(4, "little"))
        for neighbor, alpha in row:
            digest.update(int(neighbor).to_bytes(4, "little", signed=False))
            digest.update(int(alpha).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def _valid_cost(T, xy: np.ndarray, tour: np.ndarray) -> int:
    tour = np.asarray(tour, dtype=np.int32)
    if not T.is_valid_tour(tour, len(xy)):
        raise ValueError("invalid Hamilton tour")
    library_cost = int(T.tour_length(xy, tour, "EUC_2D"))
    a = np.asarray(xy[tour], dtype=np.float64)
    b = np.asarray(xy[np.roll(tour, -1)], dtype=np.float64)
    independent = int(np.floor(np.sqrt(np.sum((a - b) ** 2, axis=1)) + 0.5)
                      .sum(dtype=np.int64))
    if independent != library_cost:
        raise AssertionError(f"independent EUC_2D costs disagree: {independent} != {library_cost}")
    return library_cost


def _load_native_tour(T, path: Path) -> np.ndarray:
    first = path.read_text().split(maxsplit=1)[0]
    if first.isdigit():
        return np.asarray(read_tour(path), dtype=np.int32) - 1
    return np.asarray(T.read_tour(path), dtype=np.int32)


def _verify_applied_actions(work_dir: Path, seeds: Sequence[int],
                            actions: list[list[tuple[int, int, int]]]) -> None:
    """Check the actual LKH candidate rows, not just the callback's proposal."""
    for seed, matrix in zip(seeds, actions):
        base = work_dir / f"seed_{seed}"
        live = read_candidates(base / "live.candidates")
        post = read_candidates(base / "post.candidates")
        if len(live) != len(matrix) or len(post) != len(matrix):
            raise AssertionError(f"candidate dimension changed for seed {seed}")
        for source, ((neighbor, slot, alpha), before, after) in enumerate(
                zip(matrix, live, post), 1):
            expected = list(before)
            if neighbor:
                if any(to == neighbor for to, _ in before):
                    raise AssertionError(f"action was already in source row {source}")
                expected.insert(min(slot - 1, len(expected)), (neighbor, alpha))
            if after != expected:
                raise AssertionError(f"unexpected post-action candidate row {seed}/{source}")
        stats = dict(line.split() for line in (base / "applied.stats").read_text().splitlines()
                     if len(line.split()) == 2)
        selected = sum(neighbor != 0 for neighbor, _, _ in matrix)
        if (int(stats["added"]) != selected or int(stats["existing"]) != 0 or
                int(stats["rejected"]) != 0):
            raise AssertionError(f"LKH did not apply exactly {selected} novel actions for seed {seed}")


def run_guided_arm(
    kind: str,
    patched_binary: Path,
    problem: Path,
    xy: np.ndarray,
    work_dir: Path,
    seeds: Sequence[int],
    budget_s: float,
    model,
    device,
    T,
    M,
    *,
    fraction: float = 0.25,
    graph_k: int = 16,
) -> dict:
    if kind not in GUIDED_ARMS or len(seeds) != 6 or len(set(seeds)) != 6:
        raise ValueError("invalid guided arm or seed panel")
    xy = np.asarray(xy)
    callback_meta: dict = {}
    proposed_actions: list[list[tuple[int, int, int]]] = []

    def callback(snapshots):
        if len(snapshots) != 6:
            raise ValueError("incomplete first-tour barrier")
        first_tours = [np.asarray(s.first_tour, dtype=np.int32) - 1 for s in snapshots]
        first_costs = [_valid_cost(T, xy, tour) for tour in first_tours]
        first_hashes = [canonical_tour_hash(tour) for tour in first_tours]
        live_hashes = [candidate_hash(s.live_candidates) for s in snapshots]
        native_hashes = [candidate_hash(s.native_candidates) for s in snapshots]
        actions, audit = choose_actions(kind, snapshots, xy, model, device, T, M,
                                        fraction=fraction, graph_k=graph_k)
        if len(actions) != len(seeds):
            raise ValueError("policy returned wrong seed count")
        proposed_actions[:] = actions
        callback_meta.update({"arm": kind, "first_costs": first_costs,
                              "first_tour_hashes": first_hashes,
                              "live_candidate_hashes": live_hashes,
                              "native_candidate_hashes": native_hashes,
                              "policy_audit": audit})
        _atomic_json(work_dir / "policy_meta.json", callback_meta)
        return actions

    result = run_six(Path(patched_binary), Path(problem), Path(work_dir),
                     seeds, budget_s, callback,
                     extra_params={"MAX_CANDIDATES": 6})
    if not result["barrier_completed"]:
        raise RuntimeError("first-trial barrier incomplete")
    if any(run["best_tour"] is None or run["applied_stats"] is None for run in result["runs"]):
        raise RuntimeError("one or more LKH processes did not resume")
    _verify_applied_actions(work_dir, seeds, proposed_actions)
    seed_costs = []
    improvement_events: dict[str, list[dict]] = {}
    for seed in seeds:
        base = work_dir / f"seed_{seed}"
        tour = np.asarray(read_tour(base / "best.tour"), dtype=np.int32) - 1
        final_cost = _valid_cost(T, xy, tour)
        seed_costs.append(final_cost)
        event_file = base / "best.events"
        if not event_file.exists():
            raise RuntimeError(f"missing patched best.events for seed {seed}")
        events = []
        for line in event_file.read_text().splitlines():
            words = line.split()
            if len(words) != 2:
                raise ValueError(f"malformed best event: {event_file}")
            epoch, cost = float(words[0]), int(words[1])
            if not events or cost < events[-1]["cost"]:
                events.append({"elapsed_s": max(0.0, epoch - result["start_epoch_s"]),
                               "cost": cost, "source": "patched_exact_hook"})
        if not events:
            raise RuntimeError(f"empty patched best.events for seed {seed}")
        if events[-1]["cost"] != final_cost:
            raise AssertionError(f"patched tour/event costs disagree for seed {seed}")
        improvement_events[str(seed)] = events
    result.update(arm=kind, seed_costs=seed_costs,
                  best_cost=min(seed_costs),
                  first_costs=callback_meta["first_costs"],
                  first_tour_hashes=callback_meta["first_tour_hashes"],
                  live_candidate_hashes=callback_meta["live_candidate_hashes"],
                  native_candidate_hashes=callback_meta["native_candidate_hashes"],
                  policy_audit=callback_meta["policy_audit"],
                  guided_improvement_events=improvement_events,
                  binary_sha256=sha256_file(patched_binary))
    _atomic_json(work_dir / "endpoint.json", {
        key: result[key] for key in ("arm", "seed_costs", "best_cost",
                                       "first_costs", "first_tour_hashes",
                                       "live_candidate_hashes", "native_candidate_hashes",
                                       "policy_audit", "guided_improvement_events",
                                       "binary_sha256")
    })
    return result


def run_native_arm(
    kind: str,
    native_binary: Path,
    problem: Path,
    xy: np.ndarray,
    work_dir: Path,
    seeds: Sequence[int],
    budget_s: float,
    T,
) -> dict:
    if kind not in NATIVE_ARMS or len(seeds) != 6 or len(set(seeds)) != 6 or work_dir.exists():
        raise ValueError("invalid native arm, seed panel, or work directory")
    k = 6 if kind == "native_pop6" else 7
    work_dir.mkdir(parents=True)
    processes: list[subprocess.Popen] = []
    streams = []
    start = time.monotonic()
    start_epoch = time.time()
    deadline = start + budget_s
    seen_mtime_ns: dict[int, int] = {}
    events: dict[str, list[dict]] = {str(seed): [] for seed in seeds}

    def observe_native_tours() -> None:
        # Native LKH rewrites OUTPUT_TOUR_FILE whenever it finds a new best.
        # The filesystem poll timestamps are approximate and may miss rapid
        # overwritten incumbents; they are kept separate from exact hook logs.
        for seed in seeds:
            path = work_dir / f"seed_{seed}" / "best.tour"
            try:
                stamp = path.stat()
            except FileNotFoundError:
                continue
            if stamp.st_mtime_ns == seen_mtime_ns.get(seed):
                continue
            try:
                cost = _valid_cost(T, xy, _load_native_tour(T, path))
            except (OSError, ValueError, AssertionError, IndexError):
                # LKH writes this file in place. Retry at the next poll if a
                # concurrent read saw an incomplete TSPLIB tour.
                continue
            seen_mtime_ns[seed] = stamp.st_mtime_ns
            prior = events[str(seed)]
            if not prior or cost < prior[-1]["cost"]:
                prior.append({"elapsed_s": max(0.0, min(float(budget_s),
                                           stamp.st_mtime - start_epoch)),
                              "cost": cost, "source": "output_tour_mtime_poll"})
    try:
        for seed in seeds:
            base = work_dir / f"seed_{seed}"
            base.mkdir()
            par = base / "run.par"
            params = {"PROBLEM_FILE": str(problem.resolve()),
                      "CANDIDATE_SET_TYPE": "POPMUSIC", "MAX_CANDIDATES": k,
                      "SUBGRADIENT": "NO", "SEED": seed, "RUNS": 1,
                      "MAX_TRIALS": 100_000_000,
                      "TIME_LIMIT": max(10.0, 10.0 * budget_s),
                      "TRACE_LEVEL": 0, "STOP_AT_OPTIMUM": "NO",
                      "OUTPUT_TOUR_FILE": str(base / "best.tour")}
            T.write_par(par, params)
            stdout = (base / "stdout.txt").open("w")
            stderr = (base / "stderr.txt").open("w")
            streams.extend((stdout, stderr))
            env = dict(os.environ)
            env.pop("BETATOUR_HOOK_DIR", None)
            processes.append(subprocess.Popen([str(Path(native_binary).resolve()),
                                                 str(par.resolve())], cwd=base,
                                                stdout=stdout, stderr=stderr, env=env))
        while any(process.poll() is None for process in processes) and time.monotonic() < deadline:
            observe_native_tours()
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
    observe_native_tours()
    seed_costs = []
    for seed in seeds:
        tour_path = work_dir / f"seed_{seed}" / "best.tour"
        if not tour_path.exists():
            raise RuntimeError(f"native POP{k} yielded no tour for seed {seed}")
        seed_costs.append(_valid_cost(T, xy, _load_native_tour(T, tour_path)))
    result = {"arm": kind, "seeds": list(seeds), "wall_budget_s": budget_s,
              "wall_elapsed_s": time.monotonic() - start,
              "start_epoch_s": start_epoch,
              "seed_costs": seed_costs, "best_cost": min(seed_costs),
              "native_improvement_events": events,
              "binary_sha256": sha256_file(native_binary)}
    _atomic_json(work_dir / "endpoint.json", result)
    return result
