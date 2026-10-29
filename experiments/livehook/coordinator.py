"""Wall-clock coordinator for six state-preserving BetaTour/LKH processes.

Only one problem instance should use a given work directory. The action
callback receives the six first tours and native/live candidate snapshots.
It returns one per-node action list per seed; each action is
``(neighbor_id_1_based_or_0, slot_1_based_or_0, alpha)``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass
class Snapshot:
    seed: int
    first_tour: list[int]
    native_candidates: list[list[tuple[int, int]]]
    live_candidates: list[list[tuple[int, int]]]


def read_tour(path: Path) -> list[int]:
    words = [int(x) for x in path.read_text().split()]
    if not words or len(words) != words[0] + 1:
        raise ValueError(f"Malformed tour: {path}")
    n = words[0]
    tour = words[1:]
    if sorted(tour) != list(range(1, n + 1)):
        raise ValueError(f"Tour is not a permutation: {path}")
    return tour


def read_candidates(path: Path) -> list[list[tuple[int, int]]]:
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty candidate snapshot: {path}")
    n = int(lines[0])
    if len(lines) != n + 1:
        raise ValueError(f"Candidate snapshot has {len(lines)-1} rows, need {n}")
    rows: list[list[tuple[int, int]]] = []
    for expected_id, line in enumerate(lines[1:], 1):
        nums = [int(x) for x in line.split()]
        if len(nums) < 2 or nums[0] != expected_id or len(nums) != 2 + 2 * nums[1]:
            raise ValueError(f"Malformed candidate row {expected_id}: {path}")
        row = [(nums[i], nums[i + 1]) for i in range(2, len(nums), 2)]
        if any(v < 1 or v > n or v == expected_id for v, _ in row):
            raise ValueError(f"Invalid neighbor in row {expected_id}: {path}")
        rows.append(row)
    return rows


def noop_actions(n: int) -> list[tuple[int, int, int]]:
    return [(0, 0, 500)] * n


def write_actions(path: Path, actions: Sequence[tuple[int, int, int]]) -> None:
    n = len(actions)
    lines = [str(n)]
    for source_id, (neighbor, slot, alpha) in enumerate(actions, 1):
        if not (0 <= neighbor <= n and 0 <= slot <= n + 2 and 0 <= alpha < 2**31 - 1):
            raise ValueError(f"Invalid action at source {source_id}")
        if neighbor == source_id:
            raise ValueError(f"Self-edge at source {source_id}")
        lines.append(f"{source_id} {neighbor} {slot} {alpha}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, path)


def _parameter_file(path: Path, problem: Path, seed: int, wall_budget_s: float,
                    max_trials: int, extra: dict[str, str | int | float]) -> None:
    params: dict[str, str | int | float] = {
        "PROBLEM_FILE": str(problem.resolve()),
        "CANDIDATE_SET_TYPE": "POPMUSIC",
        "MAX_CANDIDATES": 5,
        "SUBGRADIENT": "NO",
        "SEED": seed,
        "RUNS": 1,
        "MAX_TRIALS": max_trials,
        # LKH's TIME_LIMIT measures process CPU time. The coordinator below
        # enforces the actual end-to-end wall budget.
        "TIME_LIMIT": max(10.0, 10.0 * wall_budget_s),
        "TRACE_LEVEL": 0,
        "STOP_AT_OPTIMUM": "NO",
    }
    params.update(extra)
    path.write_text("\n".join(f"{key} = {value}" for key, value in params.items()) + "\n")


def run_six(
    lkh_binary: Path,
    problem: Path,
    work_dir: Path,
    seeds: Sequence[int],
    wall_budget_s: float,
    action_fn: Callable[[list[Snapshot]], Sequence[Sequence[tuple[int, int, int]]]],
    *,
    max_trials: int = 100_000_000,
    extra_params: dict[str, str | int | float] | None = None,
    barrier_name: str = "first.tour",
    hook_after_s: float = 0.0,
) -> dict:
    """Run one instance with six seeds; charge startup, barrier and policy time.

    `work_dir` must be new to avoid stale action files. Runs may end before the
    deadline when `max_trials` is small. Results include all hook files so the
    caller can audit the actual candidate arrays and first tours.
    """
    if len(seeds) != 6 or len(set(seeds)) != 6:
        raise ValueError("Exactly six distinct seeds are required")
    if wall_budget_s <= 0 or max_trials < 1:
        raise ValueError("Positive budget and trials are required")
    if barrier_name not in ("first.tour", "barrier.tour") or hook_after_s < 0:
        raise ValueError("Invalid barrier marker or delay")
    if work_dir.exists():
        raise FileExistsError(f"Use a fresh run directory: {work_dir}")
    work_dir.mkdir(parents=True)
    processes: list[subprocess.Popen] = []
    open_logs: list = []
    start_mono = time.monotonic()
    start_epoch = time.time()
    deadline = start_mono + wall_budget_s
    try:
        for seed in seeds:
            seed_dir = work_dir / f"seed_{seed}"
            seed_dir.mkdir()
            params = seed_dir / "run.par"
            _parameter_file(params, problem, seed, wall_budget_s, max_trials,
                            extra_params or {})
            stdout = (seed_dir / "stdout.txt").open("w")
            stderr = (seed_dir / "stderr.txt").open("w")
            open_logs.extend((stdout, stderr))
            env = dict(os.environ)
            env["BETATOUR_HOOK_DIR"] = str(seed_dir.resolve())
            env["BETATOUR_HOOK_TIMEOUT_S"] = str(max(5.0, wall_budget_s + 5.0))
            env["BETATOUR_HOOK_AFTER_S"] = str(hook_after_s)
            processes.append(subprocess.Popen(
                [str(lkh_binary.resolve()), str(params.resolve())],
                cwd=seed_dir, stdout=stdout, stderr=stderr, env=env,
            ))
        first_paths = [work_dir / f"seed_{seed}" / barrier_name for seed in seeds]
        while not all(path.exists() for path in first_paths):
            failures = [p.returncode for p in processes if p.poll() not in (None, 0)]
            if failures:
                raise RuntimeError(f"LKH exited before first-tour barrier: {failures}")
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        snapshots: list[Snapshot] = []
        if all(path.exists() for path in first_paths) and time.monotonic() < deadline:
            for seed in seeds:
                base = work_dir / f"seed_{seed}"
                snapshots.append(Snapshot(
                    seed=seed,
                    first_tour=read_tour(base / barrier_name),
                    native_candidates=read_candidates(base / "native.candidates"),
                    live_candidates=read_candidates(base / "live.candidates"),
                ))
            actions = action_fn(snapshots)
            if len(actions) != len(seeds):
                raise ValueError("Policy must return one action matrix per seed")
            if time.monotonic() < deadline:
                for seed, snapshot, action in zip(seeds, snapshots, actions):
                    if len(action) != len(snapshot.first_tour):
                        raise ValueError(f"Policy dimension mismatch for seed {seed}")
                    write_actions(work_dir / f"seed_{seed}" / "action.txt", action)
        while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
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
        for stream in open_logs:
            stream.close()
    result = {
        "seeds": list(seeds),
        "wall_budget_s": wall_budget_s,
        "wall_elapsed_s": time.monotonic() - start_mono,
        "start_epoch_s": start_epoch,
        "deadline_epoch_s": start_epoch + wall_budget_s,
        "barrier_completed": len(snapshots) == 6,
        "runs": [],
    }
    for seed, process in zip(seeds, processes):
        base = work_dir / f"seed_{seed}"
        best = base / "best.tour"
        result["runs"].append({
            "seed": seed,
            "returncode": process.returncode,
            "first_tour": str(base / barrier_name) if (base / barrier_name).exists() else None,
            "best_tour": str(best) if best.exists() else None,
            "best_events": str(base / "best.events") if (base / "best.events").exists() else None,
            "native_candidates": str(base / "native.candidates") if (base / "native.candidates").exists() else None,
            "live_candidates": str(base / "live.candidates") if (base / "live.candidates").exists() else None,
            "post_candidates": str(base / "post.candidates") if (base / "post.candidates").exists() else None,
            "applied_stats": str(base / "applied.stats") if (base / "applied.stats").exists() else None,
        })
    (work_dir / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
