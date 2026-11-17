"""Wall-clock coordinator for six state-preserving BetaTour/LKH swap processes.

Only one problem instance should use a given work directory. The action
callback receives the six first tours and native/live candidate snapshots.
It returns one per-node action list per seed; each action is
``(neighbor_id_1_based_or_0, victim_id_1_based_or_0, alpha)``.
"""

from __future__ import annotations

import json
import hashlib
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
    for source_id, (neighbor, victim, alpha) in enumerate(actions, 1):
        if not (0 <= neighbor <= n and 0 <= victim <= n and 0 <= alpha < 2**31 - 1):
            raise ValueError(f"Invalid action at source {source_id}")
        if neighbor == source_id or victim == source_id or bool(neighbor) != bool(victim):
            raise ValueError(f"Self-edge at source {source_id}")
        lines.append(f"{source_id} {neighbor} {victim} {alpha}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, path)


def write_protected_edges(path: Path, tours: Sequence[Sequence[int]]) -> None:
    """Give each hook the union of all six first-tour edges before readiness.

    The hook refuses to evict these edges.  This is a safety check independent
    of the policy's own victim selection.
    """
    if len(tours) != 6 or not tours:
        raise ValueError("Six first tours are required")
    n = len(tours[0])
    edges: set[tuple[int, int]] = set()
    for tour in tours:
        if len(tour) != n or sorted(tour) != list(range(1, n + 1)):
            raise ValueError("Invalid first tour for protection")
        for a, b in zip(tour, list(tour[1:]) + [tour[0]]):
            edges.add((min(a, b), max(a, b)))
    lines = [f"{n} {len(edges)}"] + [f"{a} {b}" for a, b in sorted(edges)]
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
    snapshots: list[Snapshot] = []
    actions_published_seeds: list[int] = []
    phase_timings_s: dict[str, float] = {}
    remaining_at_policy_end_s: float | None = None
    failure_reason: str | None = None
    caught_exception: str | None = None
    phase = "startup"
    start_mono = time.monotonic()
    start_epoch = time.time()
    deadline = start_mono + wall_budget_s
    # Leave time for the six LKH processes to parse actions and export their
    # post-swap rows. A tiny test budget gets a proportionally tiny reserve.
    resume_reserve_s = min(10.0, max(0.1, 0.05 * wall_budget_s))

    def mark(name: str) -> None:
        phase_timings_s[name] = max(0.0, time.monotonic() - start_mono)

    try:
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
            phase = "barrier"
            first_paths = [work_dir / f"seed_{seed}" / barrier_name for seed in seeds]
            while not all(path.exists() for path in first_paths):
                failures = [p.returncode for p in processes if p.poll() not in (None, 0)]
                if failures:
                    failure_reason = "hook_process_failed_before_barrier"
                    raise RuntimeError(f"LKH exited before first-tour barrier: {failures}")
                if time.monotonic() >= deadline:
                    failure_reason = "deadline_before_barrier"
                    break
                time.sleep(0.02)
            if failure_reason is None and all(path.exists() for path in first_paths):
                mark("barrier_ready")
                phase = "snapshot_read"
                for seed in seeds:
                    base = work_dir / f"seed_{seed}"
                    snapshots.append(Snapshot(
                        seed=seed,
                        first_tour=read_tour(base / barrier_name),
                        native_candidates=read_candidates(base / "native.candidates"),
                        live_candidates=read_candidates(base / "live.candidates"),
                    ))
                mark("snapshots_ready")
                if deadline - time.monotonic() <= resume_reserve_s:
                    failure_reason = "deadline_before_policy"
                else:
                    phase = "policy"
                    mark("policy_start")
                    actions = action_fn(snapshots)
                    mark("policy_end")
                    remaining_at_policy_end_s = deadline - time.monotonic()
                    if len(actions) != len(seeds):
                        raise ValueError("Policy must return one action matrix per seed")
                    if remaining_at_policy_end_s <= resume_reserve_s:
                        failure_reason = "deadline_before_actions"
                    else:
                        # Validate every matrix before releasing any process.
                        for seed, snapshot, action in zip(seeds, snapshots, actions):
                            if len(action) != len(snapshot.first_tour):
                                raise ValueError(f"Policy dimension mismatch for seed {seed}")
                        phase = "action_publication"
                        mark("actions_publish_start")
                        # action.txt is the readiness marker: all protected
                        # files must be committed before the first action.
                        tours = [snapshot.first_tour for snapshot in snapshots]
                        for seed in seeds:
                            write_protected_edges(
                                work_dir / f"seed_{seed}" / "protected.edges", tours)
                        if deadline - time.monotonic() <= resume_reserve_s / 2:
                            failure_reason = "deadline_during_action_publication"
                        else:
                            for seed, action in zip(seeds, actions):
                                if deadline - time.monotonic() <= resume_reserve_s / 2:
                                    failure_reason = "deadline_during_action_publication"
                                    break
                                write_actions(work_dir / f"seed_{seed}" / "action.txt", action)
                                actions_published_seeds.append(int(seed))
                        mark("actions_publish_end")
                        if failure_reason is None:
                            phase = "search_after_action"
            while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
                if (failure_reason is None and actions_published_seeds and
                        any(p.poll() not in (None, 0) for p in processes)):
                    failure_reason = "hook_process_failed"
                    break
                time.sleep(0.02)
        except Exception as exc:
            caught_exception = repr(exc)
            if failure_reason is None:
                failure_reason = f"{phase}_exception"
            raise
        finally:
            mark("process_stop_start")
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
            mark("process_stop_end")
    finally:
        # Persist diagnostics even if an exception interrupted the callback
        # or action publication. The caller can copy this directory to Drive.
        resume_completed_seeds = [int(seed) for seed in seeds
                                  if (work_dir / f"seed_{seed}" / "applied.stats").exists()]
        if failure_reason is None and len(snapshots) != len(seeds):
            failure_reason = "deadline_before_barrier"
        if failure_reason is None and len(actions_published_seeds) != len(seeds):
            failure_reason = "deadline_before_actions"
        if failure_reason is None and len(resume_completed_seeds) != len(seeds):
            failure_reason = "actions_not_applied_before_deadline"
        result = {
            "seeds": list(seeds),
            "wall_budget_s": wall_budget_s,
            "wall_elapsed_s": time.monotonic() - start_mono,
            "start_epoch_s": start_epoch,
            "deadline_epoch_s": start_epoch + wall_budget_s,
            "barrier_completed": len(snapshots) == 6,
            "phase": phase,
            "phase_timings_s": phase_timings_s,
            "remaining_at_policy_end_s": remaining_at_policy_end_s,
            "resume_reserve_s": resume_reserve_s,
            "failure_reason": failure_reason,
            "exception": caught_exception,
            "actions_published_seeds": actions_published_seeds,
            "resume_completed_seeds": resume_completed_seeds,
            "runs": [],
        }
        for index, seed in enumerate(seeds):
            process = processes[index] if index < len(processes) else None
            base = work_dir / f"seed_{seed}"
            best = base / "best.tour"
            stderr_path = base / "stderr.txt"
            result["runs"].append({
                "seed": seed,
                "returncode": process.returncode if process is not None else None,
                "first_tour": str(base / barrier_name) if (base / barrier_name).exists() else None,
                "best_tour": str(best) if best.exists() else None,
                "best_events": str(base / "best.events") if (base / "best.events").exists() else None,
                "native_candidates": str(base / "native.candidates") if (base / "native.candidates").exists() else None,
                "live_candidates": str(base / "live.candidates") if (base / "live.candidates").exists() else None,
                "post_candidates": str(base / "post.candidates") if (base / "post.candidates").exists() else None,
                "action_exists": (base / "action.txt").exists(),
                "post_exists": (base / "post.candidates").exists(),
                "applied_exists": (base / "applied.stats").exists(),
                "stderr_tail": stderr_path.read_text(errors="replace")[-1000:]
                if stderr_path.exists() else "",
                "protected_edges_sha256": hashlib.sha256((base / "protected.edges").read_bytes()).hexdigest()
                if (base / "protected.edges").exists() else None,
                "applied_stats": str(base / "applied.stats") if (base / "applied.stats").exists() else None,
            })
        (work_dir / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
