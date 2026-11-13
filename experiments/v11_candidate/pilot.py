"""Wall-clock v11 POP6-residual candidate pilot helpers.

All guided arms start from six fresh POPMUSIC-6 LKH processes and pause at the
same completed first trial. The unchanged processes resume after their one
candidate action per selected source. Source/candidate snapshots are hashed so
the notebook can reject a comparison when the starting search states differ.

Only the policy decision runs inside an arm's deadline. Snapshot hashing,
independent cost checks, audit files and decision-cache writes happen after
the six processes have stopped, from files that the hook writes once.
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
from policy import DecisionCache, _hash_ints, choose_actions


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


def parse_tsplib_tour(data: bytes) -> np.ndarray:
    """0-based tour from an LKH TSPLIB tour file; raises if it is incomplete."""
    _header, marker, body = data.decode("ascii").partition("TOUR_SECTION")
    if not marker:
        raise ValueError("tour file has no TOUR_SECTION")
    nodes = []
    for token in body.split():
        if token == "-1":
            return np.asarray(nodes, dtype=np.int32) - 1
        nodes.append(int(token))
    raise ValueError("tour section is not terminated by -1")


def _complete_tour_file(data: bytes) -> bool:
    tail = data.rstrip()
    return b"TOUR_SECTION" in data and (tail.endswith(b"EOF") or tail.endswith(b"-1"))


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


def _patched_events(base: Path, start_epoch: float, final_cost: int) -> tuple[list[dict], list[str]]:
    """Exact hook events. A deadline kill can only cut the last append."""
    text = (base / "best.events").read_text()
    lines = text.splitlines()
    events, flags = [], []
    if lines and not text.endswith("\n"):
        # The hook ends every event with a newline; a missing one marks an
        # append cut by the deadline kill, whose digits may be incomplete.
        lines.pop()
        flags.append("partial_final_event_line_dropped")
    for line in lines:
        words = line.split()
        try:
            if len(words) != 2:
                raise ValueError(line)
            epoch, cost = float(words[0]), int(words[1])
        except ValueError:
            raise ValueError(f"malformed best event in {base}: {line!r}")
        if not events or cost < events[-1]["cost"]:
            events.append({"elapsed_s": max(0.0, epoch - start_epoch),
                           "cost": cost, "source": "patched_exact_hook"})
    if not events:
        raise RuntimeError(f"empty patched best.events in {base}")
    if events[-1]["cost"] != final_cost:
        if final_cost > events[-1]["cost"]:
            raise AssertionError(f"patched tour/event costs disagree in {base}")
        # best.tour is renamed into place before its event is appended.
        events.append({"elapsed_s": max(0.0, (base / "best.tour").stat().st_mtime - start_epoch),
                       "cost": final_cost, "source": "patched_tour_mtime_recovery"})
        flags.append("final_event_recovered_from_tour_mtime")
    return events, flags


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
    decision_cache_dir: Path | None = None,
    model_identity: str = "",
) -> dict:
    if kind not in GUIDED_ARMS or len(seeds) != 6 or len(set(seeds)) != 6:
        raise ValueError("invalid guided arm or seed panel")
    xy = np.asarray(xy)
    cache = DecisionCache(decision_cache_dir)  # read before the timed window
    decision: dict = {}

    def callback(snapshots):
        if len(snapshots) != 6:
            raise ValueError("incomplete first-tour barrier")
        actions, audit = choose_actions(kind, snapshots, xy, model, device, T, M,
                                        fraction=fraction, graph_k=graph_k,
                                        decision_cache=cache,
                                        model_identity=model_identity)
        if len(actions) != len(seeds):
            raise ValueError("policy returned wrong seed count")
        decision.update(actions=actions, audit=audit)
        return actions

    result = run_six(Path(patched_binary), Path(problem), Path(work_dir),
                     seeds, budget_s, callback,
                     extra_params={"MAX_CANDIDATES": 6})
    cache_files = cache.flush()  # written after the timed window
    if not result["barrier_completed"] or "actions" not in decision:
        raise RuntimeError("first-trial barrier incomplete")
    if any(run["best_tour"] is None or run["applied_stats"] is None for run in result["runs"]):
        raise RuntimeError("one or more LKH processes did not resume")
    actions, audit = decision["actions"], decision["audit"]

    first_costs, first_hashes, raw_hashes, live_hashes, native_hashes = [], [], [], [], []
    for seed in seeds:
        base = work_dir / f"seed_{seed}"
        first = np.asarray(read_tour(base / "first.tour"), dtype=np.int32) - 1
        first_costs.append(_valid_cost(T, xy, first))
        first_hashes.append(canonical_tour_hash(first))
        raw_hashes.append(_hash_ints(first))
        live_hashes.append(candidate_hash(read_candidates(base / "live.candidates")))
        native_hashes.append(candidate_hash(read_candidates(base / "native.candidates")))
    if raw_hashes != audit["first_tour_raw_sha256"]:
        raise AssertionError("barrier tour files changed after the policy decision")
    _atomic_json(work_dir / "policy_meta.json", {
        "arm": kind, "first_costs": first_costs, "first_tour_hashes": first_hashes,
        "live_candidate_hashes": live_hashes, "native_candidate_hashes": native_hashes,
        "decision_cache_files_written": cache_files, "policy_audit": audit})
    _verify_applied_actions(work_dir, seeds, actions)

    seed_costs, improvement_events, event_flags = [], {}, {}
    for seed in seeds:
        base = work_dir / f"seed_{seed}"
        tour = np.asarray(read_tour(base / "best.tour"), dtype=np.int32) - 1
        final_cost = _valid_cost(T, xy, tour)
        seed_costs.append(final_cost)
        events, flags = _patched_events(base, result["start_epoch_s"], final_cost)
        improvement_events[str(seed)] = events
        if flags:
            event_flags[str(seed)] = flags
    result.update(arm=kind, seed_costs=seed_costs,
                  best_cost=min(seed_costs),
                  first_costs=first_costs,
                  first_tour_hashes=first_hashes,
                  live_candidate_hashes=live_hashes,
                  native_candidate_hashes=native_hashes,
                  policy_audit=audit,
                  guided_improvement_events=improvement_events,
                  event_flags=event_flags,
                  decision_cache_files_written=cache_files,
                  binary_sha256=sha256_file(patched_binary))
    _atomic_json(work_dir / "endpoint.json", {
        key: result[key] for key in ("arm", "seed_costs", "best_cost",
                                       "first_costs", "first_tour_hashes",
                                       "live_candidate_hashes", "native_candidate_hashes",
                                       "policy_audit", "guided_improvement_events",
                                       "event_flags", "decision_cache_files_written",
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
    captured: dict[int, list[tuple[float, bytes]]] = {seed: [] for seed in seeds}

    def observe_native_tours() -> None:
        # LKH rewrites OUTPUT_TOUR_FILE in place after each new best tour.
        # Inside the deadline the poll only stats the file and copies the
        # bytes of a completed rewrite; tours are parsed and their costs
        # recomputed after the processes stop, so the native arm does not pay
        # for parsing. The mtime-based times are approximate and may miss
        # incumbents overwritten within one 20 ms poll interval.
        for seed in seeds:
            path = work_dir / f"seed_{seed}" / "best.tour"
            try:
                stamp = path.stat()
                if stamp.st_mtime_ns == seen_mtime_ns.get(seed):
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if not _complete_tour_file(data):
                continue  # still being written; the next poll sees a new mtime
            seen_mtime_ns[seed] = stamp.st_mtime_ns
            if not captured[seed] or captured[seed][-1][1] != data:
                captured[seed].append((stamp.st_mtime, data))
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
    wall_elapsed = time.monotonic() - start
    observe_native_tours()
    seed_costs, final_sources, event_flags = [], {}, {}
    events: dict[str, list[dict]] = {}
    for seed in seeds:
        trace, last_valid, flags = [], None, []
        for stamp, data in captured[seed]:
            try:
                cost = _valid_cost(T, xy, parse_tsplib_tour(data))
            except (ValueError, AssertionError, UnicodeDecodeError):
                flags.append("invalid_captured_tour_skipped")
                continue
            last_valid = cost
            if not trace or cost < trace[-1]["cost"]:
                trace.append({"elapsed_s": max(0.0, stamp - start_epoch), "cost": cost,
                              "source": "output_tour_mtime_poll"})
        tour_path = work_dir / f"seed_{seed}" / "best.tour"
        try:
            final_cost = _valid_cost(T, xy, parse_tsplib_tour(tour_path.read_bytes()))
            final_sources[str(seed)] = "final_output_tour_file"
        except (OSError, ValueError, AssertionError, UnicodeDecodeError) as exc:
            # A termination during LKH's in-place rewrite leaves a truncated
            # file; the deadline result is then the last complete tour seen.
            if last_valid is None:
                raise RuntimeError(f"native POP{k} yielded no valid tour for seed {seed}: {exc!r}")
            final_cost = last_valid
            final_sources[str(seed)] = "last_complete_poll_after_invalid_final_file"
        if not trace or trace[-1]["cost"] != final_cost:
            if trace and final_cost > trace[-1]["cost"]:
                raise AssertionError(f"native POP{k} final tour is worse than a logged incumbent: seed {seed}")
            trace.append({"elapsed_s": max(0.0, tour_path.stat().st_mtime - start_epoch),
                          "cost": final_cost, "source": "final_file_mtime"})
            flags.append("final_tour_not_seen_by_poll")
        events[str(seed)] = trace
        if flags:
            event_flags[str(seed)] = sorted(set(flags))
        seed_costs.append(final_cost)
    result = {"arm": kind, "seeds": list(seeds), "wall_budget_s": budget_s,
              "wall_elapsed_s": wall_elapsed,
              "start_epoch_s": start_epoch,
              "deadline_epoch_s": start_epoch + budget_s,
              "seed_costs": seed_costs, "best_cost": min(seed_costs),
              "native_improvement_events": events,
              "final_tour_sources": final_sources,
              "event_flags": event_flags,
              "binary_sha256": sha256_file(native_binary)}
    _atomic_json(work_dir / "endpoint.json", result)
    return result
