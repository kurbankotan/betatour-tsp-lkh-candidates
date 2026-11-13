"""Exact tour-preserving compression followed by a small LKH repair.

Selected tour windows are free. Every untouched path is represented by its two
endpoints joined by a mandatory zero-cost edge in the reduced TSP. Expanding
that edge restores the full path in either orientation. Thus every reduced
Hamilton tour that obeys the mandatory edges expands to a valid full tour.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from repair_engine import select_cuts, tour_cost


@dataclass(frozen=True)
class CompressedTour:
    xy: np.ndarray
    original_nodes: np.ndarray
    initial_tour: np.ndarray
    fixed_edges: np.ndarray
    fixed_paths: dict[tuple[int, int], np.ndarray]
    full_n: int
    free_count: int

    def expand(self, reduced_tour: np.ndarray) -> np.ndarray:
        route = np.asarray(reduced_tour, dtype=np.int32)
        m = len(self.original_nodes)
        if route.shape != (m,) or not np.array_equal(np.sort(route), np.arange(m)):
            raise ValueError("reduced tour must be a Hamilton permutation")
        expanded: list[int] = []
        used_fixed: set[tuple[int, int]] = set()
        for index, rid in enumerate(route):
            rid = int(rid)
            nxt = int(route[(index + 1) % m])
            expanded.append(int(self.original_nodes[rid]))
            key = (min(rid, nxt), max(rid, nxt))
            path = self.fixed_paths.get(key)
            if path is None:
                continue
            used_fixed.add(key)
            if expanded[-1] == int(path[0]) and int(self.original_nodes[nxt]) == int(path[-1]):
                expanded.extend(map(int, path[1:-1]))
            elif expanded[-1] == int(path[-1]) and int(self.original_nodes[nxt]) == int(path[0]):
                expanded.extend(map(int, path[-2:0:-1]))
            else:
                raise AssertionError("mandatory path endpoints inconsistent")
        if used_fixed != set(self.fixed_paths):
            raise ValueError("reduced tour omitted a mandatory path edge")
        out = np.asarray(expanded, dtype=np.int32)
        if out.shape != (self.full_n,) or not np.array_equal(np.sort(out), np.arange(self.full_n)):
            raise AssertionError("expanded tour is not Hamiltonian")
        return out


def free_windows(n: int, cuts: np.ndarray, width: int) -> np.ndarray:
    """Mark width tour positions around every cut, including both endpoints."""
    if width < 2 or width >= n:
        raise ValueError("window width must be in [2,n)")
    free = np.zeros(n, dtype=bool)
    left = width // 2 - 1
    for cut in np.asarray(cuts, dtype=np.int32):
        if not 0 <= int(cut) < n:
            raise ValueError("cut out of range")
        free[(int(cut) + np.arange(-left, width - left)) % n] = True
    return free


def compress_tour(xy: np.ndarray, tour: np.ndarray, free: np.ndarray) -> CompressedTour:
    xy = np.asarray(xy)
    tour = np.asarray(tour, dtype=np.int32)
    free = np.asarray(free, dtype=bool)
    n = len(tour)
    if xy.shape != (n, 2) or free.shape != (n,):
        raise ValueError("xy/tour/free dimension mismatch")
    if not np.isfinite(xy).all() or not np.array_equal(xy, np.rint(xy)):
        raise ValueError("EUC_2D compression requires finite integral coordinates")
    if not np.array_equal(np.sort(tour), np.arange(n)):
        raise ValueError("invalid full tour")
    if free.sum() < 3 or free.all():
        raise ValueError("need at least three free nodes and one fixed node")

    fixed = ~free
    starts = np.flatnonzero(fixed & np.roll(free, 1))
    paths: list[np.ndarray] = []
    keep = free.copy()
    for start in starts:
        positions = []
        pos = int(start)
        while fixed[pos]:
            positions.append(pos)
            pos = (pos + 1) % n
        path = tour[np.asarray(positions, dtype=np.int32)]
        paths.append(path)
        keep[positions[0]] = True
        keep[positions[-1]] = True

    original_nodes = tour[keep].astype(np.int32)
    old_to_new = {int(old): i for i, old in enumerate(original_nodes)}
    pair_paths: dict[tuple[int, int], np.ndarray] = {}
    fixed_pairs = []
    for path in paths:
        if len(path) < 2:
            continue
        a, b = old_to_new[int(path[0])], old_to_new[int(path[-1])]
        key = (min(a, b), max(a, b))
        if key in pair_paths:
            raise AssertionError("duplicate mandatory pair")
        pair_paths[key] = path.copy()
        fixed_pairs.append((a, b))
    pairs = np.asarray(fixed_pairs, dtype=np.int32).reshape(-1, 2)
    reduced = CompressedTour(
        xy=np.asarray(xy[original_nodes], dtype=np.int64),
        original_nodes=original_nodes,
        initial_tour=np.arange(len(original_nodes), dtype=np.int32),
        fixed_edges=pairs,
        fixed_paths=pair_paths,
        full_n=n,
        free_count=int(free.sum()),
    )
    reconstructed = reduced.expand(reduced.initial_tour)
    if tour_cost(xy, reconstructed) != tour_cost(xy, tour):
        raise AssertionError("compression did not preserve original cost")
    return reduced


@dataclass(frozen=True)
class RepairAttempt:
    tour: np.ndarray
    before: int
    after: int
    reduced_n: int
    free_n: int
    elapsed_s: float
    solver_returncode: int | None
    solver_timed_out: bool
    improved: bool
    cuts: tuple[int, ...] = ()


def solve_reduced(
    packed: CompressedTour,
    full_xy: np.ndarray,
    full_tour: np.ndarray,
    lkh_binary: Path,
    work_root: Path,
    *,
    seed: int,
    max_trials: int = 20,
    timeout_s: float = 2.0,
) -> RepairAttempt:
    """Solve one equivalent reduced TSP; retain the original if no gain."""
    import tsplab as T

    began = time.perf_counter()
    before = tour_cost(full_xy, full_tour)
    if timeout_s <= 0 or max_trials < 1:
        raise ValueError("positive solver budget and trials required")
    work_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="repair_", dir=work_root))
    try:
        plain = directory / "plain.tsp"
        problem = directory / "fixed.tsp"
        init = directory / "initial.tour"
        T.write_tsp(plain, packed.xy, name="BetaTour_repair", ew="EUC_2D")
        if len(packed.fixed_edges):
            T.add_fixed_edges(plain, problem, packed.fixed_edges)
        else:
            problem = plain
        T.write_tour(init, packed.initial_tour)
        params = {
            "CANDIDATE_SET_TYPE": "ALPHA",
            "MAX_CANDIDATES": 5,
            "SUBGRADIENT": "NO",
            "INITIAL_TOUR_FILE": str(init),
            "SEED": int(seed),
            "RUNS": 1,
            "MAX_TRIALS": int(max_trials),
            "TIME_LIMIT": max(1.0, timeout_s),
            "TRACE_LEVEL": 0,
            "STOP_AT_OPTIMUM": "NO",
        }
        result = T.run_lkh(str(lkh_binary), str(problem), params,
                           str(directory / "run"), timeout=timeout_s)
        candidate = result["tour"]
        after = before
        selected = np.asarray(full_tour, dtype=np.int32).copy()
        if candidate is not None:
            candidate = packed.expand(candidate)
            cost = tour_cost(full_xy, candidate)
            if cost < before:
                selected, after = candidate, cost
        return RepairAttempt(selected, before, after, len(packed.original_nodes),
                             packed.free_count, time.perf_counter() - began,
                             result["returncode"], bool(result["timed_out"]), after < before)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def repair_once(
    xy: np.ndarray,
    tour: np.ndarray,
    lkh_binary: Path,
    work_root: Path,
    *,
    strategy: str,
    ensemble_tours: list[np.ndarray] | None,
    model_scores,
    rng: np.random.Generator,
    cut_count: int = 3,
    window_size: int = 48,
    timeout_s: float = 2.0,
    max_trials: int = 20,
) -> RepairAttempt:
    cuts = select_cuts(strategy, xy, tour, ensemble_tours=ensemble_tours,
                       model_scores=model_scores, rng=rng, k=cut_count,
                       min_segment_size=window_size)
    free = free_windows(len(tour), cuts, window_size)
    packed = compress_tour(xy, tour, free)
    result = solve_reduced(packed, xy, tour, lkh_binary, work_root,
                           seed=int(rng.integers(1, 2**31 - 1)),
                           max_trials=max_trials, timeout_s=timeout_s)
    return replace(result, cuts=tuple(map(int, cuts)))
