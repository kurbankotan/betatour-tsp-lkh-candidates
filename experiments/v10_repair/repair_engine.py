"""Budgeted tour repair by exactly reconnecting selected fixed segments.

The cut edges may be selected from spatially related but tour-distant parts of
the current solution.  All uncut edges remain intact.  Held--Karp dynamic
programming then finds the cheapest cyclic order and orientation of the fixed
segments under TSPLIB EUC_2D costs.  The original tour is always feasible, and
only a strictly shorter tour is returned.

This is deliberately a small, auditable *mechanism pilot*.  It does not claim
that a neural selector helps, or that segment reconnection beats LKH.  The
same optimizer and cut budget can be used with random, length, tour-disagreement
and model-based selectors.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import numpy as np


ModelScores = np.ndarray | Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RepairResult:
    tour: np.ndarray
    cost_before: int
    cost_after: int
    attempts: int
    accepted_moves: int
    elapsed_s: float
    cuts: tuple[int, ...] = ()

    @property
    def improvement(self) -> int:
        return self.cost_before - self.cost_after


def _validated_xy_tour(xy: np.ndarray, tour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float64)
    tour = np.asarray(tour, dtype=np.int32)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 4:
        raise ValueError("xy must have shape (n, 2), n >= 4")
    if not np.isfinite(xy).all():
        raise ValueError("xy must be finite")
    if tour.shape != (len(xy),) or not np.array_equal(np.sort(tour), np.arange(len(xy))):
        raise ValueError("tour must be a zero-based permutation of all nodes")
    return xy, tour


def euc2d_lengths(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """TSPLIB EUC_2D: nearest integer, with .5 rounded upwards."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.floor(np.sqrt(np.sum((a - b) ** 2, axis=-1)) + 0.5).astype(np.int64)


def tour_cost(xy: np.ndarray, tour: np.ndarray) -> int:
    """Integer EUC_2D length of a cyclic tour."""
    xy, tour = _validated_xy_tour(xy, tour)
    return int(euc2d_lengths(xy[tour], xy[np.roll(tour, -1)]).sum(dtype=np.int64))


def _edge_keys(tour: np.ndarray) -> np.ndarray:
    n = len(tour)
    nxt = np.roll(tour, -1).astype(np.int64)
    a = tour.astype(np.int64)
    return np.minimum(a, nxt) * np.int64(n) + np.maximum(a, nxt)


def _ensemble_counts(ensemble_tours: Sequence[np.ndarray], n: int) -> dict[int, int]:
    if len(ensemble_tours) == 0:
        raise ValueError("disagreement requires at least one ensemble tour")
    counts: dict[int, int] = {}
    for item in ensemble_tours:
        t = np.asarray(item, dtype=np.int32)
        if t.shape != (n,) or not np.array_equal(np.sort(t), np.arange(n)):
            raise ValueError("every ensemble tour must be a zero-based permutation")
        for key in _edge_keys(t):
            k = int(key)
            counts[k] = counts.get(k, 0) + 1
    return counts


def _risk_scores(
    strategy: str,
    xy: np.ndarray,
    tour: np.ndarray,
    ensemble_tours: Sequence[np.ndarray] | Mapping[int, int] | None,
    model_scores: ModelScores | None,
) -> np.ndarray:
    n = len(tour)
    if strategy == "random":
        return np.zeros(n, dtype=np.float64)
    if strategy == "longest":
        return euc2d_lengths(xy[tour], xy[np.roll(tour, -1)]).astype(np.float64)
    if strategy == "disagreement":
        if ensemble_tours is None:
            raise ValueError("disagreement requires ensemble_tours")
        counts = (ensemble_tours if isinstance(ensemble_tours, Mapping)
                  else _ensemble_counts(ensemble_tours, n))
        denominator = max(counts.values()) if counts else 1
        return np.fromiter((1.0 - counts.get(int(k), 0) / denominator
                            for k in _edge_keys(tour)), dtype=np.float64, count=n)
    if strategy in ("score", "model"):
        if model_scores is None:
            raise ValueError("model selector requires model_scores")
        if callable(model_scores):
            scores = np.asarray(model_scores(tour.copy()), dtype=np.float64)
        else:
            by_node = np.asarray(model_scores, dtype=np.float64)
            if by_node.shape != (n,):
                raise ValueError("static model_scores must be node-indexed shape (n,)")
            scores = by_node[tour]
        if scores.shape != (n,) or not np.isfinite(scores).all():
            raise ValueError("model_scores must yield finite risk by tour position")
        return scores
    raise ValueError(f"unknown cut selection strategy: {strategy}")


def _circular_gap(a: int, b: int, n: int) -> int:
    return min((a - b) % n, (b - a) % n)


def select_cuts(
    strategy: str,
    xy: np.ndarray,
    tour: np.ndarray,
    ensemble_tours: Sequence[np.ndarray] | Mapping[int, int] | None = None,
    model_scores: ModelScores | None = None,
    rng: np.random.Generator | None = None,
    k: int = 8,
    min_segment_size: int = 8,
) -> np.ndarray:
    """Choose spatially nearby and tour-distant cut edges.

    ``model_scores`` are *risk* scores: higher means more desirable to cut.
    A callable is evaluated on the current tour and returns scores by tour
    position.  A static array is indexed by source node ID, so it represents
    node-level risk rather than a frozen position or edge probability.
    ``ensemble_tours`` are zero-based tours or precomputed undirected key counts.
    The same hotspot geometry is used for every strategy; the strategy changes
    only the risk score that biases the anchor and partner choices.
    """
    xy, tour = _validated_xy_tour(xy, tour)
    n = len(tour)
    if not 3 <= k <= 10:
        raise ValueError("k must be between 3 and 10")
    if min_segment_size < 1 or n < k * min_segment_size:
        raise ValueError("infeasible minimum segment size")
    rng = np.random.default_rng() if rng is None else rng
    risk = _risk_scores(strategy, xy, tour, ensemble_tours, model_scores)

    # Robust [0,1] normalization makes Gumbel perturbations comparable across
    # length, disagreement and learned scores without assuming calibration.
    lo, hi = float(np.quantile(risk, 0.05)), float(np.quantile(risk, 0.95))
    if hi <= lo:
        normalized = np.zeros(n, dtype=np.float64)
    else:
        normalized = np.clip((risk - lo) / (hi - lo), 0.0, 1.0)
    if strategy == "random":
        anchor = int(rng.integers(n))
    else:
        anchor = int(np.argmax(normalized + 0.30 * rng.gumbel(size=n)))

    midpoint = (xy[tour] + xy[np.roll(tour, -1)]) * 0.5
    squared = np.sum((midpoint - midpoint[anchor]) ** 2, axis=1)
    nearby = np.argsort(squared, kind="stable")
    chosen = [anchor]
    pool_size = min(n, max(128, 32 * k))
    while len(chosen) < k:
        pool = nearby[:pool_size]
        rank = np.arange(len(pool), dtype=np.float64)
        # Random picks retain the same spatial pool as learned selectors.
        priority = normalized[pool] + 0.35 * rng.gumbel(size=len(pool)) - 0.30 * rank / len(pool)
        for position in pool[np.argsort(-priority, kind="stable")]:
            position = int(position)
            if position in chosen:
                continue
            if all(_circular_gap(position, other, n) >= min_segment_size for other in chosen):
                chosen.append(position)
                if len(chosen) == k:
                    break
        if len(chosen) == k:
            break
        if pool_size == n:
            raise RuntimeError("failed to select feasible cuts")
        pool_size = min(n, pool_size * 2)
    return np.sort(np.asarray(chosen, dtype=np.int32))


def _segments_from_cuts(tour: np.ndarray, cuts: np.ndarray) -> list[np.ndarray]:
    n = len(tour)
    out = []
    for idx, cut in enumerate(cuts):
        begin = (int(cut) + 1) % n
        end = int(cuts[(idx + 1) % len(cuts)])
        if begin <= end:
            part = tour[begin:end + 1]
        else:
            part = np.concatenate((tour[begin:], tour[:end + 1]))
        out.append(part)
    if sum(len(p) for p in out) != n:
        raise AssertionError("cut decomposition did not cover the tour")
    return out


def repair_tour(xy: np.ndarray, tour: np.ndarray, cuts: Sequence[int]) -> RepairResult:
    """Exactly optimize cyclic reconnection of the fixed segments.

    Cut indices refer to edges ``tour[i] -> tour[(i+1) % n]``.  Segment zero is
    held in its forward orientation; for a symmetric TSP this loses no cycles,
    because any cycle with segment zero reversed can be reversed globally.
    """
    began = time.perf_counter()
    xy, tour = _validated_xy_tour(xy, tour)
    n = len(tour)
    cuts = np.asarray(cuts, dtype=np.int32)
    if cuts.ndim != 1 or not 3 <= len(cuts) <= 10:
        raise ValueError("cuts must contain 3 to 10 positions")
    if np.any((cuts < 0) | (cuts >= n)) or len(np.unique(cuts)) != len(cuts):
        raise ValueError("cut positions must be unique and within the tour")
    cuts = np.sort(cuts)
    segments = _segments_from_cuts(tour, cuts)
    k = len(segments)
    before = tour_cost(xy, tour)

    starts = np.asarray([[part[0], part[-1]] for part in segments], dtype=np.int32)
    ends = np.asarray([[part[-1], part[0]] for part in segments], dtype=np.int32)
    connector = np.empty((k, 2, k, 2), dtype=np.int64)
    for a in range(k):
        for oa in range(2):
            for b in range(k):
                connector[a, oa, b] = euc2d_lengths(
                    xy[ends[a, oa]], xy[starts[b]]
                )

    full = (1 << k) - 1
    infinity = np.iinfo(np.int64).max // 8
    dp = np.full((1 << k, k, 2), infinity, dtype=np.int64)
    parent_last = np.full((1 << k, k, 2), -1, dtype=np.int8)
    parent_ori = np.full((1 << k, k, 2), -1, dtype=np.int8)
    dp[1, 0, 0] = 0
    for mask in range(1, full + 1, 2):  # bit 0 is always present
        for last in range(k):
            if not (mask & (1 << last)):
                continue
            for ori in range(2):
                current = int(dp[mask, last, ori])
                if current == infinity:
                    continue
                for nxt in range(1, k):
                    if mask & (1 << nxt):
                        continue
                    new_mask = mask | (1 << nxt)
                    for next_ori in range(2):
                        value = current + int(connector[last, ori, nxt, next_ori])
                        if value < dp[new_mask, nxt, next_ori]:
                            dp[new_mask, nxt, next_ori] = value
                            parent_last[new_mask, nxt, next_ori] = last
                            parent_ori[new_mask, nxt, next_ori] = ori

    best_connectors = infinity
    best_last = best_ori = -1
    for last in range(1, k):
        for ori in range(2):
            value = int(dp[full, last, ori] + connector[last, ori, 0, 0])
            if value < best_connectors:
                best_connectors, best_last, best_ori = value, last, ori

    order: list[tuple[int, int]] = []
    mask, last, ori = full, best_last, best_ori
    while mask != 1:
        order.append((last, ori))
        prior_last, prior_ori = int(parent_last[mask, last, ori]), int(parent_ori[mask, last, ori])
        mask ^= 1 << last
        last, ori = prior_last, prior_ori
    order.append((0, 0))
    order.reverse()
    rebuilt = np.concatenate([segments[s] if o == 0 else segments[s][::-1]
                              for s, o in order]).astype(np.int32, copy=False)
    after = tour_cost(xy, rebuilt)
    if after > before:
        raise AssertionError("exact reconnection worsened the original tour")
    # An equivalent orientation/permutation needlessly changes the tour order.
    if after == before:
        rebuilt = tour.copy()
    return RepairResult(rebuilt, before, after, 1, int(after < before),
                        time.perf_counter() - began, tuple(map(int, cuts)))


def repeated_repair(
    xy: np.ndarray,
    tour: np.ndarray,
    *,
    strategy: str,
    ensemble_tours: Sequence[np.ndarray] | None = None,
    model_scores: ModelScores | None = None,
    seed: int = 0,
    k: int = 8,
    attempts: int = 16,
    max_seconds: float = 5.0,
    min_segment_size: int = 8,
) -> RepairResult:
    """Try several matched spatial repair opportunities within a wall cap.

    The cap is checked before each bounded k<=10 dynamic-programming solve.
    Individual solves may exceed the cap slightly; callers should reserve a
    safety margin in their total experiment budget.
    """
    began = time.perf_counter()
    xy, current = _validated_xy_tour(xy, tour)
    if attempts < 0 or max_seconds < 0 or not math.isfinite(max_seconds):
        raise ValueError("attempts and finite max_seconds must be nonnegative")
    initial_cost = tour_cost(xy, current)
    best_cost = initial_cost
    rng = np.random.default_rng(seed)
    counts: Mapping[int, int] | None = None
    if strategy == "disagreement":
        if ensemble_tours is None:
            raise ValueError("disagreement requires ensemble_tours")
        counts = _ensemble_counts(ensemble_tours, len(current))
    tried = accepted = 0
    final_cuts: tuple[int, ...] = ()
    while tried < attempts and time.perf_counter() - began < max_seconds:
        cuts = select_cuts(strategy, xy, current,
                           ensemble_tours=counts if counts is not None else ensemble_tours,
                           model_scores=model_scores, rng=rng, k=k,
                           min_segment_size=min_segment_size)
        trial = repair_tour(xy, current, cuts)
        tried += 1
        if trial.cost_after < best_cost:
            current, best_cost = trial.tour, trial.cost_after
            accepted += 1
            final_cuts = trial.cuts
    return RepairResult(current.copy(), initial_cost, best_cost, tried, accepted,
                        time.perf_counter() - began, final_cuts)
