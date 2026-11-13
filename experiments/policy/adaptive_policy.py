"""Conservative POPMUSIC candidate actions from an existing TourFold prediction.

The functions in this file do not claim to estimate search utility. They turn
TourFold's reference-edge scores into *testable* interventions for a live LKH
experiment. No action is taken when an edge cannot be scored reliably.

Inputs use zero-based node IDs. ``pop`` has shape (seeds, nodes, k_pop), with
``-1`` for an unfilled native POPMUSIC slot. ``neighbours``/``probabilities``
have shape (nodes, k_graph). The same model prediction may be applied to each
seed's different POPMUSIC candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SparseAddConfig:
    """A precommitted density cap and evidence rule for one extra edge per node.

    ``max_fraction`` limits intervention to the best-scored fraction of nodes
    in *each seed*. ``min_scored_pop`` requires a minimum number of the POP
    edges to be present in TourFold's scoring graph. ``min_margin`` is the
    probability difference between the proposed edge and the weakest scored
    POP edge; these probabilities are not assumed calibrated.
    """

    max_fraction: float = 0.25
    min_scored_pop: int = 2
    min_margin: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_fraction <= 1.0:
            raise ValueError("max_fraction must be in [0, 1]")
        if self.min_scored_pop < 1:
            raise ValueError("min_scored_pop must be positive")
        if not np.isfinite(self.min_margin):
            raise ValueError("min_margin must be finite")


@dataclass
class SparseAddResult:
    """One per-node optional action for every independent LKH process.

    ``destination == -1`` means keep the native POPMUSIC candidate row.
    ``proposed_destination`` records a possible graph-novel edge even when the
    gate declines it. ``score`` is a proxy, not an observed LKH gain.
    """

    destination: np.ndarray
    proposed_destination: np.ndarray
    proposed_probability: np.ndarray
    score: np.ndarray
    scored_pop_count: np.ndarray
    eligible: np.ndarray


def _validate(pop: np.ndarray, neighbours: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pop = np.asarray(pop)
    neighbours = np.asarray(neighbours)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if pop.ndim != 3 or neighbours.ndim != 2 or probabilities.shape != neighbours.shape:
        raise ValueError("expected pop=(seeds,nodes,k), neighbours=probabilities=(nodes,k_graph)")
    seeds, n, k_pop = pop.shape
    if seeds < 1 or n < 3 or k_pop < 1 or neighbours.shape[0] != n or neighbours.shape[1] < 1:
        raise ValueError("invalid array dimensions")
    if k_pop >= n or neighbours.shape[1] >= n:
        raise ValueError("candidate width must be below node count")
    if not np.issubdtype(pop.dtype, np.integer) or not np.issubdtype(neighbours.dtype, np.integer):
        raise ValueError("candidate IDs must be integers")
    if np.any((pop < -1) | (pop >= n)) or np.any((neighbours < 0) | (neighbours >= n)):
        raise ValueError("candidate IDs outside instance")
    origin = np.arange(n)
    if np.any(pop == origin[None, :, None]) or np.any(neighbours == origin[:, None]):
        raise ValueError("self-edge in candidates")
    # Multiple -1 slots are legitimate. Treat them as a sentinel above all
    # node IDs only for duplicate detection; never turn them into candidates.
    sorted_pop = np.sort(np.where(pop >= 0, pop, n), axis=2)
    if np.any((sorted_pop[:, :, 1:] == sorted_pop[:, :, :-1])
              & (sorted_pop[:, :, 1:] != n)):
        raise ValueError("duplicate POP candidate in a row")
    if np.any(np.sort(neighbours, axis=1)[:, 1:] == np.sort(neighbours, axis=1)[:, :-1]):
        raise ValueError("duplicate model graph neighbour in a row")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return pop.astype(np.int32, copy=False), neighbours.astype(np.int32, copy=False), probabilities


def _pop_scores(pop_row: np.ndarray, neighbours: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matches = pop_row[:, :, None] == neighbours[:, None, :]
    present = matches.any(axis=2)
    scores = np.where(present, (matches * probabilities[:, None, :]).sum(axis=2), np.nan)
    return scores, present


def rerank_scored_pop(pop: np.ndarray, neighbours: np.ndarray, probabilities: np.ndarray,
                      min_margin: float = 0.0) -> np.ndarray:
    """Reorder only model-visible POP edges, keeping unseen positions fixed.

    Candidate membership and each row's original unseen-edge positions remain
    exactly unchanged. The caller can compare this arm with a sham live hook.
    """
    pop, neighbours, probabilities = _validate(pop, neighbours, probabilities)
    if not np.isfinite(min_margin) or min_margin < 0:
        raise ValueError("min_margin must be finite and nonnegative")
    out = pop.copy()
    for seed in range(len(pop)):
        scores, present = _pop_scores(pop[seed], neighbours, probabilities)
        for node in range(len(neighbours)):
            slots = np.flatnonzero(present[node])
            if len(slots) < 2 or np.ptp(scores[node, slots]) <= min_margin:
                continue
            order = np.argsort(-scores[node, slots], kind="stable")
            out[seed, node, slots] = pop[seed, node, slots[order]]
    return out


def _edge_keys(n: int, destination: np.ndarray) -> np.ndarray:
    origin = np.arange(n, dtype=np.int64)[:, None]
    dest = np.asarray(destination, dtype=np.int64)
    keys = np.minimum(origin, dest) * np.int64(n) + np.maximum(origin, dest)
    return np.where(dest >= 0, keys, -1)


def propose_sparse_add(pop: np.ndarray, neighbours: np.ndarray, probabilities: np.ndarray,
                       config: SparseAddConfig = SparseAddConfig()) -> SparseAddResult:
    """Select at most one globally POP-novel edge for a subset of nodes.

    The no-op option is explicit. Novelty is checked against the full
    *undirected* POP candidate graph for each seed, matching the v8 protocol.
    Selection is deterministic: probability rank breaks ties by original
    graph order; node ID breaks equal proxy scores.
    """
    pop, neighbours, probabilities = _validate(pop, neighbours, probabilities)
    seeds, n, k_pop = pop.shape
    if config.min_scored_pop > k_pop:
        raise ValueError("min_scored_pop exceeds POP width")
    destination = np.full((seeds, n), -1, dtype=np.int32)
    proposed = np.full((seeds, n), -1, dtype=np.int32)
    picked_prob = np.full((seeds, n), np.nan, dtype=np.float64)
    score = np.full((seeds, n), np.nan, dtype=np.float64)
    covered = np.zeros((seeds, n), dtype=np.int16)
    eligible = np.zeros((seeds, n), dtype=bool)
    ranked_graph = np.argsort(-probabilities, axis=1, kind="stable")
    graph_keys = _edge_keys(n, neighbours)
    limit = int(np.floor(config.max_fraction * n))

    for seed in range(seeds):
        pop_keys = np.unique(_edge_keys(n, pop[seed]))
        pop_keys = pop_keys[pop_keys >= 0]
        novel = ~np.isin(graph_keys, pop_keys)
        ranked_novel = np.take_along_axis(novel, ranked_graph, axis=1)
        has_novel = ranked_novel.any(axis=1)
        first_rank = ranked_novel.argmax(axis=1)
        chosen_col = ranked_graph[np.arange(n), first_rank]
        proposed[seed, has_novel] = neighbours[np.arange(n), chosen_col][has_novel]
        picked_prob[seed, has_novel] = probabilities[np.arange(n), chosen_col][has_novel]

        pop_prob, present = _pop_scores(pop[seed], neighbours, probabilities)
        covered[seed] = present.sum(axis=1)
        weakest_scored = np.min(np.where(present, pop_prob, np.inf), axis=1)
        score[seed, has_novel] = picked_prob[seed, has_novel] - weakest_scored[has_novel]
        eligible[seed] = (has_novel & (covered[seed] >= config.min_scored_pop)
                          & (score[seed] > config.min_margin))
        if limit:
            candidates = np.flatnonzero(eligible[seed])
            order = np.argsort(-score[seed, candidates], kind="stable")
            selected = candidates[order[:limit]]
            destination[seed, selected] = proposed[seed, selected]

    return SparseAddResult(destination, proposed, picked_prob, score, covered, eligible)
