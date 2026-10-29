"""Matched sparse candidate controls for the BetaTour POP6 mechanism pilot.

The four arms all pay for one TourFold inference. Beta, frequency, and random
use the *same* model-defined source-node gate and add one candidate from the
same model graph at each selected source. A candidate must be absent from the
undirected POP6 graph and that seed's first tour. Selected additions must also
be distinct as undirected edges. POP6 edges are never deleted or reweighted.

This is a mechanism experiment. TourFold probabilities predict reference-tour
membership, not the marginal value of a candidate to LKH search.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


ARMS = ("sham", "beta_add", "frequency_add", "random_add")
GUIDED_ARMS = ARMS[1:]
ACTION_SLOT = 3
ACTION_ALPHA = 500


def _undirected_key(a: int, b: int, n: int) -> int:
    return min(a, b) * n + max(a, b)


def _canonical_edge_hash(tour: np.ndarray) -> str:
    tour = np.asarray(tour, dtype=np.int32)
    following = np.roll(tour, -1)
    edges = np.column_stack((np.minimum(tour, following),
                             np.maximum(tour, following)))
    edges = np.ascontiguousarray(edges[np.lexsort((edges[:, 1], edges[:, 0]))])
    return hashlib.sha256(edges.tobytes()).hexdigest()


def _hash_ints(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.int32).tobytes()).hexdigest()


def _snapshot_tours(snapshots: list[Any], xy: np.ndarray, T: Any) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    n = len(xy)
    tours, successors, costs = [], [], []
    for snapshot in snapshots:
        tour = np.asarray(snapshot.first_tour, dtype=np.int32) - 1
        if not T.is_valid_tour(tour, n):
            raise ValueError(f"invalid first tour for seed {snapshot.seed}")
        tours.append(tour)
        successors.append(np.asarray(T.tour_succ_pred(tour)[0], dtype=np.int32))
        costs.append(float(T.tour_length(xy, tour, "EUC_2D")))
    return tours, np.stack(successors, axis=1), np.asarray(costs, dtype=np.float64)


def _validate_prediction(prediction: dict, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    graph = np.asarray(prediction["NB"], dtype=np.int32)
    probability = np.asarray(prediction["prob"], dtype=np.float64)
    distance = np.asarray(prediction["D"], dtype=np.float64)
    if graph.shape != probability.shape or graph.shape != distance.shape or graph.ndim != 2:
        raise ValueError("model graph, probability and distance shapes differ")
    if graph.shape[0] != n or not 1 <= graph.shape[1] < n:
        raise ValueError("model graph has invalid dimensions")
    if np.any((graph < 0) | (graph >= n)) or np.any(graph == np.arange(n)[:, None]):
        raise ValueError("model graph contains invalid/self edges")
    if np.any(np.sort(graph, axis=1)[:, 1:] == np.sort(graph, axis=1)[:, :-1]):
        raise ValueError("model graph contains duplicate neighbours")
    if not np.isfinite(probability).all() or not np.isfinite(distance).all():
        raise ValueError("model prediction is not finite")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("model probability is outside [0,1]")
    return graph, probability, distance


def _existing_graph(snapshot: Any, n: int) -> set[int]:
    rows = snapshot.live_candidates
    if len(rows) != n:
        raise ValueError("candidate snapshot has wrong dimension")
    edges: set[int] = set()
    for source, row in enumerate(rows):
        for destination, _alpha in row:
            destination = int(destination) - 1
            if not 0 <= destination < n or destination == source:
                raise ValueError("candidate snapshot has invalid neighbour")
            edges.add(_undirected_key(source, destination, n))
    return edges


def _ensemble_counts(tours: list[np.ndarray], n: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    for tour in tours:
        for a, b in zip(tour, np.roll(tour, -1)):
            key = _undirected_key(int(a), int(b), n)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _candidate_options(
    source: int,
    graph: np.ndarray,
    probability: np.ndarray,
    distance: np.ndarray,
    existing: set[int],
    counts: dict[int, int],
    n: int,
    rng: np.random.Generator,
) -> dict[str, list[int]]:
    novel_cols = [column for column, destination in enumerate(graph[source])
                  if _undirected_key(source, int(destination), n) not in existing]
    if not novel_cols:
        return {arm: [] for arm in GUIDED_ARMS}
    beta_order = sorted(novel_cols, key=lambda j: (-float(probability[source, j]), j))
    frequency_order = sorted(
        novel_cols,
        key=lambda j: (-counts.get(_undirected_key(source, int(graph[source, j]), n), 0),
                       float(distance[source, j]), int(graph[source, j])),
    )
    random_order = list(np.asarray(novel_cols)[rng.permutation(len(novel_cols))])
    return {
        "beta_add": [int(graph[source, j]) for j in beta_order],
        "frequency_add": [int(graph[source, j]) for j in frequency_order],
        "random_add": [int(graph[source, j]) for j in random_order],
    }


def choose_actions(
    kind: str,
    snapshots: list[Any],
    xy: np.ndarray,
    model: Any,
    device: Any,
    T: Any,
    M: Any,
    *,
    fraction: float = 0.25,
    graph_k: int = 16,
) -> tuple[list[list[tuple[int, int, int]]], dict]:
    """Return LKH action matrices and a JSON-serializable parity audit.

    The source gate is TourFold's v9-style positive probability margin over
    the weakest model-visible POP6 edge. Frequency and random controls keep
    that exact gate. The output may contain fewer than floor(fraction*n)
    actions if no common feasible novel choice remains; all guided arms still
    get the same number and node IDs for each seed.
    """
    if kind not in ARMS:
        raise ValueError(f"unknown candidate arm: {kind}")
    if len(snapshots) != 6 or len({int(s.seed) for s in snapshots}) != 6:
        raise ValueError("exactly six distinct seed snapshots required")
    if not 0.0 < fraction <= 1.0 or graph_k < 2:
        raise ValueError("invalid gate fraction or graph width")
    xy = np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 4:
        raise ValueError("xy must have shape (n,2)")
    n = len(xy)
    tours, successors, costs = _snapshot_tours(snapshots, xy, T)
    prediction = M.predict_tourfold(model, xy, graph_k, device,
                                    ens_succ=successors, ens_cost=costs)
    graph, probability, distance = _validate_prediction(prediction, n)
    ensemble_counts = _ensemble_counts(tours, n)
    destinations = {arm: np.full((6, n), -1, dtype=np.int32) for arm in GUIDED_ARMS}
    gates: list[list[int]] = []
    eligible_counts: list[int] = []
    pop_edge_counts: list[int] = []
    limit = int(np.floor(fraction * n))

    for panel_index, snapshot in enumerate(snapshots):
        existing = _existing_graph(snapshot, n)
        pop_edge_counts.append(len(existing))
        first_tour = tours[panel_index]
        first_edges = {_undirected_key(int(a), int(b), n)
                       for a, b in zip(first_tour, np.roll(first_tour, -1))}
        blocked = existing | first_edges
        margins = np.full(n, -np.inf, dtype=np.float64)
        for source in range(n):
            visible = {int(v): float(p) for v, p in zip(graph[source], probability[source])}
            scored_pop = [visible[int(v) - 1] for v, _alpha in snapshot.live_candidates[source]
                          if int(v) - 1 in visible]
            if len(scored_pop) < 2:
                continue
            novel_prob = [float(p) for v, p in zip(graph[source], probability[source])
                          if _undirected_key(source, int(v), n) not in blocked]
            if novel_prob:
                margin = max(novel_prob) - min(scored_pop)
                if margin > 0:
                    margins[source] = margin
        eligible = np.flatnonzero(np.isfinite(margins))
        eligible_counts.append(len(eligible))
        ordered = eligible[np.argsort(-margins[eligible], kind="stable")]
        seed_material = (np.ascontiguousarray(xy).tobytes() +
                         np.ascontiguousarray(tours[panel_index]).tobytes() +
                         int(snapshot.seed).to_bytes(8, "little", signed=True))
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little"))
        reserved = {arm: set() for arm in GUIDED_ARMS}
        gate: list[int] = []
        for source in ordered:
            if len(gate) >= limit:
                break
            source = int(source)
            options = _candidate_options(source, graph, probability, distance,
                                         blocked, ensemble_counts, n, rng)
            proposal: dict[str, tuple[int, int]] = {}
            for arm in GUIDED_ARMS:
                available = ((int(destination), _undirected_key(source, int(destination), n))
                             for destination in options[arm])
                choice = next(((destination, key) for destination, key in available
                               if key not in reserved[arm]), None)
                if choice is None:
                    break
                proposal[arm] = choice
            if len(proposal) != len(GUIDED_ARMS):
                continue
            gate.append(source)
            for arm, (destination, key) in proposal.items():
                destinations[arm][panel_index, source] = destination
                reserved[arm].add(key)
        gates.append(gate)
        if len({len(reserved[arm]) for arm in GUIDED_ARMS}) != 1:
            raise AssertionError("guided arm capacities differ")

    chosen = (np.full((6, n), -1, dtype=np.int32) if kind == "sham"
              else destinations[kind])
    actions = [[(int(v) + 1, ACTION_SLOT, ACTION_ALPHA) if v >= 0 else (0, 0, ACTION_ALPHA)
                for v in row] for row in chosen]
    gate_hashes = [_hash_ints(np.asarray(gate, dtype=np.int32)) for gate in gates]
    action_hashes = {arm: [_hash_ints(destinations[arm][i]) for i in range(6)]
                     for arm in GUIDED_ARMS}
    selected_frequency = {}
    selected_probability = {}
    unique_undirected = {}
    for arm in GUIDED_ARMS:
        frequencies_per_seed, probabilities_per_seed, unique_per_seed = [], [], []
        for i, gate in enumerate(gates):
            frequencies, probabilities, keys = [], [], set()
            for source in gate:
                destination = int(destinations[arm][i, source])
                keys.add(_undirected_key(source, destination, n))
                frequencies.append(int(ensemble_counts.get(_undirected_key(source, destination, n), 0)))
                column = np.flatnonzero(graph[source] == destination)
                if len(column) != 1:
                    raise AssertionError("selected edge missing from model graph")
                probabilities.append(float(probability[source, int(column[0])]))
            frequencies_per_seed.append(frequencies)
            probabilities_per_seed.append(probabilities)
            unique_per_seed.append(len(keys))
            if len(keys) != len(gate):
                raise AssertionError("selected actions are not unique undirected edges")
        selected_frequency[arm] = frequencies_per_seed
        selected_probability[arm] = probabilities_per_seed
        unique_undirected[arm] = unique_per_seed
    audit = {
        "arm": kind,
        "gate_fraction": float(fraction),
        "graph_k": int(graph_k),
        "model_inference_s": float(prediction["t_total"]),
        "seed_ids": [int(s.seed) for s in snapshots],
        "first_tour_raw_sha256": [_hash_ints(tour) for tour in tours],
        "first_tour_edge_sha256": [_canonical_edge_hash(tour) for tour in tours],
        "model_probability_sha256": hashlib.sha256(np.ascontiguousarray(probability).tobytes()).hexdigest(),
        "native_undirected_edges_per_seed": pop_edge_counts,
        "eligible_nodes_per_seed": eligible_counts,
        "gate_nodes_per_seed": gates,
        "gate_sha256_per_seed": gate_hashes,
        "guided_actions_per_seed": [len(gate) for gate in gates],
        "selected_actions_per_seed": [int((row >= 0).sum()) for row in chosen],
        "selected_unique_undirected_edges_per_seed": (
            [0] * 6 if kind == "sham" else unique_undirected[kind]),
        "selected_pairs_per_seed": [[[int(source), int(chosen[i, source])]
                                     for source in gates[i] if chosen[i, source] >= 0]
                                    for i in range(6)],
        "all_guided_action_sha256_per_seed": action_hashes,
        "all_guided_unique_undirected_edges_per_seed": unique_undirected,
        "all_guided_selected_ensemble_frequency_per_seed": selected_frequency,
        "all_guided_selected_model_probability_per_seed": selected_probability,
        "action_slot": ACTION_SLOT,
        "action_alpha": ACTION_ALPHA,
    }
    return actions, audit
