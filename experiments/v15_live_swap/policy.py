"""Matched live candidate replacement policies for the v15 pilot.

Every arm pays for its own TourFold inference. The three guided arms use the
same model-defined source gate and evict the exact same candidate at each
source. Each directed row therefore has the same size immediately after the
barrier. Neither a protected first-tour edge nor either of a row's first two
positions can be evicted. A new edge must be absent from the whole undirected
live graph and from all six first tours. LKH may later modify its own rows.

The nearest-novel control is deliberately not called a frequency control:
all eligible new edges have zero frequency in the six first tours. Cached
model output fixes decisions across arms while each arm still pays its own
inference cost.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("sham", "beta_swap", "nearest_swap", "random_swap")
GUIDED_ARMS = ARMS[1:]
ACTION_ALPHA = 500
POLICY_VERSION = "v15-live-swap-policy-1"


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


def _hash_floats(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


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


def model_input_key(xy: np.ndarray, tours: list[np.ndarray], seeds: list[int],
                    graph_k: int, model_identity: str) -> str:
    """Hash of everything the TourFold forward pass reads, plus the model ID."""
    digest = hashlib.sha256()
    xy = np.ascontiguousarray(xy)
    digest.update(f"{POLICY_VERSION}|{model_identity}|{xy.dtype.str}|{xy.shape}|{int(graph_k)}".encode())
    digest.update(xy.tobytes())
    for seed, tour in zip(seeds, tours):
        digest.update(int(seed).to_bytes(8, "little", signed=True))
        digest.update(np.ascontiguousarray(tour, dtype=np.int64).tobytes())
    return digest.hexdigest()


class DecisionCache:
    """First-computed model outputs, keyed by the exact model input.

    Entries are loaded before an arm's timed window and new entries are written
    after it, so cache input/output is never charged to one arm only.
    """

    def __init__(self, directory: Path | str | None = None):
        self.directory = Path(directory) if directory is not None else None
        self._entries: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._new: list[str] = []
        if self.directory is not None and self.directory.exists():
            for path in sorted(self.directory.glob("*.npz")):
                if path.name.endswith(".tmp.npz"):
                    continue
                with np.load(path, allow_pickle=False) as data:
                    key = str(data["key"])
                    if key != path.stem:
                        raise ValueError(f"decision cache key mismatch: {path}")
                    self._entries[key] = (data["NB"].copy(), data["prob"].copy(),
                                          data["D"].copy())

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        return self._entries.get(key)

    def put(self, key: str, graph: np.ndarray, probability: np.ndarray,
            distance: np.ndarray) -> None:
        if key not in self._entries:
            self._entries[key] = (np.array(graph, copy=True),
                                  np.array(probability, copy=True),
                                  np.array(distance, copy=True))
            self._new.append(key)

    def flush(self) -> list[str]:
        """Persist new entries atomically; returns the written file names."""
        written: list[str] = []
        if self.directory is None:
            self._new.clear()
            return written
        self.directory.mkdir(parents=True, exist_ok=True)
        for key in self._new:
            final = self.directory / f"{key}.npz"
            if final.exists():
                continue
            temporary = self.directory / f"{key}.tmp.npz"
            graph, probability, distance = self._entries[key]
            np.savez(temporary, key=np.array(key), NB=graph, prob=probability, D=distance)
            os.replace(temporary, final)
            written.append(final.name)
        self._new.clear()
        return written


def _padded_rows(rows: list, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Directed 0-based targets and alpha values, both padded to a common width."""
    if len(rows) != n:
        raise ValueError("candidate snapshot has wrong dimension")
    width = max(1, max(len(row) for row in rows))
    targets = np.full((n, width), -1, dtype=np.int32)
    alphas = np.zeros((n, width), dtype=np.int32)
    for source, row in enumerate(rows):
        seen: set[int] = set()
        for slot, (destination, alpha) in enumerate(row):
            target = int(destination) - 1
            if target < 0 or target >= n or target == source or target in seen:
                raise ValueError("candidate snapshot has invalid/duplicate neighbour")
            seen.add(target)
            targets[source, slot] = target
            alphas[source, slot] = int(alpha)
    return targets, alphas


def _undirected_edge_count(padded: np.ndarray) -> int:
    n = len(padded)
    sources = np.broadcast_to(np.arange(n)[:, None], padded.shape)
    mask = padded >= 0
    a, b = sources[mask], padded[mask]
    return int(np.unique(np.minimum(a, b) * n + np.maximum(a, b)).size)


def _on_any_first_tour(neighbors: np.ndarray, successors: np.ndarray) -> np.ndarray:
    """Whether a source-neighbor edge occurs in any of the six first tours."""
    n = len(neighbors)
    if successors.shape != (n, 6):
        raise ValueError("expected six first-tour successor columns")
    predecessor = np.empty_like(successors)
    for column in range(6):
        predecessor[successors[:, column], column] = np.arange(n, dtype=successors.dtype)
    protected = np.zeros(neighbors.shape, dtype=bool)
    for column in range(6):
        protected |= (neighbors == successors[:, column, None])
        protected |= (neighbors == predecessor[:, column, None])
    return protected


def _row_membership(graph: np.ndarray, padded: np.ndarray,
                    successors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Model edge in this directed row, and unavailable new undirected edge."""
    n = len(graph)
    in_row = np.any(padded[:, None, :] == graph[:, :, None], axis=2)
    reverse = np.any(padded[graph] == np.arange(n)[:, None, None], axis=2)
    blocked = in_row | reverse | _on_any_first_tour(graph, successors)
    return in_row, blocked


def _victim_choice(padded: np.ndarray, probability: np.ndarray,
                   graph: np.ndarray, successors: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose lowest-model-score eligible live edge in each row.

    Only graph-visible candidates can supply a probability for the gate. On
    equal scores, prefer the later original slot. Slots 1 and 2, and every
    edge from any of the six first tours, are protected.
    """
    n, width = padded.shape
    visible_matches = padded[:, :, None] == graph[:, None, :]
    visible_probability = np.where(visible_matches, probability[:, None, :], np.inf).min(axis=2)
    valid = ((padded >= 0) & np.isfinite(visible_probability)
             & ~_on_any_first_tour(padded, successors))
    valid[:, :min(2, width)] = False
    costs = np.where(valid, visible_probability, np.inf)
    reverse_slot = np.argmin(costs[:, ::-1], axis=1)
    slot = width - 1 - reverse_slot
    has_victim = np.isfinite(costs[np.arange(n), slot])
    slot = np.where(has_victim, slot, -1).astype(np.int32)
    victim = np.where(has_victim, padded[np.arange(n), np.maximum(slot, 0)], -1).astype(np.int32)
    victim_probability = np.where(has_victim,
                                  visible_probability[np.arange(n), np.maximum(slot, 0)],
                                  np.inf)
    return slot, victim, victim_probability


def _gate_margins(probability: np.ndarray, blocked: np.ndarray,
                  victim_probability: np.ndarray) -> np.ndarray:
    strongest = np.where(~blocked, probability, -np.inf).max(axis=1)
    with np.errstate(invalid="ignore"):
        margin = strongest - victim_probability
    valid = np.isfinite(victim_probability) & (~blocked).any(axis=1) & (margin > 0)
    return np.where(valid, margin, -np.inf)


def _candidate_options(source: int, graph: np.ndarray, probability: np.ndarray,
                       distance: np.ndarray, blocked_row: np.ndarray,
                       victim_probability: float,
                       rng: np.random.Generator) -> dict[str, list[int]]:
    novel = [int(column) for column in np.flatnonzero(~blocked_row)]
    if not novel:
        return {arm: [] for arm in GUIDED_ARMS}
    beta = sorted((j for j in novel if probability[source, j] > victim_probability),
                  key=lambda j: (-float(probability[source, j]),
                                 float(distance[source, j]),
                                 int(graph[source, j])))
    nearest = sorted(novel, key=lambda j: (float(distance[source, j]),
                                          int(graph[source, j])))
    random = list(np.asarray(novel)[rng.permutation(len(novel))])
    return {
        "beta_swap": [int(graph[source, j]) for j in beta],
        "nearest_swap": [int(graph[source, j]) for j in nearest],
        "random_swap": [int(graph[source, j]) for j in random],
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
    fraction: float = 0.05,
    graph_k: int = 16,
    decision_cache: DecisionCache | None = None,
    model_identity: str = "",
) -> tuple[list[list[tuple[int, int, int]]], dict]:
    """Return six n-row action matrices and a JSON-serializable parity audit.

    Action tuple is (new neighbor ID, victim neighbor ID, alpha), using
    1-based LKH IDs. No-op is (0, 0, ACTION_ALPHA). The source ID is the row
    index + 1 in the action file. All guided arms share source nodes and
    victim IDs, and have distinct new undirected edges within each arm.
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
    fresh_graph, fresh_probability, fresh_distance = _validate_prediction(prediction, n)
    cache_key = model_input_key(xy, tours, [int(s.seed) for s in snapshots],
                                graph_k, model_identity)
    cached = decision_cache.get(cache_key) if decision_cache is not None else None
    if cached is None:
        graph, probability, distance = fresh_graph, fresh_probability, fresh_distance
        decision_basis = "fresh_inference"
        if decision_cache is not None:
            decision_cache.put(cache_key, graph, probability, distance)
    else:
        graph, probability, distance = _validate_prediction(
            {"NB": cached[0], "prob": cached[1], "D": cached[2]}, n)
        decision_basis = "cached_inference"
    same_graph = bool(np.array_equal(fresh_graph, graph))
    fresh_difference = (float(np.max(np.abs(fresh_probability - probability)))
                        if same_graph else None)

    destinations = {arm: np.full((6, n), -1, dtype=np.int32) for arm in GUIDED_ARMS}
    victims = np.full((6, n), -1, dtype=np.int32)
    victim_slots = np.full((6, n), -1, dtype=np.int32)
    gates: list[list[int]] = []
    eligible_counts: list[int] = []
    native_edge_counts: list[int] = []
    reverse_victim_counts: list[int] = []
    limit = int(np.floor(fraction * n))
    for panel_index, snapshot in enumerate(snapshots):
        padded, _alphas = _padded_rows(snapshot.live_candidates, n)
        native_edge_counts.append(_undirected_edge_count(padded))
        _in_row, blocked = _row_membership(graph, padded, successors)
        slots, eligible_victims, victim_probability = _victim_choice(
            padded, probability, graph, successors)
        margins = _gate_margins(probability, blocked, victim_probability)
        eligible = np.flatnonzero(np.isfinite(margins))
        eligible_counts.append(len(eligible))
        ordered = eligible[np.argsort(-margins[eligible], kind="stable")]
        seed_material = (np.ascontiguousarray(xy).tobytes() +
                         np.ascontiguousarray(tours[panel_index]).tobytes() +
                         int(snapshot.seed).to_bytes(8, "little", signed=True))
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little"))
        reserved = {arm: set() for arm in GUIDED_ARMS}
        gate: list[int] = []
        for source_raw in ordered:
            if len(gate) >= limit:
                break
            source = int(source_raw)
            options = _candidate_options(source, graph, probability, distance,
                                         blocked[source], float(victim_probability[source]), rng)
            proposal: dict[str, tuple[int, int]] = {}
            for arm in GUIDED_ARMS:
                available = ((int(target), _undirected_key(source, int(target), n))
                             for target in options[arm])
                choice = next(((target, key) for target, key in available
                               if key not in reserved[arm]), None)
                if choice is None:
                    break
                proposal[arm] = choice
            if len(proposal) != len(GUIDED_ARMS):
                continue
            victim = int(eligible_victims[source])
            slot = int(slots[source])
            if slot < 2 or victim < 0:
                raise AssertionError("invalid victim selected")
            gate.append(source)
            victims[panel_index, source] = victim
            victim_slots[panel_index, source] = slot
            for arm, (target, key) in proposal.items():
                destinations[arm][panel_index, source] = target
                reserved[arm].add(key)
        gates.append(gate)
        if len({len(reserved[arm]) for arm in GUIDED_ARMS}) != 1:
            raise AssertionError("guided arm capacities differ")
        reverse_victim_counts.append(
            sum(source in padded[victim] for source, victim in
                ((source, int(victims[panel_index, source])) for source in gate)))

    chosen = (np.full((6, n), -1, dtype=np.int32) if kind == "sham"
              else destinations[kind])
    actions = [[(int(chosen[i, source]) + 1, int(victims[i, source]) + 1, ACTION_ALPHA)
                if chosen[i, source] >= 0 else (0, 0, ACTION_ALPHA)
                for source in range(n)] for i in range(6)]
    gate_hashes = [_hash_ints(np.asarray(gate, dtype=np.int32)) for gate in gates]
    victim_hashes = [_hash_ints(victims[i]) for i in range(6)]
    action_hashes = {
        arm: [_hash_ints(np.stack((destinations[arm][i], victims[i]), axis=1))
              for i in range(6)] for arm in GUIDED_ARMS
    }
    selected_probability: dict[str, list[list[float]]] = {}
    unique_undirected: dict[str, list[int]] = {}
    for arm in GUIDED_ARMS:
        probabilities_per_seed, unique_per_seed = [], []
        for i, gate in enumerate(gates):
            sources = np.asarray(gate, dtype=np.int64)
            targets = destinations[arm][i, sources].astype(np.int64)
            match = graph[sources] == targets[:, None]
            if len(gate) and not np.all(match.sum(axis=1) == 1):
                raise AssertionError("selected edge missing from model graph")
            columns = match.argmax(axis=1) if len(gate) else np.zeros(0, dtype=np.int64)
            keys = np.minimum(sources, targets) * n + np.maximum(sources, targets)
            probabilities_per_seed.append([float(v) for v in probability[sources, columns]])
            unique_per_seed.append(int(np.unique(keys).size))
            if unique_per_seed[-1] != len(gate):
                raise AssertionError("selected actions are not unique undirected edges")
        selected_probability[arm] = probabilities_per_seed
        unique_undirected[arm] = unique_per_seed
    audit = {
        "arm": kind,
        "policy_version": POLICY_VERSION,
        "gate_fraction": float(fraction),
        "graph_k": int(graph_k),
        "model_inference_s": float(prediction["t_total"]),
        "decision_basis": decision_basis,
        "decision_cache_key": cache_key,
        "model_probability_sha256": _hash_floats(probability),
        "fresh_model_probability_sha256": _hash_floats(fresh_probability),
        "fresh_graph_matches_decision_graph": same_graph,
        "fresh_distance_matches_decision_distance": bool(np.array_equal(fresh_distance, distance)),
        "fresh_vs_decision_max_abs_probability_difference": fresh_difference,
        "seed_ids": [int(s.seed) for s in snapshots],
        "first_tour_raw_sha256": [_hash_ints(tour) for tour in tours],
        "first_tour_edge_sha256": [_canonical_edge_hash(tour) for tour in tours],
        "native_undirected_edges_per_seed": native_edge_counts,
        "eligible_nodes_per_seed": eligible_counts,
        "gate_nodes_per_seed": gates,
        "gate_sha256_per_seed": gate_hashes,
        "guided_actions_per_seed": [len(gate) for gate in gates],
        "selected_actions_per_seed": [int((row >= 0).sum()) for row in chosen],
        "selected_unique_undirected_edges_per_seed": (
            [0] * 6 if kind == "sham" else unique_undirected[kind]),
        "selected_swaps_per_seed": [
            [{"from_id": int(source + 1),
              "to_id": int(chosen[i, source] + 1),
              "victim_id": int(victims[i, source] + 1),
              "victim_original_slot": int(victim_slots[i, source] + 1)}
             for source in gates[i] if chosen[i, source] >= 0] for i in range(6)
        ],
        "victim_ids_per_seed": victims.tolist(),
        "victim_original_slots_per_seed": victim_slots.tolist(),
        "victim_sha256_per_seed": victim_hashes,
        "all_guided_action_sha256_per_seed": action_hashes,
        "all_guided_unique_undirected_edges_per_seed": unique_undirected,
        "all_guided_selected_model_probability_per_seed": selected_probability,
        "evicted_edges_present_in_reverse_row_per_seed": reverse_victim_counts,
        "expected_directed_row_length_delta_per_seed": [0] * 6,
        "action_alpha": ACTION_ALPHA,
        "action_rule": "in-place replacement; no slot restoration",
    }
    return actions, audit
