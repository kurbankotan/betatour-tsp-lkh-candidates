"""Synthetic invariants for the v15 matched live-swap policy."""

from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace

import numpy as np

from policy import ARMS, GUIDED_ARMS, DecisionCache, choose_actions


class TinyTours:
    @staticmethod
    def is_valid_tour(tour, n):
        return len(tour) == n and sorted(map(int, tour)) == list(range(n))

    @staticmethod
    def tour_succ_pred(tour):
        n = len(tour)
        succ = np.empty(n, dtype=np.int32)
        pred = np.empty(n, dtype=np.int32)
        succ[tour] = np.roll(tour, -1)
        pred[tour] = np.roll(tour, 1)
        return succ, pred

    @staticmethod
    def tour_length(xy, tour, metric):
        assert metric == "EUC_2D"
        delta = xy[tour] - xy[np.roll(tour, -1)]
        return int(np.floor(np.sqrt((delta * delta).sum(axis=1)) + 0.5).sum())


class TinyModel:
    def __init__(self, graph, probability):
        self.graph = graph
        self.probability = probability
        self.calls = 0

    def predict_tourfold(self, model, xy, graph_k, device, **kwargs):
        self.calls += 1
        assert len(xy) == len(self.graph)
        assert graph_k == self.graph.shape[1]
        assert kwargs["ens_succ"].shape == (len(xy), 6)
        assert kwargs["ens_cost"].shape == (6,)
        distance = np.tile(np.arange(graph_k, dtype=np.float64), (len(xy), 1))
        return {"NB": self.graph, "prob": self.probability.copy(),
                "D": distance, "t_total": 0.01}


class NoisyModel(TinyModel):
    def predict_tourfold(self, model, xy, graph_k, device, **kwargs):
        result = super().predict_tourfold(model, xy, graph_k, device, **kwargs)
        noise = 1e-12 * self.calls * np.linspace(0.0, 1.0, self.probability.size)
        result["prob"] = np.clip(result["prob"] + noise.reshape(self.probability.shape), 0, 1)
        return result


def fixtures(n=48, full_rows=True, alternating_tour=False):
    xy = np.column_stack((np.arange(n), np.arange(n) % 3)).astype(np.int32)
    steps = (1, n - 1, 2, 3, 4, 5, 6, 7, 8, 9)
    graph = np.stack([(np.arange(n) + step) % n for step in steps], axis=1).astype(np.int32)
    probability = np.tile(np.array([0.2, 0.2, 0.05, 0.1, 0.15,
                                    0.75, 0.95, 0.3, 0.25, 0.22]), (n, 1))
    snapshots = []
    for i, seed in enumerate(range(301, 307)):
        if alternating_tour and i == 1:
            tour = np.r_[np.arange(0, n, 2), np.arange(1, n, 2)]
        else:
            tour = np.arange(n)
        width = (1, n - 1, 2, 3, 4) if full_rows else (1, n - 1)
        rows = [[(int((source + step) % n) + 1, 10 * slot)
                 for slot, step in enumerate(width)]
                for source in range(n)]
        snapshots.append(SimpleNamespace(seed=seed,
                                         first_tour=(tour + 1).tolist(),
                                         live_candidates=rows,
                                         native_candidates=rows))
    return xy, graph, probability, snapshots


def _undirected_live_edges(rows):
    return {(min(i, to - 1), max(i, to - 1))
            for i, row in enumerate(rows) for to, _alpha in row}


def _first_tour_edges(snapshots):
    edges = set()
    for snapshot in snapshots:
        tour = np.asarray(snapshot.first_tour) - 1
        edges.update((min(int(a), int(b)), max(int(a), int(b)))
                     for a, b in zip(tour, np.roll(tour, -1)))
    return edges


def check_swaps(alternating_tour=False):
    xy, graph, probability, snapshots = fixtures(alternating_tour=alternating_tour)
    model = TinyModel(graph, probability)
    runs = {arm: choose_actions(arm, snapshots, xy, None, None,
                                TinyTours, model, fraction=0.05, graph_k=10)
            for arm in ARMS}
    assert model.calls == len(ARMS), "sham must pay for inference"
    audits = {arm: audit for arm, (_actions, audit) in runs.items()}
    base = audits["beta_swap"]
    assert base["guided_actions_per_seed"] == [2] * 6
    assert base["expected_directed_row_length_delta_per_seed"] == [0] * 6
    assert all(audits[arm]["gate_sha256_per_seed"] == base["gate_sha256_per_seed"]
               for arm in ARMS)
    assert all(audits[arm]["victim_sha256_per_seed"] == base["victim_sha256_per_seed"]
               for arm in ARMS)
    assert all(audits[arm]["all_guided_action_sha256_per_seed"]
               == base["all_guided_action_sha256_per_seed"] for arm in ARMS)
    assert all(all(to == 0 and victim == 0 for to, victim, _alpha in row)
               for row in runs["sham"][0])
    protected = _first_tour_edges(snapshots)
    for arm in GUIDED_ARMS:
        actions, audit = runs[arm]
        assert audit["selected_actions_per_seed"] == [2] * 6
        assert audit["selected_unique_undirected_edges_per_seed"] == [2] * 6
        json.dumps(audit, allow_nan=False)
        for snapshot, gate, rows in zip(snapshots, audit["gate_nodes_per_seed"], actions):
            existing = _undirected_live_edges(snapshot.live_candidates)
            inserted = []
            for source, (to, victim, alpha) in enumerate(rows):
                if source not in gate:
                    assert (to, victim, alpha) == (0, 0, 500)
                    continue
                assert alpha == 500
                row = snapshot.live_candidates[source]
                victim_slot = [t for t, (id_, _a) in enumerate(row) if id_ == victim]
                assert len(victim_slot) == 1 and victim_slot[0] >= 2
                old_edge = (min(source, victim - 1), max(source, victim - 1))
                new_edge = (min(source, to - 1), max(source, to - 1))
                assert old_edge not in protected
                assert new_edge not in protected
                assert new_edge not in existing
                if arm == "beta_swap":
                    to_col = np.flatnonzero(graph[source] == to - 1)
                    victim_col = np.flatnonzero(graph[source] == victim - 1)
                    assert len(to_col) == len(victim_col) == 1
                    assert probability[source, to_col[0]] > probability[source, victim_col[0]]
                inserted.append(new_edge)
                replaced = list(row)
                replaced[victim_slot[0]] = (to, alpha)
                assert len(replaced) == len(row)
            assert len(inserted) == len(set(inserted)) == len(gate)
    beta_targets = [[to for to, _, _ in row if to] for row in runs["beta_swap"][0]]
    near_targets = [[to for to, _, _ in row if to] for row in runs["nearest_swap"][0]]
    assert beta_targets != near_targets, "learned and nearest policies should differ in fixture"
    repeated = choose_actions("random_swap", snapshots, xy, None, None,
                              TinyTours, model, fraction=0.05, graph_k=10)
    assert repeated[0] == runs["random_swap"][0]


def check_protection_blocks_all_victims():
    xy, graph, probability, snapshots = fixtures(alternating_tour=True)
    actions, audit = choose_actions("beta_swap", snapshots, xy, None, None,
                                    TinyTours, TinyModel(graph, probability),
                                    fraction=0.05, graph_k=10)
    protected = _first_tour_edges(snapshots)
    for row in actions:
        for source, (_to, victim, _alpha) in enumerate(row):
            if victim:
                assert (min(source, victim - 1), max(source, victim - 1)) not in protected
    assert sum(audit["guided_actions_per_seed"]) > 0


def check_empty_gate():
    xy, graph, probability, snapshots = fixtures(full_rows=False)
    for arm in ARMS:
        actions, audit = choose_actions(arm, snapshots, xy, None, None,
                                        TinyTours, TinyModel(graph, probability),
                                        fraction=0.05, graph_k=10)
        assert audit["guided_actions_per_seed"] == [0] * 6
        assert all(all(to == 0 for to, _victim, _alpha in row) for row in actions)


def check_decision_cache():
    xy, graph, probability, snapshots = fixtures()
    model = NoisyModel(graph, probability)
    with tempfile.TemporaryDirectory() as temp:
        cache = DecisionCache(temp)
        first = choose_actions("beta_swap", snapshots, xy, None, None,
                               TinyTours, model, fraction=0.05, graph_k=10,
                               decision_cache=cache, model_identity="checkpoint")[1]
        assert first["decision_basis"] == "fresh_inference"
        assert cache.flush() == [first["decision_cache_key"] + ".npz"]
        reloaded = DecisionCache(temp)
        later = {arm: choose_actions(arm, snapshots, xy, None, None,
                                     TinyTours, model, fraction=0.05, graph_k=10,
                                     decision_cache=reloaded, model_identity="checkpoint")[1]
                 for arm in ARMS}
        assert model.calls == len(ARMS) + 1
        for audit in later.values():
            assert audit["decision_basis"] == "cached_inference"
            assert audit["model_probability_sha256"] == first["model_probability_sha256"]
            assert audit["fresh_model_probability_sha256"] != first["model_probability_sha256"]
            assert audit["gate_sha256_per_seed"] == first["gate_sha256_per_seed"]
            assert audit["victim_sha256_per_seed"] == first["victim_sha256_per_seed"]
            assert (audit["all_guided_action_sha256_per_seed"]
                    == first["all_guided_action_sha256_per_seed"])


def main():
    check_swaps()
    check_swaps(alternating_tour=True)
    check_protection_blocks_all_victims()
    check_empty_gate()
    check_decision_cache()
    print("v15 matched live-swap policy checks passed")


if __name__ == "__main__":
    main()
