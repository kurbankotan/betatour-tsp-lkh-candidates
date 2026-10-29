"""Synthetic invariants for the v11 matched candidate policy."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from policy import ARMS, GUIDED_ARMS, choose_actions


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
        return {"NB": self.graph, "prob": self.probability,
                "D": distance, "t_total": 0.01}


def fixtures(n=24, k=8):
    xy = np.column_stack((np.arange(n), np.arange(n) % 3)).astype(np.int32)
    graph = np.stack([(np.arange(n) + step) % n for step in (1, n - 1, 2, 3, 4, 5, 6, 7)][:k], axis=1).astype(np.int32)
    probability = np.tile(np.array([0.12, 0.08, 0.95, 0.84, 0.72, 0.61, 0.50, 0.39][:k]), (n, 1))
    snapshots = []
    for seed in range(301, 307):
        tour = (np.roll(np.arange(n), seed % n) + 1).tolist()
        candidate_rows = [[((i + 1) % n + 1, 10), ((i - 1) % n + 1, 20)]
                          for i in range(n)]
        snapshots.append(SimpleNamespace(seed=seed, first_tour=tour,
                                         live_candidates=candidate_rows,
                                         native_candidates=candidate_rows))
    return xy, graph, probability, snapshots


def _edge_set(rows):
    n = len(rows)
    return {(min(i, to - 1), max(i, to - 1))
            for i, row in enumerate(rows) for to, _alpha in row}


def main():
    xy, graph, prob, snapshots = fixtures()
    M = TinyModel(graph, prob)
    runs = {arm: choose_actions(arm, snapshots, xy, None, None, TinyTours, M,
                                fraction=0.25, graph_k=8)
            for arm in ARMS}
    assert M.calls == len(ARMS), "sham must pay for model inference too"
    audits = {arm: audit for arm, (_actions, audit) in runs.items()}
    base_gate = audits["beta_add"]["gate_nodes_per_seed"]
    assert all(audits[arm]["gate_nodes_per_seed"] == base_gate for arm in ARMS)
    assert all(audits[arm]["gate_sha256_per_seed"] == audits["beta_add"]["gate_sha256_per_seed"]
               for arm in ARMS)
    for arm in GUIDED_ARMS:
        actions, audit = runs[arm]
        assert audit["guided_actions_per_seed"] == [len(gate) for gate in base_gate]
        assert audit["selected_actions_per_seed"] == [len(gate) for gate in base_gate]
        assert all(count > 0 for count in audit["selected_actions_per_seed"])
        json.dumps(audit, allow_nan=False)
        for snapshot, gate, action_rows in zip(snapshots, base_gate, actions):
            existing = _edge_set(snapshot.live_candidates)
            selected = []
            for source, (to, slot, alpha) in enumerate(action_rows):
                if source not in gate:
                    assert (to, slot, alpha) == (0, 0, 500)
                    continue
                assert slot == 3 and alpha == 500
                destination = to - 1
                assert destination != source
                key = (min(source, destination), max(source, destination))
                assert key not in existing
                selected.append(key)
            assert len(selected) == len(set(selected)) == len(gate)
    assert all(all(to == 0 for to, _slot, _alpha in row) for row in runs["sham"][0])

    # Repeated calls on identical snapshots must yield identical controls.
    repeated = choose_actions("random_add", snapshots, xy, None, None, TinyTours, M,
                              fraction=0.25, graph_k=8)
    assert repeated[0] == runs["random_add"][0]
    assert repeated[1]["all_guided_action_sha256_per_seed"] == audits["random_add"]["all_guided_action_sha256_per_seed"]

    # If the graph offers no novel edge, all guided arms must stay no-op.
    xy2, graph2, prob2, snapshots2 = fixtures(n=24, k=2)
    M2 = TinyModel(graph2, prob2)
    for arm in GUIDED_ARMS:
        actions, audit = choose_actions(arm, snapshots2, xy2, None, None, TinyTours, M2,
                                        fraction=0.25, graph_k=2)
        assert audit["guided_actions_per_seed"] == [0] * 6
        assert all(all(to == 0 for to, _slot, _alpha in row) for row in actions)
    print("v11 matched candidate policy checks passed")


if __name__ == "__main__":
    main()
