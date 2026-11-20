"""Run v23: restart markings. Helper functions embedded in notebook 25.

LKH-3 starts every trial after the first from a tour built by a random walk (ChooseInitialTour.c). The walk follows
an edge of the best or next best tour by rule only if that edge is a candidate whose alpha-value is zero; elsewhere
it moves to a random unvisited candidate. With restricted search (the default), a move of the trial can start only at
an edge that is not in the best tour. The entries with alpha = 0 in a candidate file therefore decide which edges of
the best tour every trial keeps and where its search starts.

LKH-3 itself gives alpha = 0 to the edges of its minimum 1-tree. This module writes other markings on given candidate
lists:

  * count marking: alpha = 0 on the entries whose edge is in at least a given number of the ensemble tours, and
  * model marking: alpha = 0 on the entries whose edge has a predicted probability of at least a threshold; the
    threshold is set so that the marking trusts as many edges of a given tour as the count marking does.

The lists themselves and the fathers (Dad) of the file are not changed by a marking.
"""

from __future__ import annotations

import numpy as np

VERSION = "v23-marking-1"


def edge_keys(a, b, n):
    """Undirected edge keys min * n + max for 0-based node arrays a and b (as alphaenc.edge_keys)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return np.minimum(a, b) * int(n) + np.maximum(a, b)


def entry_keys(cand):
    """Edge key of every entry of the lists `cand` ((n, K) 0-based neighbour ids, -1 padded); -1 for empty entries."""
    cand = np.asarray(cand, dtype=np.int64)
    assert cand.ndim == 2
    n = cand.shape[0]
    rows = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], cand.shape)
    valid = cand >= 0
    return np.where(valid, edge_keys(rows, np.where(valid, cand, 0), n), -1)


def tour_edge_keys(tour):
    """Sorted keys of the n edges of a tour given as a permutation of 0..n-1."""
    tour = np.asarray(tour, dtype=np.int64)
    return np.unique(edge_keys(tour, np.roll(tour, -1), len(tour)))


def key_count(keys, tours):
    """For every edge key, the number of the given tours that contain the edge (0 for the key -1)."""
    keys = np.asarray(keys, dtype=np.int64)
    count = np.zeros(keys.shape, dtype=np.int64)
    for tour in tours:
        count += np.isin(keys, tour_edge_keys(tour))
    return count


def ensemble_count(cand, tours):
    """For every entry of `cand`, the number of the given tours that contain its edge (0 for empty entries)."""
    return key_count(entry_keys(cand), tours)


def probability_table(prob, nb):
    """Undirected edge probabilities from the directed output of the model.

    prob[i, j] is the predicted probability of the edge from node i to its graph neighbour nb[i, j]. The probability
    of an undirected edge is the maximum over the directions in which the graph holds it. Returns the sorted edge
    keys and their probabilities."""
    nb = np.asarray(nb, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    assert nb.ndim == 2 and prob.shape == nb.shape and np.all(nb >= 0)
    n, k = nb.shape
    keys = edge_keys(np.repeat(np.arange(n, dtype=np.int64), k), nb.reshape(-1), n)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    unique, start = np.unique(keys, return_index=True)
    return unique, np.maximum.reduceat(prob.reshape(-1)[order], start)


def key_probability(keys, table):
    """Probability of every edge key from probability_table(); 0 for the key -1 and for edges that the graph of the
    model does not hold."""
    unique, probability = table
    keys = np.asarray(keys, dtype=np.int64)
    pos = np.clip(np.searchsorted(unique, keys), 0, len(unique) - 1)
    return np.where((keys >= 0) & (unique[pos] == keys), probability[pos], 0.0)


def entry_probability(cand, table):
    """Probability of every entry of `cand` (0 for empty entries and for edges outside the graph of the model)."""
    return key_probability(entry_keys(cand), table)


def matched_threshold(tour_probability, keep):
    """Probability threshold that trusts `keep` edges of a tour: the probability of the tour's keep-th most probable
    edge (infinity for keep = 0). Edges with exactly that probability are trusted together."""
    p = np.sort(np.asarray(tour_probability, dtype=np.float64))[::-1]
    keep = int(keep)
    assert p.ndim == 1 and 0 <= keep <= len(p)
    return float("inf") if keep == 0 else float(p[keep - 1])


def trusted_by_probability(probability, threshold):
    """Entries trusted by the model marking: probability at least the threshold; an edge outside the graph of the
    model (probability 0) is never trusted."""
    probability = np.asarray(probability, dtype=np.float64)
    return (probability >= float(threshold)) & (probability > 0.0)


def mark_alpha(cand, trusted, rest=None):
    """Alpha-values of a marking: 0 on the trusted entries of `cand`; the other entries keep their order behind them.

    rest: alpha-values that order the other entries (for example LKH's own; values below 1 are raised to 1). Without
    it the other entries get 100, 200, ... by list position, as in the tree encoding of run v21. LKH sorts each list
    by alpha-value and then by length when it reads the file."""
    cand = np.asarray(cand, dtype=np.int64)
    trusted = np.asarray(trusted, dtype=bool)
    assert cand.ndim == 2 and trusted.shape == cand.shape
    if rest is None:
        rest = np.broadcast_to(100 * (np.arange(cand.shape[1], dtype=np.int64)[None, :] + 1), cand.shape)
    rest = np.asarray(rest, dtype=np.int64)
    assert rest.shape == cand.shape
    return np.where(trusted & (cand >= 0), 0, np.maximum(rest, 1)).astype(np.int64)
