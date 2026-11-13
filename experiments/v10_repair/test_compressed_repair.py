"""Focused invariants for tour compression and the exact EUC_2D objective."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from compressed_repair import compress_tour, free_windows, repair_once  # noqa: E402
from repair_engine import tour_cost  # noqa: E402
import tsplab as T  # noqa: E402


def edges(tour: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted((int(tour[i]), int(tour[(i + 1) % len(tour)]))))
            for i in range(len(tour))}


class CompressionTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(721)
        self.xy = rng.integers(0, 10000, size=(100, 2), dtype=np.int64)
        self.tour = rng.permutation(100).astype(np.int32)

    def test_initial_tour_is_exactly_preserved_up_to_rotation(self) -> None:
        free = free_windows(100, np.array([2, 31, 65]), 12)
        packed = compress_tour(self.xy, self.tour, free)
        restored = packed.expand(packed.initial_tour)
        self.assertEqual(edges(restored), edges(self.tour))
        self.assertEqual(tour_cost(self.xy, restored), tour_cost(self.xy, self.tour))
        self.assertEqual(tour_cost(self.xy, restored),
                         int(T.tour_length(self.xy, restored, "EUC_2D")))

    def test_mandatory_edge_expansion_allows_reordering(self) -> None:
        free = free_windows(100, np.array([2, 31, 65]), 12)
        packed = compress_tour(self.xy, self.tour, free)
        route = packed.initial_tour.copy()
        # Reversing the entire reduced cycle retains every mandatory pair.
        route = route[::-1].copy()
        restored = packed.expand(route)
        self.assertEqual(len(np.unique(restored)), 100)
        self.assertEqual(edges(restored), edges(self.tour))
        self.assertEqual(tour_cost(self.xy, restored), tour_cost(self.xy, self.tour))

    def test_missing_mandatory_pair_is_rejected(self) -> None:
        free = free_windows(100, np.array([2, 31, 65]), 12)
        packed = compress_tour(self.xy, self.tour, free)
        a, b = map(int, packed.fixed_edges[0])
        route = packed.initial_tour.copy()
        ia, ib = int(np.where(route == a)[0][0]), int(np.where(route == b)[0][0])
        swap = (ib + 2) % len(route)
        route[ib], route[swap] = route[swap], route[ib]
        with self.assertRaises(ValueError):
            packed.expand(route)

    @unittest.skipUnless(os.environ.get("BETATOUR_TEST_LKH"), "set BETATOUR_TEST_LKH for LKH smoke")
    def test_small_lkh_repair_never_worsens(self) -> None:
        binary = Path(os.environ["BETATOUR_TEST_LKH"])
        result = repair_once(self.xy, self.tour, binary, Path("/private/tmp/betatour-v10-unittest"),
                             strategy="random", ensemble_tours=None, model_scores=None,
                             rng=np.random.default_rng(2), cut_count=3, window_size=12,
                             timeout_s=2.0, max_trials=30)
        self.assertLessEqual(result.after, result.before)
        self.assertEqual(result.after, tour_cost(self.xy, result.tour))


if __name__ == "__main__":
    unittest.main()
