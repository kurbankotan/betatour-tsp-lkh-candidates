"""Local smoke test for the exact-source v11 search telemetry patch.

Usage: python test_search_hook.py --source-dir LKH-3.0.14 --binary PATCHED/LKH
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

from build_search_hook import MOVE_LOOP_COUNTS, UPSTREAM_SHA256, patch_findtour, patch_move_file, patch_storetour

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "livehook"))
from coordinator import noop_actions, run_six
from pilot import _verify_applied_actions


def source_checks(source_dir: Path) -> None:
    src = source_dir / "SRC"
    patched = patch_findtour((src / "FindTour.c").read_text())
    assert "BT_AfterFirstTrial()" in patched
    for name, count in MOVE_LOOP_COUNTS.items():
        patched = patch_move_file((src / name).read_text(), name)
        assert patched.count("BT_NoteCandidate(") == count
    patched = patch_storetour((src / "StoreTour.c").read_text())
    assert patched.count("BT_NoteAcceptedEdge(t)") == 1
    assert len(UPSTREAM_SHA256) == 11


def run_smoke(binary: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="betatour_v11_test_") as temp:
        root = Path(temp)
        rng = random.Random(117)
        coords = [(rng.randrange(10000), rng.randrange(10000)) for _ in range(180)]
        problem = root / "smoke.tsp"
        problem.write_text(
            "NAME : smoke\nTYPE : TSP\nDIMENSION : 180\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n"
            + "".join(f"{i + 1} {x} {y}\n" for i, (x, y) in enumerate(coords))
            + "EOF\n")

        proposed = []

        def actions(snapshots):
            output = []
            for snapshot in snapshots:
                rows = []
                for node, candidates in enumerate(snapshot.live_candidates, 1):
                    if node % 5:
                        rows.append((0, 0, 500))
                        continue
                    native = {to for to, _ in candidates}
                    to = next(v for v in range(1, 181) if v != node and v not in native)
                    rows.append((to, 3, 500))
                output.append(rows)
            proposed[:] = output
            return output

        result = run_six(binary, problem, root / "actions", list(range(301, 307)),
                         8.0, actions, max_trials=8,
                         extra_params={"MAX_CANDIDATES": 6})
        assert result["barrier_completed"]
        _verify_applied_actions(root / "actions", list(range(301, 307)), proposed)
        for seed in range(301, 307):
            seed_dir = root / "actions" / f"seed_{seed}"
            telemetry = dict(line.split() for line in
                             (seed_dir / "telemetry.txt").read_text().splitlines())
            assert int(telemetry["action_count"]) == 36
            assert int(telemetry["action_candidate_enumerations"]) > 0
            assert (0 <= int(telemetry["top_level_action_positive_g1"])
                    <= int(telemetry["top_level_action_first_gate_passes"])
                    <= int(telemetry["top_level_action_enumerations"]))
            assert len((seed_dir / "action_telemetry.tsv").read_text().splitlines()) == 37
            assert (seed_dir / "applied.stats").exists()
            assert (seed_dir / "best.tour").exists()

        sham = run_six(binary, problem, root / "sham", list(range(301, 307)),
                       8.0, lambda snapshots: [noop_actions(180) for _ in snapshots],
                       max_trials=4, extra_params={"MAX_CANDIDATES": 6})
        assert sham["barrier_completed"]
        for seed in range(301, 307):
            text = (root / "sham" / f"seed_{seed}" / "telemetry.txt").read_text()
            telemetry = dict(line.split() for line in text.splitlines())
            assert int(telemetry["action_count"]) == 0
            assert int(telemetry["action_candidate_enumerations"]) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    source_checks(args.source_dir)
    run_smoke(args.binary)
    print("v11 search hook smoke passed")


if __name__ == "__main__":
    main()
