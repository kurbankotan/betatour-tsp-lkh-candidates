"""Compile and run the mock-LKH unit test of the v11 search hook.

Usage: python test_hook_logic.py   (needs gcc; no LKH sources are required)

The C driver applies three slot-3 actions, imitates two LKH row reorderings
between trials and logs a better tour. This script checks the files that the
hook wrote. It tests hook logic only; the Colab notebook still runs the real
patched LKH-3.0.14 binary before the pilot starts.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def key_values(path: Path) -> dict[str, float]:
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        assert key not in values, f"duplicate telemetry key {key}"
        values[key] = float(value)
    return values


def candidate_rows(path: Path) -> list[list[tuple[int, int]]]:
    lines = path.read_text().splitlines()
    rows = []
    for expected, line in enumerate(lines[1:], 1):
        numbers = [int(x) for x in line.split()]
        assert numbers[0] == expected and len(numbers) == 2 + 2 * numbers[1]
        rows.append([(numbers[i], numbers[i + 1]) for i in range(2, len(numbers), 2)])
    assert int(lines[0]) == len(rows)
    return rows


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="betatour_hook_logic_") as temp:
        root = Path(temp)
        binary = root / "hook_logic_test"
        compile_command = ["gcc", "-std=c99", "-D_DEFAULT_SOURCE", "-Wall", "-Wextra",
                           "-Werror", "-Wno-unused-function", "-g", "-O1",
                           "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                           "-I", str(HERE), "-I", str(HERE.parent),
                           str(HERE / "hook_logic_test.c"), "-o", str(binary)]
        subprocess.run(compile_command, check=True)
        hook_dir = root / "hook"
        hook_dir.mkdir()
        run = subprocess.run([str(binary), str(hook_dir)], check=True,
                             capture_output=True, text=True)
        assert "hook logic test passed" in run.stdout, run.stdout + run.stderr

        native = candidate_rows(hook_dir / "native.candidates")
        live = candidate_rows(hook_dir / "live.candidates")
        post = candidate_rows(hook_dir / "post.candidates")
        assert native == live and all(len(row) == 6 for row in native)
        for source, action in ((1, 7), (5, 11), (9, 3)):
            expected = list(live[source - 1])
            expected.insert(2, (action, 500))
            assert post[source - 1] == expected
        assert all(post[i] == live[i] for i in range(12) if i + 1 not in (1, 5, 9))

        stats = key_values(hook_dir / "applied.stats")
        assert (stats["added"], stats["existing"], stats["moved"], stats["rejected"]) == (3, 0, 3, 0)
        events = [line.split() for line in (hook_dir / "best.events").read_text().splitlines()]
        assert [int(cost) for _epoch, cost in events] == [1000, 990]
        tour = [int(x) for x in (hook_dir / "best.tour").read_text().split()]
        assert tour == [12, 1, 7, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]

        telemetry = key_values(hook_dir / "telemetry.txt")
        expected = {"action_count": 3, "total_candidate_enumerations": 3,
                    "action_candidate_enumerations": 2,
                    "top_level_action_first_gate_passes": 1,
                    "top_level_action_positive_g1": 1,
                    "action_accepted_edge_entries": 1,
                    "action_incumbent_entries": 1,
                    "better_tour_events_after_action": 1,
                    "trials_completed_after_action": 3,
                    "slot_floor_enforced": 1, "slot_checks": 3,
                    "slot_restorations": 4, "slot_restoration_displacement": 14,
                    "slot_found_ahead": 2, "slot_ahead_at_last_check": 1,
                    "slot_missing": 0, "action_tsv_snapshot_trial": 4}
        for key, value in expected.items():
            assert telemetry[key] == value, (key, telemetry[key], value)

        lines = (hook_dir / "action_telemetry.tsv").read_text().splitlines()
        header = lines[0].split("\t")
        assert header[-1] == "slot_restorations" and len(header) == 10
        rows = {int(r.split("\t")[0]): [int(x) for x in r.split("\t")] for r in lines[1:]}
        assert sorted(rows) == [1, 5, 9]
        assert rows[1] == [1, 7, 2, 1, 1, 1, 1, 1, 1, 2]
        assert rows[5][-1] == 1 and rows[9][-1] == 1
        assert not list(hook_dir.glob("*.tmp.*")), "temporary files left behind"
    print("v11 hook logic test passed")


if __name__ == "__main__":
    main()
