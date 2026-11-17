"""Compile and run the v15 swap hook against a mock LKH state."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _key_values(path: Path) -> dict[str, float]:
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        assert key not in values, key
        values[key] = float(value)
    return values


def _candidate_rows(path: Path) -> list[list[tuple[int, int]]]:
    lines = path.read_text().splitlines()
    rows = []
    for expected, line in enumerate(lines[1:], 1):
        fields = [int(x) for x in line.split()]
        assert fields[0] == expected and len(fields) == 2 + 2 * fields[1]
        rows.append([(fields[i], fields[i + 1]) for i in range(2, len(fields), 2)])
    assert int(lines[0]) == len(rows)
    return rows


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="betatour_v15_hook_") as temp:
        root = Path(temp)
        binary = root / "hook_logic_test"
        command = ["cc", "-std=c99", "-D_DEFAULT_SOURCE", "-Wall", "-Wextra",
                   "-Werror", "-Wno-unused-function", "-g", "-O1",
                   "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                   "-I", str(HERE), "-I", str(HERE.parent),
                   str(HERE / "hook_logic_test.c"), "-o", str(binary)]
        subprocess.run(command, check=True)
        hook_dir = root / "hook"
        hook_dir.mkdir()
        run = subprocess.run([str(binary), str(hook_dir)], capture_output=True,
                             text=True, check=True)
        assert "v15 hook logic test passed" in run.stdout

        native = _candidate_rows(hook_dir / "native.candidates")
        live = _candidate_rows(hook_dir / "live.candidates")
        post = _candidate_rows(hook_dir / "post.candidates")
        assert native == live and all(len(row) == 6 for row in native)
        for source, victim, new in ((1, 3, 7), (5, 7, 11), (9, 11, 3)):
            expected = list(live[source - 1])
            assert expected[2][0] == victim
            expected[2] = (new, 500)
            assert post[source - 1] == expected
        assert all(post[i] == live[i] for i in range(12) if i + 1 not in (1, 5, 9))

        stats = _key_values(hook_dir / "applied.stats")
        assert stats["swapped"] == 3 and stats["rejected"] == 0
        assert stats["row_length_preserved"] == 1
        telemetry = _key_values(hook_dir / "telemetry.txt")
        assert telemetry["action_count"] == 3
        assert telemetry["swap_at_barrier"] == 1
        assert telemetry["slot_floor_enforced"] == 0
        assert telemetry["action_candidate_enumerations"] == 1
        assert telemetry["action_incumbent_entries"] == 1
        header = (hook_dir / "action_telemetry.tsv").read_text().splitlines()[0]
        assert header.startswith("from\tto\tvictim\t")
        assert not list(hook_dir.glob("*.tmp.*"))

        noop_dir = root / "noop"
        noop_dir.mkdir()
        noop = subprocess.run([str(binary), str(noop_dir), "noop"],
                              capture_output=True, text=True, check=True)
        assert "v15 hook no-op test passed" in noop.stdout
        assert _candidate_rows(noop_dir / "live.candidates") == _candidate_rows(
            noop_dir / "post.candidates")
        assert _key_values(noop_dir / "applied.stats")["swapped"] == 0

        omit_dir = root / "omit"
        omit_dir.mkdir()
        omit = subprocess.run([str(binary), str(omit_dir), "omit"],
                              capture_output=True, text=True, check=False)
        assert omit.returncode == 97, omit.stderr
        assert "omit seed tour edge" in omit.stderr
        assert not (omit_dir / "post.candidates").exists()

        reject_dir = root / "reject"
        reject_dir.mkdir()
        reject = subprocess.run([str(binary), str(reject_dir), "reject"],
                                capture_output=True, text=True, check=False)
        assert reject.returncode == 97, reject.stderr
        assert "from=1 to=8 victim=2 alpha=500" in reject.stderr
        assert "victim is a protected first-tour edge" in reject.stderr
        assert not (reject_dir / "post.candidates").exists()
    print("v15 hook logic test passed")


if __name__ == "__main__":
    main()
