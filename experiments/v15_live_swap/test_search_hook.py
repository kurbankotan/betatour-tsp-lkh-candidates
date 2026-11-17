"""Real LKH-3.0.14 smoke of the v15 first-trial swap hook.

Usage: python test_search_hook.py --source-dir LKH-3.0.14 --binary PATCHED/LKH
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

from build_search_hook import (
    MOVE_LOOP_COUNTS, UPSTREAM_SHA256, patch_findtour, patch_move_file,
    patch_storetour,
)


def source_checks(source_dir: Path) -> None:
    src = source_dir / "SRC"
    patched = patch_findtour((src / "FindTour.c").read_text())
    assert '#include "BetaTourSwapHook.h"' in patched
    assert "BT_AfterFirstTrial()" in patched
    for name, count in MOVE_LOOP_COUNTS.items():
        patched = patch_move_file((src / name).read_text(), name)
        assert patched.count("BT_NoteCandidate(") == count
    patched = patch_storetour((src / "StoreTour.c").read_text())
    assert patched.count("BT_NoteAcceptedEdge(t)") == 1
    assert len(UPSTREAM_SHA256) == 11


def _read_tour(path: Path) -> list[int]:
    fields = [int(x) for x in path.read_text().split()]
    assert len(fields) == fields[0] + 1
    tour = fields[1:]
    assert sorted(tour) == list(range(1, len(tour) + 1))
    return tour


def _read_rows(path: Path) -> list[list[tuple[int, int]]]:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    assert len(lines) == n + 1
    rows = []
    for source, line in enumerate(lines[1:], 1):
        fields = [int(x) for x in line.split()]
        assert fields[0] == source and len(fields) == 2 + 2 * fields[1]
        rows.append([(fields[i], fields[i + 1]) for i in range(2, len(fields), 2)])
    return rows


def _edges(tour: list[int]) -> set[tuple[int, int]]:
    return {tuple(sorted((a, b))) for a, b in zip(tour, tour[1:] + tour[:1])}


def run_smoke(binary: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="betatour_v15_real_") as temp:
        root = Path(temp)
        hook = root / "hook"
        hook.mkdir()
        n = 180
        rng = random.Random(117)
        coords = [(rng.randrange(10000), rng.randrange(10000)) for _ in range(n)]
        problem = root / "smoke.tsp"
        problem.write_text(
            "NAME : smoke\nTYPE : TSP\nDIMENSION : 180\n"
            "EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n"
            + "".join(f"{i + 1} {x} {y}\n" for i, (x, y) in enumerate(coords))
            + "EOF\n")
        params = hook / "run.par"
        params.write_text(
            f"PROBLEM_FILE = {problem}\nCANDIDATE_SET_TYPE = POPMUSIC\n"
            "MAX_CANDIDATES = 6\nSUBGRADIENT = NO\nSEED = 301\n"
            "RUNS = 1\nMAX_TRIALS = 100\nTIME_LIMIT = 100\n"
            "TRACE_LEVEL = 0\nSTOP_AT_OPTIMUM = NO\n")
        env = dict(os.environ, BETATOUR_HOOK_DIR=str(hook),
                   BETATOUR_HOOK_TIMEOUT_S="12")
        with (hook / "stdout.txt").open("w") as out, (hook / "stderr.txt").open("w") as err:
            process = subprocess.Popen([str(binary.resolve()), str(params)],
                                       cwd=hook, stdout=out, stderr=err, env=env)
            try:
                deadline = time.monotonic() + 12
                while not (hook / "first.tour").exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("LKH exited before barrier: "
                                           + (hook / "stderr.txt").read_text())
                    time.sleep(0.02)
                assert (hook / "first.tour").exists(), "first-trial barrier timeout"
                tour = _read_tour(hook / "first.tour")
                live = _read_rows(hook / "live.candidates")
                protected = _edges(tour)
                graph = {tuple(sorted((u, v))) for u, row in enumerate(live, 1)
                         for v, _alpha in row}
                action = None
                for u, row in enumerate(live, 1):
                    for victim, _alpha in row[2:]:
                        if tuple(sorted((u, victim))) in protected:
                            continue
                        for v in range(1, n + 1):
                            if v != u and tuple(sorted((u, v))) not in graph:
                                action = (u, v, victim)
                                break
                        if action:
                            break
                    if action:
                        break
                assert action is not None, "no safe smoke action"
                edge_lines = [f"{a} {b}" for a, b in sorted(protected)]
                (hook / "protected.edges").write_text(
                    f"{n} {len(edge_lines)}\n" + "\n".join(edge_lines) + "\n")
                u, v, victim = action
                action_lines = [str(n)] + [
                    f"{source} {v} {victim} 500" if source == u else
                    f"{source} 0 0 0" for source in range(1, n + 1)]
                temporary = hook / "action.txt.tmp"
                temporary.write_text("\n".join(action_lines) + "\n")
                os.replace(temporary, hook / "action.txt")
                while not (hook / "post.candidates").exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("LKH exited at swap: "
                                           + (hook / "stderr.txt").read_text())
                    time.sleep(0.02)
                assert (hook / "post.candidates").exists(), "post-swap timeout"
                post = _read_rows(hook / "post.candidates")
                expected = list(live[u - 1])
                index = next(i for i, (to, _alpha) in enumerate(expected)
                             if to == victim)
                expected[index] = (v, 500)
                assert post[u - 1] == expected
                assert all(post[i] == live[i] for i in range(n) if i != u - 1)
                stats = dict(line.split() for line in
                             (hook / "applied.stats").read_text().splitlines())
                assert stats["swapped"] == "1" and stats["rejected"] == "0"
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    source_checks(args.source_dir)
    run_smoke(args.binary)
    print("v15 real LKH swap smoke passed")


if __name__ == "__main__":
    main()
