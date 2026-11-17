"""A slow policy must leave an explicit, auditable failed run."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from coordinator import run_six


FAKE_LKH = """#!/usr/bin/env python3
import os
import time
from pathlib import Path

base = Path(os.environ['BETATOUR_HOOK_DIR'])
n = 6
rows = [str(n)]
for source in range(1, n + 1):
    rows.append(f'{source} 2 {source % n + 1} 0 {(source + 1) % n + 1} 100')
data = '\\n'.join(rows) + '\\n'
(base / 'native.candidates').write_text(data)
(base / 'live.candidates').write_text(data)
(base / 'first.tour').write_text('6\\n1\\n2\\n3\\n4\\n5\\n6\\n')
while not (base / 'action.txt').exists():
    time.sleep(0.01)
(base / 'post.candidates').write_text(data)
(base / 'applied.stats').write_text('swapped 0\\nrejected 0\\n')
while True:
    time.sleep(0.05)
"""


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="betatour_v15_deadline_") as temp:
        root = Path(temp)
        binary = root / "fake_lkh.py"
        binary.write_text(FAKE_LKH)
        binary.chmod(0o755)
        problem = root / "dummy.tsp"
        problem.write_text("unused\n")
        seeds = list(range(301, 307))
        work = root / "slow_policy"

        def slow_policy(snapshots):
            assert len(snapshots) == 6
            time.sleep(3.1)
            return [[(0, 0, 500)] * 6 for _ in snapshots]

        result = run_six(binary, problem, work, seeds, 3.0, slow_policy)
        saved = json.loads((work / "run_summary.json").read_text())
        assert result == saved
        assert result["barrier_completed"] is True
        assert result["failure_reason"] == "deadline_before_actions", result
        assert result["phase_timings_s"]["policy_end"] >= 3.0
        assert result["remaining_at_policy_end_s"] < 0
        assert result["actions_published_seeds"] == []
        assert result["resume_completed_seeds"] == []
        assert all(not run["action_exists"] and not run["applied_exists"]
                   for run in result["runs"])

        completed = root / "fast_policy"
        result = run_six(binary, problem, completed, seeds, 1.5,
                         lambda snapshots: [[(0, 0, 500)] * 6 for _ in snapshots])
        assert result["failure_reason"] is None, result
        assert result["actions_published_seeds"] == seeds
        assert result["resume_completed_seeds"] == seeds
        assert all(run["action_exists"] and run["post_exists"] and
                   run["applied_exists"] for run in result["runs"])

        failed = root / "policy_exception"

        def bad_policy(_snapshots):
            raise ValueError("deliberate policy failure")

        try:
            run_six(binary, problem, failed, seeds, 3.0, bad_policy)
        except ValueError as exc:
            assert "deliberate" in str(exc)
        else:
            raise AssertionError("callback exception was swallowed")
        saved = json.loads((failed / "run_summary.json").read_text())
        assert saved["failure_reason"] == "policy_exception"
        assert "deliberate" in saved["exception"]
        assert saved["barrier_completed"]
    print("v15 coordinator deadline diagnostics passed")


if __name__ == "__main__":
    main()
