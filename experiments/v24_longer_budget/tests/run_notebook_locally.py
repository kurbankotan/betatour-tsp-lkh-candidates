"""Run the code cells of notebook 26 in one Python process, outside Colab, with the real LKH-3 binary.

    BETATOUR_V24_TEST=1 BETATOUR_V24_REPO=<repository> BETATOUR_V24_LKH=<LKH binary> BETATOUR_V24_WORK=<work dir> \\
        python experiments/v24_longer_budget/tests/run_notebook_locally.py [--notebook PATH] [--stop-after N]

In test mode the notebook uses two seeds and two instances of 1,000 nodes with a budget of 6 s (see its first cell;
BETATOUR_V24_W, _N, _PER_FAMILY, _BUDGET and _PAD change these). The %%writefile cell is executed as IPython would:
the cell body is written to the named file unchanged.

--stop-after N   stops before instance N + 1 starts (a session that ends between two instances).
--fail-arm ARM   raises inside the first instance when that restarted arm is reached (a session lost in the middle of
                 an instance). A second call in the same work directory must continue that instance from its stage
                 files; after BETATOUR_V24_WORK/runtime_session_id.txt has been deleted (a new machine) it must run
                 the instance again from the start.
--oversubscribe  lets the test run more LKH processes than the machine has CPUs (the set-up cell refuses this in a
                 real run); for a check of the real number of processes on a small machine. Times are then not
                 meaningful.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parent.parent.parent / "notebooks" / "26_BetaTour_v24_longer_budget_20k.ipynb"


class Interrupted(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=DEFAULT)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--fail-arm", default=None)
    parser.add_argument("--oversubscribe", action="store_true")
    args = parser.parse_args()
    assert os.environ.get("BETATOUR_V24_TEST") == "1", "set BETATOUR_V24_TEST=1"
    work = Path(os.environ["BETATOUR_V24_WORK"])
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)                                    # the notebook writes alphaenc_v21.py to its working directory
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    space = {"__name__": "__main__"}
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    for number, c in enumerate(code_cells, 1):
        source = "".join(c["source"])
        if source.startswith("%%writefile "):
            first, _, body = source.partition("\n")
            if not body.endswith("\n"):               # IPython ends every cell with a newline before it runs it
                body += "\n"
            Path(first.split(None, 1)[1].strip()).write_text(body, encoding="utf-8")
            print(f"[cell {number}] wrote {first.split(None, 1)[1].strip()}", flush=True)
            continue
        if args.oversubscribe and "assert CPU_COUNT >= W," in source:
            assert source.count("assert CPU_COUNT >= W,") == 1
            source = source.replace("assert CPU_COUNT >= W,", "assert CPU_COUNT >= 1,")
        if args.stop_after is not None and "def run_one_instance(" in source:
            source += ("\n_real_run_one_instance = run_one_instance\n_finished = []\n"
                       "def run_one_instance(inst, ordinal):\n"
                       f"    if len(_finished) >= {args.stop_after}:\n"
                       "        raise _Interrupted('simulated loss of the session')\n"
                       "    _finished.append(_real_run_one_instance(inst, ordinal))\n"
                       "    return _finished[-1]\n")
            space["_Interrupted"] = Interrupted
        if args.fail_arm is not None and "def run_one_instance(" in source:
            source += ("\n_real_make_resumed_arm = make_resumed_arm\n"
                       "def make_resumed_arm(inst, phase, phase_meta, cand, alpha, dad, overhead_s, arm_name):\n"
                       f"    if arm_name == {args.fail_arm!r}:\n"
                       "        raise RuntimeError('simulated crash in ' + arm_name)\n"
                       "    return _real_make_resumed_arm(inst, phase, phase_meta, cand, alpha, dad, overhead_s, "
                       "arm_name)\n")
        print(f"[cell {number}] running", flush=True)
        try:
            exec(compile(source, f"<notebook cell {number}>", "exec"), space)
        except Interrupted as exc:
            print("stopped:", exc)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
