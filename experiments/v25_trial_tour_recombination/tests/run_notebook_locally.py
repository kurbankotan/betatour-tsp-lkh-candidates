"""Run the code cells of notebook 27 in one Python process, outside Colab, with the real LKH-3 and GA-EAX.

    BETATOUR_V25_TEST=1 BETATOUR_V25_REPO=<repository> BETATOUR_V25_LKH=<LKH binary inside its source tree> \\
    BETATOUR_V25_EAX_SRC=<clone of GA-EAX> BETATOUR_V25_WORK=<work dir> \\
        python experiments/v25_trial_tour_recombination/tests/run_notebook_locally.py [--notebook PATH] [...]

In test mode the notebook uses two LKH processes, three small instances (1,200 and 2,500 synthetic nodes and a
TSPLIB-format file of 900 nodes with decimal coordinates), budgets of 10 to 15 s, five tours per process, twelve
first-trial tours and a population of 30 for the reference (see its first cell; BETATOUR_V25_W, _N and _BUDGET
change the first instance). The %%writefile cell is executed as IPython would: the cell body is written to the
named file unchanged.

--stop-after N   stops before instance N + 1 starts (a session that ends between two instances).
--fail-step S    raises inside the first instance when that step is reached (S is "merges" or "cascade"; a session
                 lost in the middle of an instance). A second call in the same work directory must continue that
                 instance from its stage files; after BETATOUR_V25_WORK/runtime_session_id.txt has been deleted
                 (a new machine) it must run the instance again from the start.
--oversubscribe  lets the test run more processes than the machine has CPUs (the set-up cell refuses this in a
                 real run). Times are then not meaningful.
--decide         runs the analysis cell as a real run would (the decision rule is evaluated on the test instances);
                 only to execute that code path, the outcome means nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parent.parent.parent / "notebooks" / "27_BetaTour_v25_trial_tour_recombination.ipynb"


class Interrupted(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=DEFAULT)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--fail-step", choices=["merges", "cascade"], default=None)
    parser.add_argument("--oversubscribe", action="store_true")
    parser.add_argument("--decide", action="store_true")
    args = parser.parse_args()
    assert os.environ.get("BETATOUR_V25_TEST") == "1", "set BETATOUR_V25_TEST=1"
    work = Path(os.environ["BETATOUR_V25_WORK"])
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)                                    # the notebook writes byprod_v25.py to its working directory
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
        if args.fail_step is not None and "def run_one_instance(" in source:
            target = {"merges": "make_merges", "cascade": "make_cascade"}[args.fail_step]
            source += (f"\n_real_{target} = {target}\n"
                       f"def {target}(*a, **k):\n"
                       f"    raise RuntimeError('simulated crash in {target}')\n")
        if args.decide and source.startswith("# (anahtar, ilk kol, ikinci kol"):
            space["TEST_MODE"] = False                # the analysis cell only; nothing is run after it but the archive
        print(f"[cell {number}] running", flush=True)
        try:
            exec(compile(source, f"<notebook cell {number}>", "exec"), space)
        except Interrupted as exc:
            print("stopped:", exc)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
