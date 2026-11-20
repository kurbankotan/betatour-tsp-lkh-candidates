"""Run the code cells of notebook 24 in one Python process, outside Colab, with the real LKH-3 binary.

    BETATOUR_V22_TEST=1 BETATOUR_V22_REPO=<repository> BETATOUR_V22_LKH=<LKH binary> BETATOUR_V22_WORK=<work dir> \\
        python experiments/v22_other_sizes/tests/run_notebook_locally.py [--notebook PATH] [--stop-after N]

In test mode the notebook uses two seeds, two small synthetic instances and two generated TSPLIB-format files, one of
them with CEIL_2D distances (see its first cell). The %%writefile cell is executed as IPython would: the cell body is
written to the named file unchanged.

--stop-after N   stops before instance N + 1 starts (a session that ends between two instances).
--fail-arm ARM   raises inside the first instance when that restarted arm is reached (a session lost in the middle of
                 an instance). A second call in the same work directory must continue that instance from its stage
                 files; after BETATOUR_V22_WORK/runtime_session_id.txt has been deleted (a new machine) it must run
                 the instance again from the start.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parent.parent.parent / "notebooks" / "24_BetaTour_v22_corrected_files_other_sizes.ipynb"


class Interrupted(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=DEFAULT)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--fail-arm", default=None)
    args = parser.parse_args()
    assert os.environ.get("BETATOUR_V22_TEST") == "1", "set BETATOUR_V22_TEST=1"
    work = Path(os.environ["BETATOUR_V22_WORK"])
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
