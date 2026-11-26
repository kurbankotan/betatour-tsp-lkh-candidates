"""Run the pilot notebook locally in its test mode (small instances, no Drive, no network).

    python tests/run_notebook_locally.py --repo REPO --lkh /path/to/LKH [--work /tmp/cp_work]

Executes the code cells of notebooks/32_Certified_candidate_graph_pilot.ipynb in one namespace with
CERTPILOT_TEST=1: google.colab is replaced by a stub, LKH is the given binary, and Concorde is the stand-in
tests/fake_concorde.py (subtour LP and exact MILP; not Concorde). The BetaTour model runs on the CPU.
"""
import argparse
import json
import os
import shutil
import stat
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="repository root (src/, checkpoints/, notebooks/)")
    ap.add_argument("--lkh", required=True, help="LKH-3.0.14 binary")
    ap.add_argument("--work", default="/tmp/certpilot_local")
    ap.add_argument("--keep", action="store_true", help="keep an earlier run's files (tests resuming)")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    work = Path(args.work).resolve()
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    (work / "cwd").mkdir(parents=True, exist_ok=True)
    wrapper = work / "concorde"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {HERE / 'fake_concorde.py'} \"$@\"\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    os.environ.update(CERTPILOT_TEST="1", CERTPILOT_REPO=str(repo), CERTPILOT_LKH=str(Path(args.lkh).resolve()),
                      CERTPILOT_CONCORDE=str(wrapper), CERTPILOT_WORK=str(work / "work"))
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    drive = types.ModuleType("google.colab.drive")
    drive.mount = lambda *a, **k: print("(drive.mount stub)")
    colab.drive = drive
    google.colab = colab
    sys.modules.update({"google": google, "google.colab": colab, "google.colab.drive": drive})
    notebook = json.loads((repo / "notebooks" / "32_Certified_candidate_graph_pilot.ipynb").read_text())
    os.chdir(work / "cwd")
    namespace = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%%writefile "):
            name, body = source.split("\n", 1)
            Path(name.split()[1]).write_text(body)
            continue
        print(f"\n===== {cell['id']} =====", flush=True)
        exec(compile(source, cell["id"], "exec"), namespace)
    print("\nlocal run finished:", namespace["OUTPUT_ROOT"])


if __name__ == "__main__":
    main()
