"""Run the code cells of notebook 33 in one Python process, outside Colab, with the real LKH-3 (test mode).

    python experiments/v30_time_to_optimum/tests/run_notebook_locally.py --repo REPO --lkh /path/to/LKH [--work DIR]
        [--keep] [--delete-runs K]

The script writes a test specification: small synthetic instances from the generator of the study (named as in the
run), one TSPLIB-format file with decimal coordinates, their optima proven by an exact MILP (HiGHS, through
experiments/certified_pilot/tests/fake_concorde.py) and a pilot folder with the optimal tours (for the coverage
analysis). google.colab is replaced by a stub, the model runs on the CPU and the check of the protocol on GitHub is
off. The %%writefile cells are executed as IPython would: the cell body is written to the named file unchanged.

--keep           keeps the work folder of an earlier call (the notebook must skip everything that is done);
--delete-runs K  with --keep: deletes K run records first, as if a session had ended in the middle of the runs.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SYNTHETIC = [["uni1k", 60, 2, "uniform", 15101, [0, 1]], ["clu1k", 70, 1, "clustered", 15202, [0]],
             ["uni2k", 80, 2, "uniform", 15303, [1]], ["clu2k", 90, 1, "clustered", 15404, [0]]]


def euc2d(xy):
    xy = np.asarray(xy, dtype=np.float64)
    return np.floor(np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) + 0.5).astype(np.int64)


def make_spec(repo, work):
    sys.path[:0] = [str(repo / "src"), str(repo / "experiments" / "certified_pilot" / "tests")]
    import tsplab as T
    from fake_concorde import optimal_tour
    spec = {"synthetic": SYNTHETIC, "optima": {}, "sha256": {}, "tsplib_names": ["test_euc80"],
            "tsplib_files": {}, "tsplib_optima": {}, "cap_per_node": 0.02, "pilot_dir": str(work / "pilot")}
    for group, n, count, family, seed, keep in SYNTHETIC:
        rng = np.random.default_rng(seed)
        for i in range(count):
            xy = T.to_int_coords(T.gen_instance(n, family, rng))
            if i not in keep:
                continue
            name = f"cp1_{group}_{i:02d}"
            path = work / "spec" / f"{name}.tsp"
            path.parent.mkdir(parents=True, exist_ok=True)
            T.write_tsp(path, xy, name=name, ew="EUC_2D")
            spec["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            tour, opt = optimal_tour(euc2d(xy))
            assert T.tour_length(xy, tour) == opt
            spec["optima"][name] = int(opt)
            (work / "pilot" / "instances" / name).mkdir(parents=True, exist_ok=True)
            np.savez(work / "pilot" / "instances" / name / "opt_tour.npz", tour=tour.astype(np.int32))
    xy = np.random.default_rng(153).random((80, 2)) * 5000.0
    path = work / "spec" / "test_euc80.tsp"
    with open(path, "w") as f:
        f.write("NAME : test_euc80\nTYPE : TSP\nDIMENSION : 80\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
        f.writelines(f"{i} {x:.3f} {y:.3f}\n" for i, (x, y) in enumerate(xy, 1))
        f.write("EOF\n")
    data = T.read_tsplib(str(path))
    tour, opt = optimal_tour(euc2d(data["xy"]))
    spec["tsplib_files"]["test_euc80"] = str(path)
    spec["tsplib_optima"]["test_euc80"] = int(opt)
    (work / "spec.json").write_text(json.dumps(spec, indent=1))
    return work / "spec.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--lkh", required=True)
    ap.add_argument("--work", default="/tmp/v30_local")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--delete-runs", type=int, default=0)
    args = ap.parse_args()
    repo, work = Path(args.repo).resolve(), Path(args.work).resolve()
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    spec = work / "spec.json"
    if not spec.exists():
        spec = make_spec(repo, work)
    if args.delete_runs:
        runs = sorted((work / "work" / "results" / "v30_selftest" / "runs").rglob("*.json"))
        for path in np.random.default_rng(1).choice(runs, size=min(args.delete_runs, len(runs)), replace=False):
            Path(path).unlink()
        print(f"deleted {args.delete_runs} run records")
    os.environ.update(BETATOUR_V30_TEST="1", BETATOUR_V30_TEST_SPEC=str(spec), BETATOUR_V30_REPO=str(repo),
                      BETATOUR_V30_LKH=str(Path(args.lkh).resolve()), BETATOUR_V30_WORK=str(work / "work"))
    google, colab, drive = (types.ModuleType(m) for m in ("google", "google.colab", "google.colab.drive"))
    drive.mount = lambda *a, **k: print("(drive.mount stub)")
    colab.drive, google.colab = drive, colab
    sys.modules.update({"google": google, "google.colab": colab, "google.colab.drive": drive})
    notebook = json.loads((repo / "notebooks" / "33_BetaTour_v30_time_to_optimum.ipynb").read_text())
    (work / "cwd").mkdir(exist_ok=True)
    os.chdir(work / "cwd")
    space = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%%writefile "):
            first, _, body = source.partition("\n")
            Path(first.split()[1]).write_text(body if body.endswith("\n") else body + "\n")
            continue
        print(f"\n===== {cell['id']} =====", flush=True)
        exec(compile(source, cell["id"], "exec"), space)
    print("\nlocal run finished:", space["OUTPUT_ROOT"])


if __name__ == "__main__":
    main()
