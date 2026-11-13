"""Run the code cells of the run-v12 notebook end to end with the protocol stand-in for LKH, then run them again to
check that completed instances are skipped and the archives are rebuilt.

    python tests/test_notebook_fake_lkh.py --repo REPO --checkpoint CHECKPOINT
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main(repo: Path, checkpoint: Path) -> None:
    import build_colab_notebook as B
    sys.path.insert(0, str(repo / "src"))
    import tsplab as T
    import selfcorrect as S
    nb = B.make_notebook()
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    fake = HERE / "fake_lkh_v12.py"
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for sub in ("out", "work", "eval", "tsplib"):
            (tmp / sub).mkdir()
        # instances of the smoke profile, recreated exactly as the notebook does
        rng = np.random.default_rng(2024)
        insts = []
        for name, dist in (("uni1k_000", "uniform"), ("clu1k_000", "clustered")):
            insts.append((name, T.to_int_coords(T.gen_instance(180, dist, rng)), "EUC_2D"))
        xyc = np.round(T.gen_instance(160, "clustered", np.random.default_rng(5)) * 4000.0, 1)
        T.write_tsp(str(tmp / "tsplib" / "ceil160.tsp"), xyc, name="ceil160", ew="CEIL_2D")
        insts.append(("ceil160", T.read_tsplib(str(tmp / "tsplib" / "ceil160.tsp"))["xy"], "CEIL_2D"))
        os.environ["FAKE_LKH_TRIAL_SLEEP_S"] = "0.005"
        records = []
        for name, xy, ew in insts:
            problem = tmp / f"ref_{name}.tsp"
            T.write_tsp(str(problem), xy, name=name, ew=ew)
            ref = S.run_native_pop(T, str(fake), problem, xy, ew, tmp / f"ref_{name}", [7, 8, 9, 10, 11, 12], 3.0)
            best = min((S.decode_captures(T, xy, ew, r["captures"], 3.0)[0][-1] for r in ref["seeds"].values()),
                       key=lambda r: r[1])
            records.append({"n": len(xy), "name": name, "xy": np.asarray(xy), "tour": best[2].astype(np.int32),
                            "ew": ew})
        T.save_shard(str(tmp / "eval" / "eval_part000.npz"), records)
        env = {"BETATOUR_V12_TEST": "1", "BETATOUR_V12_REPO": str(repo), "BETATOUR_V12_LKH": str(fake),
               "BETATOUR_V12_CHECKPOINT": str(checkpoint), "BETATOUR_V12_OUT": str(tmp / "out"),
               "BETATOUR_V12_WORK": str(tmp / "work"), "BETATOUR_V12_EVAL": str(tmp / "eval"),
               "BETATOUR_V12_TSPLIB": str(tmp / "tsplib")}
        os.environ.update(env)
        for attempt in (1, 2):
            ns = {"__name__": "__main__"}
            for i, src in enumerate(cells, 1):
                exec(compile(src, f"<v12 cell {i}>", "exec"), ns)
            out = tmp / "out" / "v12_selfcorrect43"
            done = sorted(p.stem for p in (out / "instances").glob("*.json"))
            assert done == ["ceil160", "clu1k_000", "uni1k_000"], done
            primary = json.loads((out / "summary" / "primary.json").read_text())
            assert primary["instances"] == 3 and primary["complete_instances"] == 3
            prog = json.loads((out / "progress.json").read_text())
            assert prog["state"] == "complete" and prog["completed"] == 3
            with zipfile.ZipFile(out / "archives" / "v12_selfcorrect43_results.zip") as z:
                names = set(z.namelist())
                assert {"manifest.json", "progress.json", "summary/primary.json", "summary/group_table.csv",
                        "instances/uni1k_000.json", "code/v12_self_correction/selfcorrect.py"} <= names, names
                assert z.read("code/v12_self_correction/selfcorrect.py") == (HERE.parent / "selfcorrect.py").read_bytes()
            with zipfile.ZipFile(out / "archives" / "v12_selfcorrect43_audit.zip") as z:
                assert {"audit/uni1k_000.npz", "audit/ceil160.npz"} <= set(z.namelist())
            rec = json.loads((out / "instances" / "ceil160.json").read_text())
            assert rec["ew"] == "CEIL_2D" and rec["complete"] and len(rec["seed_rows"]) == 6
            print(f"attempt {attempt}: notebook cells ran; primary = {primary}")
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["code_sha256"]["v12_self_correction/selfcorrect.py"] == \
            S.sha256_file(HERE.parent / "selfcorrect.py")
    print("v12 notebook pipeline passed with the stand-in solver (resume included)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    a = ap.parse_args()
    main(Path(a.repo).resolve(), Path(a.checkpoint).resolve())
