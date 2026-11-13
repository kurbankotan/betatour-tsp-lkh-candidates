"""End-to-end test of the v11r2 notebook cells with a protocol-level fake LKH.

Usage:
  python test_pipeline_fake_lkh.py --src REPO/src [--checkpoint REPO/checkpoints/betatour_v2.pt]

It regenerates the notebook, then executes its warm-up (cell 5), run (cell 7),
analysis (cell 8) and archive (cell 9) cells on two 200-node instances with
the fake solver in place of LKH. Cells 1-4 and 6 need Colab, Drive, a GPU and
LKH sources, so the globals they define are set here instead. Without a
checkpoint a deterministic stand-in model is used. Afterwards it checks the
truncated-native-file fallback and the patched-event recovery directly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import time
import types
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V11 = HERE.parent
EXPERIMENTS = V11.parent


class StandInModel:
    """Distance-based probabilities on the 16-NN graph (no checkpoint needed)."""

    cfg = {"use_ensemble": True}

    @staticmethod
    def predict_tourfold(model, xy, K, device, ens_succ=None, ens_cost=None, k_out=5):
        start = time.perf_counter()
        xy = np.asarray(xy, dtype=np.float64)
        distance = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(distance, np.inf)
        graph = np.argsort(distance, axis=1, kind="stable")[:, :K].astype(np.int64)
        d = np.take_along_axis(distance, graph, axis=1)
        in_ensemble = np.zeros(graph.shape)
        for column in range(ens_succ.shape[1]):
            successor = ens_succ[:, column]
            predecessor = np.empty_like(successor)
            predecessor[successor] = np.arange(len(xy))
            in_ensemble += (graph == successor[:, None]) | (graph == predecessor[:, None])
        probability = np.clip(0.15 * in_ensemble + np.exp(-d / d[:, :1].clip(min=1)) / 2, 0, 1)
        return {"NB": graph, "prob": probability, "D": d, "t_total": time.perf_counter() - start}


def notebook_cells() -> list[str]:
    spec = importlib.util.spec_from_file_location("v11r2_builder", V11 / "build_colab_notebook.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return ["".join(cell["source"]) for cell in builder.make_notebook()["cells"]]


def hook_telemetry_keys() -> list[str]:
    text = (V11 / "BetaTourSearchHook.h").read_text()
    body = text.split("static void BT_WriteTelemetry", 1)[1].split("/* Per-action rows", 1)[0]
    return re.findall(r'fprintf\(File, "([a-z0-9_]+) ', body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="repository src/ with tsplab.py and tsp_model.py")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    sys.path[:0] = [str(args.src), str(EXPERIMENTS / "livehook"), str(V11), str(HERE)]
    import tsplab as T
    import pilot
    from coordinator import read_tour
    from pilot import run_guided_arm, run_native_arm, sha256_file
    from policy import POLICY_VERSION
    import fake_lkh

    keys = hook_telemetry_keys()
    assert tuple(keys) == fake_lkh.TELEMETRY_KEYS, ("fake LKH telemetry keys differ from the hook", keys)

    if args.checkpoint:
        import torch
        import tsp_model as M
        device = torch.device("cpu")
        model, _ = M.load_model(str(args.checkpoint), device)
        model_kind = "checkpoint on CPU"
    else:
        M, model, device, model_kind = StandInModel(), None, None, "stand-in model"

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    downloads = []
    colab.files = types.SimpleNamespace(download=downloads.append)
    google.colab = colab
    sys.modules.update({"google": google, "google.colab": colab})

    with tempfile.TemporaryDirectory(prefix="betatour_v11r2_e2e_") as temp:
        root = Path(temp)
        fake = root / "fake_lkh.py"
        shutil.copy2(HERE / "fake_lkh.py", fake)
        fake.chmod(0o755)
        instances = []
        for group, distribution, seed in (("uni", "uniform", 11), ("clu", "clustered", 12)):
            xy = T.to_int_coords(T.gen_instance(200, distribution, np.random.default_rng(seed)))
            instances.append({"name": f"test_{group}200_000", "group": f"{group}200", "n": 200,
                              "budget_s": 4.0, "xy": xy,
                              "xy_sha256": hashlib.sha256(np.ascontiguousarray(xy).tobytes()).hexdigest()})
        arms = ["native_pop6", "native_pop7", "sham", "random_add", "frequency_add", "beta_add"]
        g = {"__name__": "__main__", "json": json, "os": os, "shutil": shutil, "time": time,
             "uuid": uuid, "hashlib": hashlib, "zipfile": zipfile, "np": np, "pd": pd, "Path": Path,
             "display": lambda value: None, "T": T, "M": M, "MODEL": model, "DEVICE": device,
             "EXPERIMENT_NAME": "v11r2_fake_lkh_test", "SEEDS": list(range(301, 307)),
             "ARMS": arms, "INSTANCE_SPECS": [["fake", 200, 2, "mixed", 0, 4.0]],
             "EXPECTED_XY_SHA256": {i["name"]: i["xy_sha256"] for i in instances},
             "TARGET_COSTS": {instances[0]["name"]: 10**12, instances[1]["name"]: 1},
             "MAX_NEW_ARMS": None, "SPARSE_FRACTION": 0.25, "GRAPH_K": 16,
             "REPO_COMMIT": "test", "CHECKPOINT_SHA256": "test-checkpoint",
             "WORK_ROOT": root / "work", "OUTPUT_ROOT": root / "drive",
             "HARDWARE": {"gpu": "test", "vcpus": os.cpu_count()},
             "CODE_SHA256": {"test": "x"}, "BUILD_INFO": {"native_sha256": "fake"},
             "NATIVE_LKH": fake, "PATCHED_LKH": fake, "POLICY_VERSION": POLICY_VERSION,
             "run_guided_arm": run_guided_arm, "run_native_arm": run_native_arm,
             "sha256_file": sha256_file, "read_tour": read_tour, "INSTANCES": instances}
        for sub in ("instances", "code", "summary", "errors", "archives"):
            (g["OUTPUT_ROOT"] / sub).mkdir(parents=True, exist_ok=True)
        g["WORK_ROOT"].mkdir()
        g["arm_order"] = lambda name: sorted(arms, key=lambda arm: hashlib.sha256((name + ":" + arm).encode()).hexdigest())

        cells = notebook_cells()
        for index in (5, 7, 8, 9):
            exec(compile(cells[index], f"<notebook cell {index}>", "exec"), g)

        table = pd.read_csv(g["OUTPUT_ROOT"] / "summary" / "complete_instances.csv")
        assert len(table) == 2 and table["matched_policy_state"].all(), table.T
        assert (table["fresh_inference_arms"] == 1).all()
        assert table["fresh_predictions_match"].all()
        mechanism = pd.read_csv(g["OUTPUT_ROOT"] / "summary" / "candidate_mechanism.csv")
        assert set(mechanism.arm) == {"sham", "random_add", "frequency_add", "beta_add"}
        assert mechanism.loc[mechanism.arm != "sham", "slot_checks"].sum() > 0
        rates = pd.read_csv(g["OUTPUT_ROOT"] / "summary" / "candidate_mechanism_rates.csv")
        assert list(rates.arm) == ["random_add", "frequency_add", "beta_add"]
        events = pd.read_csv(g["OUTPUT_ROOT"] / "summary" / "improvement_trace_quality.csv")
        assert (events.loc[events.arm.str.startswith("native"), "final_tour_source"] == "final_output_tour_file").all()
        targets = pd.read_csv(g["OUTPUT_ROOT"] / "summary" / "time_to_target_exploratory.csv")
        assert targets.groupby("instance").reached.sum().to_dict() == {instances[0]["name"]: 6, instances[1]["name"]: 0}
        caches = list((g["OUTPUT_ROOT"] / "instances").glob("*/decision_cache/*.npz"))
        assert len(caches) == 2, caches
        info = json.loads((g["OUTPUT_ROOT"] / "archives" / "archive_info.json").read_text())
        assert info["complete"] and downloads == [info["results"]["local"]]
        with zipfile.ZipFile(info["results"]["local"]) as archive:
            names = archive.namelist()
        assert not any("decision_cache" in name or name.endswith("best.tour") for name in names)
        with zipfile.ZipFile(info["audit"]["local"]) as archive:
            assert any("decision_cache" in name for name in archive.namelist())

        # Native arm killed during an in-place rewrite: fall back to the last complete tour.
        os.environ["FAKE_LKH_TRUNCATE_AFTER_S"] = "1.5"
        try:
            problem = root / "truncate.tsp"
            T.write_tsp(problem, instances[0]["xy"])
            native = run_native_arm("native_pop6", fake, problem, instances[0]["xy"],
                                    root / "truncate", g["SEEDS"], 3.0, T)
        finally:
            del os.environ["FAKE_LKH_TRUNCATE_AFTER_S"]
        assert set(native["final_tour_sources"].values()) == {"last_complete_poll_after_invalid_final_file"}
        for seed in g["SEEDS"]:
            assert native["native_improvement_events"][str(seed)][-1]["cost"] == native["seed_costs"][g["SEEDS"].index(seed)]

        # Patched arm killed between the atomic best.tour rename and its event append,
        # and killed during the append itself.
        base = root / "events"
        base.mkdir()
        (base / "best.tour").write_text("3\n1\n2\n3\n")
        (base / "best.events").write_text(f"{time.time() - 2:.6f} 120\n{time.time() - 1:.6f} 110\n")
        events, flags = pilot._patched_events(base, time.time() - 5, 100)
        assert [e["cost"] for e in events] == [120, 110, 100] and flags == ["final_event_recovered_from_tour_mtime"]
        (base / "best.events").write_text(f"{time.time() - 2:.6f} 120\n{time.time() - 1:.6f} 1")
        events, flags = pilot._patched_events(base, time.time() - 5, 100)
        assert flags == ["partial_final_event_line_dropped", "final_event_recovered_from_tour_mtime"]
        try:
            pilot._patched_events(base, time.time() - 5, 130)
        except AssertionError:
            pass
        else:
            raise AssertionError("a final tour worse than a logged incumbent must be rejected")
    print(f"v11r2 notebook pipeline passed with the fake LKH ({model_kind})")


if __name__ == "__main__":
    main()
