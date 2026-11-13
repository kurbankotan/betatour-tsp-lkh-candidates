"""Tests of the run-v12 module: edge accounting against tsplab, cost checks, tour-file parsing, and the full
first-trial / native-run / analysis pipeline with the protocol stand-in for LKH (tests/fake_lkh_v12.py).

    python tests/test_selfcorrect.py --src REPO/src --checkpoint REPO/checkpoints/betatour_v2.pt
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import selfcorrect as S  # noqa: E402


def unit_tests(T):
    rng = np.random.default_rng(7)
    for n in (5, 17, 200):
        xy = rng.random((n, 2)) * 1000
        tour = rng.permutation(n)
        cand = np.stack([rng.choice(np.delete(np.arange(n), i), size=min(5, n - 1), replace=False)
                         for i in range(n)])
        cand[0, -1] = -1                                     # padding must be ignored
        keys = S.tour_keys(tour)
        assert len(keys) == n and np.all(np.diff(keys) > 0)
        missing = len(np.setdiff1d(keys, S.list_keys(cand)))
        assert missing == T.candidate_coverage(cand, tour, ks=[cand.shape[1]])[cand.shape[1]]["missing"]
        for ew in ("EUC_2D", "CEIL_2D"):
            c = S.checked_cost(T, xy, tour, ew)
            assert c == int(round(T.tour_length(xy, tour, ew)))
    try:
        S.checked_cost(T, xy, np.r_[tour[:-1], tour[0]], "EUC_2D")
        raise AssertionError("an invalid tour was accepted")
    except ValueError:
        pass
    text = b"NAME : x\nTYPE : TOUR\nDIMENSION : 3\nTOUR_SECTION\n2\n3\n1\n-1\nEOF\n"
    assert S.tour_file_complete(text) and list(S.parse_tour_bytes(text)) == [1, 2, 0]
    assert not S.tour_file_complete(text[:-12])
    try:
        S.parse_tour_bytes(text[:-9])
        raise AssertionError("an incomplete tour was accepted")
    except ValueError:
        pass
    assert S.sign_test_p(0, 0) == 1.0 and abs(S.sign_test_p(0, 7) - 2 / 128) < 1e-12
    assert S.parse_trace("* 1: Cost = 10, Gap = 0.1%, Time = 0.20 sec.\n* 7: Cost = 9, Time = 1.50 sec.\n") == \
        [(1, 10, 0.2), (7, 9, 1.5)]
    print("unit tests passed")


def pipeline_test(T, M, checkpoint: str):
    import torch
    device = torch.device("cpu")
    model, _ = M.load_model(checkpoint, device)
    fake = HERE / "fake_lkh_v12.py"
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    os.environ["FAKE_LKH_TRIAL_SLEEP_S"] = "0.005"
    seeds = [101, 102, 103, 104, 105, 106]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rng = np.random.default_rng(11)
        instances = []
        xy = T.to_int_coords(T.gen_instance(160, "uniform", rng), scale=100000)
        T.write_tsp(str(tmp / "uni.tsp"), xy, name="uni", ew="EUC_2D")
        instances.append(dict(name="uni_test", group="uni1k", xy=xy, ew="EUC_2D", problem=tmp / "uni.tsp"))
        xyc = np.round(T.gen_instance(150, "clustered", rng) * 5000.0, 2)
        T.write_tsp(str(tmp / "ceil.tsp"), xyc, name="ceil", ew="CEIL_2D")
        instances.append(dict(name="ceil_test", group="tsplib", xy=xyc, ew="CEIL_2D", problem=tmp / "ceil.tsp"))
        records = []
        for inst in instances:
            # reference: best of a few long stand-in runs
            ref = S.run_native_pop(T, str(fake), inst["problem"], inst["xy"], inst["ew"], tmp / f"ref_{inst['name']}",
                                   [1, 2, 3, 4, 5, 6], budget_s=4.0)
            best = None
            for seed, raw in ref["seeds"].items():
                caps, _ = S.decode_captures(T, inst["xy"], inst["ew"], raw["captures"], 4.0)
                if caps and (best is None or caps[-1][1] < best[1]):
                    best = caps[-1]
            first = S.run_first_trials(T, str(fake), inst["problem"], inst["xy"], inst["ew"],
                                       tmp / f"first_{inst['name']}", seeds)
            assert first["tours"].shape == (6, len(inst["xy"])) and first["cands"].shape == (6, len(inst["xy"]), 5)
            lists = S.model_and_count_lists(M, T, model, device, inst["xy"], first["tours"], first["costs"])
            assert lists["beta"].shape == (len(inst["xy"]), 5) and lists["count"].shape == lists["beta"].shape
            native = S.run_native_pop(T, str(fake), inst["problem"], inst["xy"], inst["ew"],
                                      tmp / f"native_{inst['name']}", seeds, budget_s=2.5)
            summary, audit = S.analyse_instance(T, dict(inst, reference_source="test"), best[2], first, native,
                                                lists, 2.5)
            n = len(inst["xy"])
            for row in summary["seed_rows"]:
                seed = row["seed"]
                m = seeds.index(seed)
                cov = T.candidate_coverage(native["seeds"][seed]["c0"], best[2], ks=[5])[5]["missing"]
                assert row["missing_c0"] == cov, (row["missing_c0"], cov)
                for f in ("25", "50", "75", "100"):
                    assert row[f"recovered_{f}"] + row[f"missing_dynamic_{f}"] == row["missing_c0"]
                seq = [row[f"missing_dynamic_{f}"] for f in ("25", "50", "75", "100")]
                assert seq == sorted(seq, reverse=True), seq
                assert row["r0_in_final"] <= row["recovered_100"]
                assert row["captures"] >= 2, "the stand-in improved at least once"
                assert np.array_equal(native["seeds"][seed]["c0"], first["cands"][m])
                assert summary["flags"].get(str(seed), []) in ([], ["first_trial_not_captured"]), summary["flags"]
                assert row["final_cost"] <= row["first_capture_cost"]
            assert summary["missing_beta"] == T.candidate_coverage(lists["beta"], best[2], ks=[5])[5]["missing"]
            assert 0 <= summary["per1000"]["missing_dynamic_100"] <= summary["per1000"]["missing_c0"]
            assert set(audit) >= {"ref_tour", "beta", "count", "first_tours"}
            S.atomic_json(tmp / "out" / f"{inst['name']}.json", summary)
            S.atomic_npz(tmp / "out" / f"{inst['name']}.npz", **audit)
            records.append(summary)
            print(inst["name"], {k: round(v, 2) for k, v in summary["per1000"].items()},
                  {k: (round(v, 3) if v is not None else None) for k, v in summary["rates"].items()})
        inst_table, group_table, primary = S.summary_tables(records)
        assert len(inst_table) == 2 and "all" in group_table.index
        assert primary["instances"] == 2 and 0.0 <= primary["sign_test_p"] <= 1.0
        print(group_table.round(3).to_string())
        print(primary)
    print("pipeline test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--checkpoint", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import tsplab as T
    import tsp_model as M
    unit_tests(T)
    pipeline_test(T, M, a.checkpoint)
