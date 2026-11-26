"""Tests of tto.py with the real LKH-3.0.14 on small instances whose optima are proven by an exact MILP.

    python experiments/v30_time_to_optimum/tests/test_tto.py --repo REPO --lkh /path/to/LKH

1. Statistics: Holm, the paired sign test, PAR and the bootstrap ratio against hand-made values.
2. One LKH-3 run stops at the optimum (and reports Time.total), stops by itself at TOTAL_TIME_LIMIT when the
   optimum is out of reach, and the one-trial run of `pop` gives the same tour as first_trial (same seed).
3. The candidate files of the restarted arms are valid, `mix` equals the lists of run v21, `union` holds BetaTour's
   top five and the seed's POPMUSIC-5, and every restarted arm runs and reaches the optimum on a 60-node instance.
4. time_to_optimum: first trial at the optimum, pause plus Time.total, and the cap.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--lkh", required=True)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    sys.path[:0] = [str(repo / "src"), str(repo / "experiments" / "v21_alpha_encoding"), str(HERE.parent),
                    str(repo / "experiments" / "certified_pilot" / "tests")]
    import torch
    import tsplab as T
    import tsp_model as M
    import alphaenc as A
    import tto as V
    from fake_concorde import optimal_tour

    # 1. statistics
    assert np.allclose(V.holm([0.01, 0.04]), [0.02, 0.04]) and np.allclose(V.holm([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])
    c = V.compare([1.0, 2.0, 3.0, 5.0, 7.0], [2.0, 2.0, 4.0, 4.0, 9.0])
    assert (c["wins"], c["ties"], c["losses"]) == (3, 1, 1) and abs(c["p"] - A.sign_test_p(3, 1)) < 1e-12
    c = V.compare([1.0, 2.0, 3.0, 5.0, 7.0], [2.0, 2.0, 4.0, 4.0, 9.0], tol=[0.5, 0.5, 1.0, 1.0, 1.0])
    assert (c["wins"], c["ties"], c["losses"]) == (2, 3, 0), c
    assert abs(V.par([1.0, float("inf"), 3.0], 10.0) - (1.0 + 100.0 + 3.0) / 3) < 1e-12
    r = V.ratio_interval([2.0, 2.0, 2.0], [1.0, 1.0, 1.0], seed=1, replicates=200)
    assert abs(r["ratio"] - 2.0) < 1e-12 and abs(r["low"] - 2.0) < 1e-12 and abs(r["high"] - 2.0) < 1e-12
    print("1. statistics: ok")

    # 2. LKH runs on a 60-node instance with a proven optimum
    tmp = tempfile.mkdtemp(prefix="tto_test_")
    rng = np.random.default_rng(7)
    xy = T.to_int_coords(T.gen_instance(60, "clustered", rng))
    pf = str(Path(tmp) / "p.tsp")
    T.write_tsp(pf, xy, name="p")
    D = np.floor(np.sqrt(((xy[:, None, :] - xy[None, :, :]).astype(float) ** 2).sum(-1)) + 0.5).astype(np.int64)
    tour, opt = optimal_tour(D)
    assert T.tour_length(xy, tour) == opt
    pop = dict(V.UNINTERRUPTED["pop"], MAX_CANDIDATES=5)
    run = V.lkh_run(args.lkh, tmp, pf, xy, "EUC_2D", 201, opt, 20.0, params=pop)
    assert run["found"] and run["time_total"] is not None and run["time_total"] < 20.0 and not run["shorter"], run
    limited = V.lkh_run(args.lkh, tmp, pf, xy, "EUC_2D", 201, opt - 1, 1.0, params=pop)
    assert not limited["found"] and limited["returncode"] == 0 and 0.9 <= limited["time_total"] <= 1.6, limited
    assert limited["cost"] == opt and not limited["shorter"]       # LKH finds the optimum and goes on to the limit
    first = V.first_trial(args.lkh, tmp, 203, xy, pf, "EUC_2D", 1)
    one = V.lkh_run(args.lkh, tmp, pf, xy, "EUC_2D", 203, 1, 30.0, params=pop, max_trials=1, return_tour=True)
    assert np.array_equal(one["tour"], first["tour"].astype(np.int64)), "one-trial pop run != first trial"
    print(f"2. LKH runs: optimum {opt} reached in {run['time_total']} s; limit 1.0 s -> {limited['time_total']} s; "
          "one-trial pop run equals first_trial: ok")

    # 3. candidate files of the restarted arms
    firsts = [V.first_trial(args.lkh, tmp, s, xy, pf, "EUC_2D", opt) for s in range(201, 207)]
    succ = np.stack([T.tour_succ_pred(f["tour"])[0] for f in firsts], axis=1).astype(np.int32)
    model, _ = M.load_model(str(repo / "checkpoints" / "betatour_v2.pt"), torch.device("cpu"))
    pred = M.predict_tourfold(model, xy, 16, torch.device("cpu"), ens_succ=succ,
                              ens_cost=np.array([f["cost"] for f in firsts]))
    lists, build, rank, count_rank = V.arm_lists(firsts, pred)
    for j, f in enumerate(firsts):
        tree_keys = A.tree_edge_keys(f["cand"], f["alpha"])
        mix_v21 = T.mix_candidates(pred["rank"], f["cand"], k_first=3, k=5, fill=pred["rank"][:, 3:])
        assert np.array_equal(lists["mix"][j][0], mix_v21)
        assert np.array_equal(lists["mix"][j][1], A.tree_alpha(mix_v21, tree_keys)[0])
        uni = lists["union"][j][0]
        assert np.array_equal(uni[:, :5], rank[:, :5])
        for v in range(len(xy)):
            row = set(uni[v][uni[v] >= 0].tolist())
            pop_row = f["cand"][v]                          # POPMUSIC may list fewer than five (-1 padded)
            assert row == set(rank[v, :5].tolist()) | set(pop_row[pop_row >= 0].tolist()), v
        assert np.array_equal(lists["pop_r"][j][1], f["alpha"]) and np.array_equal(lists["pop_r"][j][2], f["dad"])
    assert np.array_equal(lists["count"][0][0], count_rank[:, :5])
    pause = V.pauses([f["time_total"] for f in firsts], 0.5, build)
    assert abs(pause["beta"] - (max(f["time_total"] for f in firsts) + 0.5 + build["beta"])) < 1e-12
    assert abs(pause["count"] - (max(f["time_total"] for f in firsts) + build["count"])) < 1e-12
    for arm in V.RESTARTED:
        res = V.lkh_run(args.lkh, tmp, pf, xy, "EUC_2D", 201, opt, 20.0, files=lists[arm][0],
                        init_tour=firsts[0]["tour"])
        assert res["found"] and res["returncode"] == 0, (arm, res)
    print("3. candidate files valid, mix = run v21 lists, union = BetaTour-5 + POPMUSIC-5, all restarted arms reach "
          "the optimum: ok")

    # 4. time to the optimum
    assert V.time_to_optimum({"first_found": True, "first_cpu": 0.3}, 1.0) == 0.3
    assert V.time_to_optimum({"found": True, "time_total": 0.4, "pause": 0.5}, 1.0) == 0.9
    assert V.time_to_optimum({"found": True, "time_total": 0.6, "pause": 0.5}, 1.0) == float("inf")
    assert V.time_to_optimum({"found": False, "time_total": 0.2}, 1.0) == float("inf")
    print("4. time_to_optimum: ok")
    print("all tests passed")


if __name__ == "__main__":
    main()
