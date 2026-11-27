"""Time BetaTour's inference on the CPU with one and two PyTorch threads (a check after run v30).

Usage:
    python experiments/v30_time_to_optimum/time_cpu_inference.py SRC_DIR LKH_BINARY CHECKPOINT

For a random uniform instance of 1,000 and of 2,000 nodes, the script runs the six first trials of LKH-3
(POPMUSIC-5, seeds 201-206), as in run v30, and then times `predict_tourfold` on the CPU three times after one warm-up
run, with 1 and with 2 threads. It prints the median of the graph construction (`prep`), the model and the total.
On the two-core development machine the totals were 0.74 s and 0.48 s at 1,000 nodes and 1.44 s and 0.86 s at
2,000 nodes (1 and 2 threads). The run itself did inference on a Tesla T4 (0.10-0.18 s).
"""
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
import tsplab as T  # noqa: E402
import tsp_model as TM  # noqa: E402

LKH, CKPT = sys.argv[2], sys.argv[3]
model, _ = TM.load_model(CKPT, torch.device("cpu"))
for n in (1000, 2000):
    rng = np.random.default_rng(7 + n)
    xy = rng.random((n, 2)) * 1_000_000
    work = tempfile.mkdtemp()
    firsts = [T.task_first_trial(LKH, work, s, xy) for s in range(201, 207)]
    succ = np.stack([T.tour_succ_pred(f["tour"])[0] for f in firsts], axis=1)
    cost = np.array([f["cost"] for f in firsts], dtype=float)
    for threads in (1, 2):
        torch.set_num_threads(threads)
        TM.predict_tourfold(model, xy, 16, torch.device("cpu"), ens_succ=succ, ens_cost=cost)   # warm-up
        times = []
        for _ in range(3):
            out = TM.predict_tourfold(model, xy, 16, torch.device("cpu"), ens_succ=succ, ens_cost=cost)
            times.append((out["t_prep"], out["t_model"], out["t_total"]))
        times = np.median(np.array(times), axis=0)
        print(f"n={n} threads={threads}: total {times[2]:.3f} s (prep {times[0]:.3f}, model {times[1]:.3f})", flush=True)
