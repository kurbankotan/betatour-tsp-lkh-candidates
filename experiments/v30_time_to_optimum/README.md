# Run v30: time to the proven optimum with different candidate lists

Protocol: [`../PROTOCOL_v30.md`](../PROTOCOL_v30.md), written before the run. Notebook:
`notebooks/33_BetaTour_v30_time_to_optimum.ipynb`, built by `build_colab_notebook.py` from `cells/`, the module
`tto.py` and run v21's `alphaenc.py`.

## Question

Do lists that miss fewer optimal edges let LKH-3 reach the proven optimum sooner?

## Design

Earlier runs measured tour length at a fixed budget. This run measures the time to the proven optimum instead.

**Instances (37):**

- 18 synthetic instances of 1,000 and 2,000 nodes from the certified-graph pilot (`../certified_pilot/`), whose
  optima Concorde proved;
- 19 TSPLIB instances of 1,002 to 2,392 nodes with published optima.

**Arms (7)**, each run with six seeds and a cap of 0.05 s per node on LKH-3's CPU clock:

- `pop`, `alpha`: uninterrupted runs with POPMUSIC-5 and with LKH-3's default α-5 and ascent;
- `pop_r`, `beta`, `count`, `mix`, `union`: one first trial per seed, a pause until all six first trials are done
  and the lists are built, then a restart from the first-trial tour. The lists are the seed's own POPMUSIC-5 lists,
  BetaTour's top five, counting's top five, the lists of runs v21–v23, and BetaTour's top five followed by the
  seed's POPMUSIC-5 (up to ten candidates).

**Endpoint:** PAR10 per instance and arm.

**Tests:** H1 `beta` < `pop` and H2 `beta` < `alpha`, each an exact sign test over instances with Holm correction.
PAR10 values within 1% of the cap count as ties.

## Running

1. Commit and push the protocol, this folder and the notebook. The notebook refuses to start unless the protocol and
   the notebook on GitHub match its build.
2. Open the notebook in Colab with an L4 GPU and High RAM (12 vCPUs); a T4 with 8 vCPUs also works, but takes
   longer.
3. Run all cells. The first cell mounts Google Drive, and results go to `MyDrive/BetaTour/v30_tto37`.
4. If the session drops, run the notebook again on the same machine type. Every finished run is on Drive and is
   skipped.

Estimated time:

| Step | 12 vCPUs | 8 vCPUs |
|---|---|---|
| Set-up | 3–5 min | 3–5 min |
| Self-check | 1–2 min | 1–2 min |
| First trials and lists | 4–8 min | 5–10 min |
| 1,554 runs | 3–5 h | 4–7 h |
| Analysis and archive | 1–2 min | 1–2 min |

The total of the caps is 32.7 core-hours; most runs stop at the optimum before their cap.

## Outputs (on Drive)

- `runs/<instance>/<arm>_<seed>.json`: one record per run (time to the optimum, LKH-3's CPU time, pause, trials,
  final length);
- `instances/<instance>/first.npz`: the six first-trial tours with their lists, alpha-values, fathers and CPU times;
- `instances/<instance>/lists.json`: pauses and, where the pilot's optimal tour is available, the optimal edges each
  list missed;
- `instances/<instance>/lists.npz`: the candidate files (not in the ZIP);
- `selfcheck/selfcheck.json`;
- `summary/`: the tables and the decision;
- `v30_tto37_results.zip`: everything above except `lists.npz`.

## Tests

```
python experiments/v30_time_to_optimum/tests/test_tto.py --repo . --lkh /path/to/LKH-3.0.14/LKH
python experiments/v30_time_to_optimum/tests/run_notebook_locally.py --repo . --lkh /path/to/LKH-3.0.14/LKH
```

`test_tto.py` checks the module with the real LKH-3 on a 60-node instance whose optimum an exact MILP proved:

- the statistics;
- stopping at the optimum and at the CPU-time cap;
- that the one-trial `pop` run equals the first trial;
- the lists of every restarted arm, and that each arm reaches the optimum;
- the time-to-optimum rule.

`run_notebook_locally.py` runs the whole notebook in its test mode. It uses six small instances with MILP-proven
optima and a stand-in for the pilot's folder. With `--keep --delete-runs K` it checks that the notebook continues an
interrupted session.

## Technical checks before the run (not part of the results)

Two 1,000-node instances were run with two seeds per arm (201 and 202) on a two-core development machine, with
inference on its CPU. Times are in seconds and include the pause.

| Arm | `cp1_uni1k_02` (seed 201 / 202) | `cp1_uni1k_00` (seed 201 / 202) |
|---|---|---|
| `pop` | 0.36 / 0.39 | not reached in 50 s / 4.21 |
| `alpha` | 0.92 / 1.04 | 1.87 / 0.97 |
| `beta` | 0.98 / 1.01 | 1.62 / 1.85 |
| `count` | 0.42 / 0.45 | not reached in 50 s / 9.68 |
| `union` | 0.99 / 1.34 | 1.66 / 1.61 |

- **Overhead.** The six first trials took 0.28–0.36 s of CPU and inference about 0.55 s, so the pauses were
  0.34–0.36 s (`pop_r`, `count`) and 0.88–0.95 s (`beta`, `mix`, `union`).
- **`cp1_uni1k_02`.** Every run reached the optimum within 1.4 s, so overhead decided the order of the arms. This is
  why the protocol counts PAR10 differences within 1% of the cap as ties.
- **`cp1_uni1k_00`.** With seed 201, `pop` and `count` stayed 0.071% above the optimum for 50 s (6,813 and 10,437
  trials). `beta`, `union` and `alpha` reached the optimum within 2 s with both seeds.

Two instances say nothing about the tests; the run decides.

## Deviations from the protocol

None so far.
