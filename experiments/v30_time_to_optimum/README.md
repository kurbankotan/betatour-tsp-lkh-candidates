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

## Result (26 November 2026)

The notebook ran once, in one Colab session, on a Tesla T4 runtime with 8 vCPUs (Intel Xeon at 2.00 GHz, 51 GB of
RAM; four LKH-3 processes at a time). It found the protocol and itself on GitHub as built (commit `06a8dac`) at
15:14:21 UTC, and the 1,554 runs finished between 15:17 and 18:20 UTC (3.1 h). The result ZIP is
`results/v30/v30_tto37_results.zip`; `analysis/v30_time_to_optimum.py` recomputes every number below from it.

**Checks.** Every LKH-3 process ended normally, and every run that reached the optimum had its length; no tour was
shorter than the optimum. On rl1323 the first trial of seed 201 already reached the optimum, so that seed's
restarted arms took its CPU time (0.62 s). The pause of the restarted arms averaged 0.64–0.76 s at 1,000 nodes,
1.38–1.86 s at 2,000 nodes and 0.94–1.08 s on TSPLIB; most of it was the longest first trial, and BetaTour's
inference took 0.10–0.18 s on the GPU.

**Tests** (PAR10 over the 37 instances; beta lower / tied / higher; ties within 1% of the cap; ratio = geometric
mean of beta's PAR10 over the other arm's, with the notebook's 95% bootstrap interval):

| Test | Lower / tied / higher | p | Holm p | PAR10 ratio (95% CI) | Outcome |
|---|---|---|---|---|---|
| H1: `beta` < `pop` | 29 / 5 / 3 | 2.6 × 10⁻⁶ | 5.1 × 10⁻⁶ | 0.37 (0.23–0.63) | supported |
| H2: `beta` < `alpha` | 25 / 2 / 10 | 0.017 | 0.017 | 0.40 (0.18–0.87) | supported |
| secondary: `beta` < `count` | 22 / 11 / 4 | 0.00053 | – | 0.55 (0.36–0.85) | `beta` lower |

**Decision.** By the rule of the protocol, H1 and H2 are supported: with BetaTour's lists, single LKH-3 processes
reached the proven optima of these instances sooner than with LKH-3's own lists. Because `beta` also beat `count`, the gain is credited to
the learned ranking. Without the tie rule (not in the protocol), the counts were 31/1/5 (p = 1.3 × 10⁻⁵), 27/0/10
(p = 0.0076) and 28/1/8 (p = 0.0012).

**By subset** (descriptive; lower / tied / higher, PAR10 ratio):

| Pair | Synthetic (18) | TSPLIB (19) |
|---|---|---|
| `beta` / `pop` | 14/2/2, p = 0.0042, 0.34 | 15/3/1, p = 0.0005, 0.40 |
| `beta` / `alpha` | 14/1/3, p = 0.013, 0.24 | 11/1/7, p = 0.48, 0.64 (0.21–1.97) |
| `beta` / `count` | 10/6/2, p = 0.039, 0.38 | 12/5/2, p = 0.013, 0.79 |
| `beta` / `pop_r` | 12/3/3, p = 0.035, 0.44 | 14/2/3, p = 0.013, 0.25 |

On the TSPLIB instances alone, `beta` was not significantly faster than `alpha`. On u2319, `alpha` reached the
optimum with all six seeds (5.8 s on average) and no other arm reached it with any seed.

**Reaching the optimum** (% of runs) and **geometric mean of PAR10** (s):

| Group | `pop` | `alpha` | `pop_r` | `beta` | `count` | `mix` | `union` |
|---|---|---|---|---|---|---|---|
| uniform 1,000 (5) | 83.3 | 100.0 | 93.3 | 96.7 | 93.3 | 86.7 | 100.0 |
| clustered 1,000 (5) | 93.3 | 73.3 | 96.7 | 100.0 | 96.7 | 96.7 | 100.0 |
| uniform 2,000 (3) | 44.4 | 44.4 | 55.6 | 72.2 | 44.4 | 55.6 | 83.3 |
| clustered 2,000 (5) | 93.3 | 50.0 | 96.7 | 80.0 | 70.0 | 100.0 | 96.7 |
| TSPLIB (19) | 64.9 | 71.1 | 60.5 | 76.3 | 65.8 | 71.9 | 78.1 |
| all (37) | 73.4 | 70.3 | 74.3 | 82.4 | 72.5 | 79.7 | 86.9 |
| PAR10, all (s) | 48.6 | 45.3 | 54.9 | 18.0 | 32.6 | 23.2 | 14.9 |
| median time, all runs (s) | 10.5 | 11.6 | 13.1 | 4.1 | 10.1 | 5.0 | 4.7 |
| instances with 6/6 | 18 | 19 | 16 | 23 | 20 | 23 | 27 |

**Other pairs** (all 37 instances): `pop_r` / `pop` 13/7/17 (p = 0.58), so the pause and the restart alone did not
help; `union` / `beta` 11/8/18 (p = 0.26) and `mix` / `beta` 8/10/19 (p = 0.052), so neither longer list was
significantly better than BetaTour's top five, although `union` reached the optimum in the most runs (193 of 222).
On `cp1_clu2k_03`, BetaTour's lists missed seven optimal edges and all six `beta` runs ended 13 units (0.00006%)
above the optimum, whereas `pop`, `mix` and `union` reached it with all six seeds.

**Coverage and success (exploratory; the 21 instances whose optimal tours the pilot saved).** The initial lists
missed, per run, 0.60 optimal edges (`union`), 1.10 (`beta`), 2.42 (`mix`), 3.00 (`count`), 5.00 (`alpha`) and
8.21 (`pop`, `pop_r`). Runs whose lists missed no optimal edge reached the optimum in 99.1% of cases (225 runs);
with 1–2, 3–5, 6–10 and more than 10 missed edges, in 93.0%, 84.2%, 73.6% and 65.9%. These pooled shares mix
instances with lists: across the 21 instances, the mean number of missed edges and the share of successful runs had
a Spearman correlation of −0.78. Within instances, the correlation between missed edges and success was negative on
12 of the 14 instances on which both varied (median −0.19). Across arms the relation was not monotone: `pop` and
`pop_r` reached the optimum more often on these instances than `count` and `alpha`. This is an association, not a
causal estimate.

**Checks after the run (not in the protocol; 27 November 2026).** `analysis/v30_time_to_optimum.py` prints both.

1. *Overhead.* BetaTour's inference took 0.10–0.18 s on the T4. On the two-core CPU of the development machine,
   measured after the run on random uniform instances (`time_cpu_inference.py`), it took 0.74 s at 1,000 nodes with
   one PyTorch thread and 0.48 s with two (1.44 s and 0.86 s at 2,000 nodes); building the graph took under 0.02 s
   of this. The notebook sets `OMP_NUM_THREADS=1` before it imports PyTorch, so a CPU run of the notebook would use
   one thread. With a fixed number of seconds added to every `beta`
   run (a run that then exceeds the cap counts as missed):

   | Added | `beta` < `pop` | `beta` < `alpha` | `beta` < `count` |
   |---|---|---|---|
   | 0 s (protocol) | 29/5/3, p = 2.6 × 10⁻⁶ | 25/2/10, p = 0.017 | 22/11/4, p = 0.00053 |
   | 0.1 s | 27/7/3, p = 8.4 × 10⁻⁶ | 23/3/11, p = 0.058 | 22/11/4, p = 0.00053 |
   | 0.25 s | 27/7/3, p = 8.4 × 10⁻⁶ | 22/4/11, p = 0.08 | 21/11/5, p = 0.0025 |
   | 0.5 s | 26/7/4, p = 5.9 × 10⁻⁵ | 21/5/11, p = 0.11 | 20/9/8, p = 0.036 |
   | 1 s | 25/5/7, p = 0.0021 | 21/3/13, p = 0.23 | 19/6/12, p = 0.28 |
   | 2 s | 24/3/10, p = 0.024 | 19/2/16, p = 0.74 | 18/3/16, p = 0.86 |

   H1 holds with 2 s added; H2 does not hold with 0.1 s added, and the credit to the learned ranking does not hold
   with 1 s added.
2. *Six processes in parallel.* PAR10 describes single processes. Taking for each arm and instance the time at
   which the first of its six seeds reached the optimum (10 × the cap if none did), `beta` did not differ from
   `pop` (13/18/6, p = 0.17; geometric means 3.5 s and 3.4 s; 35 instances solved by each) but was faster than
   `alpha` (23/7/7, p = 0.0052) and `count` (14/20/3, p = 0.013). `union` did not differ from `beta` (6/22/9) or
   `pop` (8/19/10).

3. *Speed of the search.* In runs of at least 5 s, `beta` completed a median of 24% fewer trials per CPU second
   than `pop_r`, `union` 61% fewer and `count` 3% fewer (ratios 0.76, 0.39 and 0.97). With runs of at least 1 s
   or 10 s, or only the runs that reached the cap, the shares were 20–22%, 61–63% and 5–20%.

So the advantage over `pop` holds per process but not for six processes in parallel at these sizes, and the
advantages over `alpha` and `count` depend on fast inference.

**Limits.** One machine type, one cap (0.05 s per node), one LKH-3 version and 1,000–2,392 nodes. Times are LKH-3's
CPU times plus the wall time of inference and list building. The 18 synthetic instances motivated the run (their
coverage in the pilot), which is why the TSPLIB results are also given separately.

The manuscript reports this run in Sections 4.17 and 5.12 (Tables 14 and 15) and Appendix A.16.

## Deviations from the protocol

None. The protocol prefers 12 vCPUs and accepts 8; the run used 8 (four LKH-3 processes at a time).
