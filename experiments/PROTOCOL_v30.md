# Protocol for run v30: time to the proven optimum with different candidate lists (written before the run)

## Why this run

The runs of the study measured the length of the tour at a fixed budget. None of them measured the time LKH-3 needs
to reach a proven optimal tour, and the manuscript lists this as a limitation. Heins et al. measured exactly this time
on evolved instances of up to 2,000 nodes and found that the candidate lists mattered there: lists that missed optimal
edges slowed LKH down, and rebuilding the lists at every restart helped about tenfold.

A pilot for a follow-up project (`certified_pilot/`, 26 November 2026; not a run of the study) used Concorde to prove
the optima of 21 instances of 1,000 to 2,000 nodes. Against these proven optimal tours, five-candidate lists missed
these numbers of optimal edges in total (29,479 edges):

| Lists | Missed optimal edges | Instances with the whole optimal tour inside the lists |
|---|---|---|
| BetaTour top five | 23 | 11 |
| counting top five | 63 | 8 |
| α-5 (LKH-3's default) | 105 | 0 |
| POPMUSIC-5 | 168 | 0 |

These comparisons were exploratory. This run tests whether lists that miss fewer optimal edges let LKH-3 reach the
optimum sooner.

## Endpoint and hypotheses

The endpoint is the PAR10 of an arm on an instance. It is the mean, over the six seeds, of the time to the optimum,
where a run that does not reach the optimum within the cap counts as 10 × the cap.

Each test is an exact two-sided sign test over instances on PAR10, with Holm correction over the two tests. Two
PAR10 values count as a tie when they differ by at most 1% of the instance's cap (0.5 s at 1,000 nodes), and ties are
excluded. Without this rule, the pause of the restarted arms (about a second) alone would decide the instances that
LKH-3 solves within a second.

- **H1.** `beta` has a lower PAR10 than `pop` (the baseline of the study).
- **H2.** `beta` has a lower PAR10 than `alpha` (LKH-3's default configuration).

A hypothesis is supported if `beta` is lower on more instances than it is higher and its Holm-adjusted p is below
0.05.

## Decision rule

- **H1 and H2 both supported.** With BetaTour's lists, LKH-3 reaches proven optima sooner than with its own lists at
  these sizes. The manuscript reports the run.
  - The learned ranking is credited only if `beta` also has a lower PAR10 than `count` (secondary sign test,
    unadjusted p < 0.05).
  - Otherwise the gain is attributed to lists built from the six first-trial tours, as in runs v21 and v22.
- **Only one hypothesis supported.** The run is reported as a partial result.
- **Neither supported.** No advantage in time to the optimum; the run is reported as a negative result.
- **Everything else is descriptive:**
  - success rates (out of six), median times and PAR10 by group (uniform and clustered at 1,000 and 2,000 nodes,
    TSPLIB);
  - PAR10 ratios with 95% intervals from 5,000 bootstrap resamples over instances;
  - `pop_r` against `pop` (the cost of the pause and the restart), `mix` and `union` against `beta`, and every other
    pair;
  - for the 21 instances whose optimal tours the pilot saved, the optimal edges each run's lists missed, against
    whether the run reached the optimum.

## Design

### Instances (37)

- **18 synthetic instances of the pilot**, with optima proven there by Concorde:
  - 5 uniform and 5 clustered of 1,000 nodes (generator seeds 15101 and 15202);
  - 3 uniform of 2,000 nodes (seed 15303, instances 1, 3 and 4; Concorde did not finish instances 0 and 2);
  - 5 clustered of 2,000 nodes (seed 15404).

  They are regenerated from the generator of the study, and their files are checked against SHA-256 values computed
  from the same code.
- **19 TSPLIB instances** (EUC_2D, 1,002 to 2,392 nodes) with their published optima: pr1002, u1060, vm1084, pcb1173,
  d1291, rl1304, rl1323, nrw1379, fl1400, u1432, fl1577, d1655, vm1748, u1817, rl1889, d2103, u2152, u2319 and pr2392.

### Cap, seeds and time

- **Cap:** 0.05 s per node on LKH-3's own CPU clock (`TOTAL_TIME_LIMIT`, counted from the start of the program).
- **Seeds:** six per arm (201 to 206).
- **Time to the optimum:**
  - uninterrupted arms: LKH-3's `Time.total` of the run;
  - restarted arms: the pause plus the `Time.total` of the continuation. The pause is the longest CPU time of the six
    first trials, plus the wall time of BetaTour's inference (`beta`, `mix`, `union`), plus the wall time of
    building the lists;
  - a seed whose first trial already reaches the optimum gets the CPU time of that first trial in every restarted
    arm.
- **Success:** a run reaches the optimum when its output tour, recomputed, has the optimal length within the cap. A
  tour shorter than the optimum would stop the run as an error.

### Arms (7)

**Uninterrupted arms:**

- `pop`: POPMUSIC-5, ascent-free. This is the study's baseline, and its first trial is the first trial below.
- `alpha`: α-5 with subgradient ascent, LKH-3's default.

**Restarted arms.** Each seed runs one ascent-free POPMUSIC-5 trial. The six first trials run in parallel, as in the
study, and the pause lasts until all six are done and the lists are built. LKH-3 then restarts from the seed's
first-trial tour with a candidate file, zero penalties and no ascent.

- `pop_r`: the seed's own POPMUSIC-5 lists in LKH-3's own encoding. This is the control for the pause and the
  restart.
- `beta`: BetaTour's top five (the model reads the six first-trial tours), in the tree encoding of run v21.
- `count`: the top five by counting over the six first-trial tours, on BetaTour's graph with ties broken by length,
  in the tree encoding.
- `mix`: the lists of runs v21 to v23 (BetaTour's top three, then the seed's POPMUSIC-5, then BetaTour's next ones),
  in the tree encoding.
- `union`: BetaTour's top five, then the seed's POPMUSIC-5 entries not yet listed (up to ten candidates), in the
  tree encoding.

### LKH-3 settings

LKH-3.0.14 with RUNS = 1, MAX_TRIALS = 10^8, OPTIMUM = the optimum, STOP_AT_OPTIMUM = YES, TOTAL_TIME_LIMIT = the cap
(restarted arms: the cap minus the pause) and TRACE_LEVEL = 1. MAX_CANDIDATES is 5, or 10 for `union`.

### Machine

- Half of the visible vCPUs run LKH-3 processes at a time (6 on 12 vCPUs).
- All runs of all arms go in one random order (fixed seed 3030), so the load is spread evenly over the arms.
- Colab with an L4 GPU and high RAM (12 vCPUs) is preferred; a T4 with 8 vCPUs is accepted. The machine is recorded,
  and all sessions use the same machine type.

## Scope and limits

- One cap per node, one LKH-3 version and one machine type, with sizes of 1,000 to 2,392 nodes. Times are on LKH-3's
  CPU clock, plus the wall time of inference and list building.
- The restarted arms pay for the pause. The pause assumes six processes running at the same time, as in the study.
  A technical check on one 1,000-node instance found pauses of 0.4 s (`pop_r`, `count`) and 0.9 s (with inference on a
  CPU), while LKH-3 reached the optimum there within about one second in every arm. Instances like this are decided
  by overhead, not by the lists; the tie rule above keeps them out of the tests.
- Synthetic optima rest on the pilot's Concorde runs (exact lower bounds reported by Concorde); TSPLIB optima are the
  published values.
- The 18 synthetic instances motivated this run (their coverage in the pilot). The TSPLIB results are therefore also
  reported separately.
- No interim decision: the run ends when all runs are done. Every deviation from this protocol is listed in
  `v30_time_to_optimum/README.md`.

## Common rules

- **Code and files:**
  - LKH-3.0.14 from the official archive or from the pinned copy checked by SHA-256;
  - repository commit d63f7e17 for `tsplab.py`, `tsp_model.py` and the checkpoint (SHA-256 2322f66c…);
  - TSPLIB files from the TSPLIB website.
- **Modules:** `alphaenc.py` (run v21, unchanged) and `tto.py` (this run) are embedded in the notebook and compared
  with their SHA-256 values. They are copied to the result archive and recorded in the manifest, together with the
  SHA-256 of this protocol and one SHA-256 over all code cells, the modules and this protocol.
- **Pushed before the run:** the notebook does not start unless this protocol and the notebook are on the main
  branch on GitHub as built.
- **Changes:** a changed setting or code requires a new experiment name.
- **After the run:** the repository is updated with the result.
