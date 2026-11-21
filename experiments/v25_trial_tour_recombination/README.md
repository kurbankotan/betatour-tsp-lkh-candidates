# Run v25 (pilot): recombining the trial tours that LKH-3 discards

Every trial of LKH-3 ends in a locally optimal tour that is merged with the best tour and then discarded. This
pilot records these tours and gives them to GA-EAX (Nagata and Kobayashi, 2013) as its population, and it compares
the result with the six LKH-3 processes left running, at the same wall-clock budget. The protocol is in
`../PROTOCOL_v25.md`; the probe that led to the pilot is in `probe/README.md`. The pilot is exploratory: it decides
whether a confirmation run is prepared.

## Design

- 21 instances: 12 new instances of 10,000 nodes (budget 120 s), five TSPLIB instances with known optimum (rl11849,
  usa13509, brd14051, d15112, d18512; budget 0.02 s per node) and 4 new instances of 50,000 nodes (budget 600 s).
- `lkh6`: six LKH-3.0.14 processes (POPMUSIC-5, seeds 101-106) for the whole budget; the shortest of their tours.
- `trials_33`, `trials_50`, `trials_67`: the same processes stopped at that share of the budget; per process its
  best tour and its last 16 trial tours (102 tours) are recombined by GA-EAX in the remaining time.
  `trials_50` is the primary arm.
- `best6_33`, `best6_50`, `best6_67`: the same with the six best tours only (control).
- `cascade`: the known hybrid; up to 100 first-trial tours of LKH-3, then GA-EAX, within the budget.
- `eax_scratch`: reference; GA-EAX from scratch in one process with six times the budget (instances of at most
  20,000 nodes).
- All times are wall-clock times on one machine with 12 virtual CPUs.

## Files

- `byprod.py`: the three source patches (the recording calls in `FindTour.c` of LKH-3.0.14; time stamps and a
  wall-clock deadline in `env.cpp` of GA-EAX; a grid search for the lists of nearest cities in `evaluator.cpp` of
  GA-EAX, which returns the lists of the original loops in less than a second where the loops take about a minute
  at 50,000 nodes), reading of the recorded tours, populations, the GA-EAX runner and the paired statistics. Pure
  Python and NumPy.
- `cells/`: the sources of the notebook cells (configuration, set-up with both builds, instances, the arms,
  self-check, run loop, analysis, archive).
- `build_colab_notebook.py`: writes `notebooks/27_BetaTour_v25_trial_tour_recombination.ipynb` and embeds
  `byprod.py`.
- `tests/test_byprod.py`: tests of `byprod.py` with the real LKH-3.0.14 and GA-EAX sources (same search with and
  without recording, valid records, GA-EAX from recorded tours, deadline, the grid search against GA-EAX's own
  lists entry by entry, and the same GA-EAX result with and without it).
- `tests/run_notebook_locally.py`: runs the notebook's code cells outside Colab in test mode (two processes, three
  small instances) and imitates a lost session between or inside instances.
- `probe/`: record, scripts and logs of the development probe of 20 November 2026.

```
python experiments/v25_trial_tour_recombination/tests/test_byprod.py --src src \
    --lkh-source /path/to/LKH-3.0.14 --eax-source /path/to/GA-EAX --work /tmp/v25_unit
BETATOUR_V25_TEST=1 BETATOUR_V25_REPO=. BETATOUR_V25_LKH=/path/to/LKH-3.0.14/LKH \
    BETATOUR_V25_EAX_SRC=/path/to/GA-EAX BETATOUR_V25_WORK=/tmp/v25_test \
    python experiments/v25_trial_tour_recombination/tests/run_notebook_locally.py
```

## Running it

Open the notebook in Colab with an L4 GPU and high RAM (12 vCPUs, 53 GB; the GPU is not used) and run all cells.
Estimated time: set-up with two builds of LKH-3 and two of GA-EAX 4-6 min; self-check 1 min; 12 instances of 10,000
nodes at about 6 min each, 1.2 h; five TSPLIB instances at 11 to 17 min each, 1.1 h; four instances of 50,000 nodes
at about 28 min each, 1.9 h; the GA-EAX reference in three groups, 1.0 h; analysis and archive 1 min; about 5.3 h in
total. Completed instances are kept on Drive (`MyDrive/BetaTour/v25_trialtours21`) and skipped when the notebook is
run again; an instance interrupted by a change of machine runs again from the start. `MAX_NEW_INSTANCES = 12` in the
first cell ends the first session after the instances of 10,000 nodes (about 1.3 h); `RUN_REFERENCE = False` leaves
out the reference.

## Result (run on 20 November 2026)

All 21 instances and the 17 reference runs completed in one Colab session (NVIDIA L4, 12 vCPUs, 53 GB of RAM). The
measurements took 4.6 h, from the first LKH-3 process (11:19 UTC) to the last reference run (15:57 UTC). The
self-check passed on the machine of the run: the recording copy of LKH-3 gave the same cost at each of 25 trials as
the unmodified LKH-3, and GA-EAX with the grid search returned the same tour after the same number of generations as
a build without it, in both versions. All 126 processes of `lkh6` were stopped from outside as designed, the
notebook's checks of the population tours and of the result tours raised no error, and its report lists no warning.

One point differs from the protocol, whose last limit says that the repository is updated after the result. The
protocol, the code and the notebook were committed at 11:20:47 UTC (commit 121b4e5, whose message does not name run
v25), a minute and a half after the first LKH-3 process had started and three and a half minutes before the first
instance finished. The commit therefore precedes every result of the run and does not precede its start, and the
repository does not record when the commit was pushed. The readings and the decision rule are part of the analysis
cell of the notebook, and the manifest that the notebook wrote before the first instance holds the SHA-256 over the
notebook's code cells (`notebook_build`, 8e6b71c0...).

### Decision rule

Differences at the budget; instances on which the first arm was shorter / equal / longer:

| Reading | Instances | Mean difference (%) | 95% interval | Sign test p | Holds |
|---|---|---|---|---|---|
| D1: `trials_50` / `lkh6`, 10,000 nodes | 12 / 0 / 0 | -0.0139 | -0.0172 to -0.0106 | 0.0005 | yes |
| D2: `trials_50` / `lkh6`, 50,000 nodes | 4 / 0 / 0 | -0.0189 | -0.0255 to -0.0125 | 0.125 | yes |
| D3: `trials_50` / `best6_50`, 10,000 nodes (informative) | 12 / 0 / 0 | -0.0066 | -0.0100 to -0.0036 | 0.0005 | yes |

D1 and D2 held, so the rule calls for a confirmation run. On the five TSPLIB instances, which the rule does not use,
`trials_50` was shorter than `lkh6` on all five (mean -0.0208%, interval -0.0294 to -0.0106, p = 0.0625), and over
the 21 instances it was shorter on all 21 (mean -0.0165%, interval -0.0198 to -0.0130).

### All arms against `lkh6`

Instances on which the arm was shorter / equal / longer than `lkh6`, and the mean difference (%):

| Arm | 10,000 nodes (12) | TSPLIB (5) | 50,000 nodes (4) |
|---|---|---|---|
| `trials_33` | 12 / 0 / 0, -0.0136 | 5 / 0 / 0, -0.0193 | 3 / 0 / 1, -0.0134 |
| `trials_50` (primary) | 12 / 0 / 0, -0.0139 | 5 / 0 / 0, -0.0208 | 4 / 0 / 0, -0.0189 |
| `trials_67` | 12 / 0 / 0, -0.0122 | 5 / 0 / 0, -0.0182 | 4 / 0 / 0, -0.0123 |
| `best6_33` | 7 / 0 / 5, -0.0023 | 5 / 0 / 0, -0.0103 | 4 / 0 / 0, -0.0610 |
| `best6_50` (control) | 9 / 0 / 3, -0.0073 | 5 / 0 / 0, -0.0133 | 4 / 0 / 0, -0.0662 |
| `best6_67` | 12 / 0 / 0, -0.0100 | 5 / 0 / 0, -0.0145 | 4 / 0 / 0, -0.0706 |
| `cascade` | 8 / 0 / 4, -0.0070 | 5 / 0 / 0, -0.0187 | 0 / 0 / 4, +0.0512 |
| `eax_scratch` (reference) | 4 / 0 / 8, +0.0189 | 4 / 0 / 1, -0.0060 | not run |

### The two populations

At 10,000 nodes and on TSPLIB the population with the trial tours gave shorter tours than the six best tours alone:
`trials_50` was shorter than `best6_50` on 12 of the 12 instances of 10,000 nodes (D3) and on 4 of the 5 TSPLIB
instances, with one equal (mean -0.0075%). At 50,000 nodes the order was the opposite, which the rule did not
anticipate: `best6_50` was 0.0662% shorter than `lkh6` and `trials_50` 0.0189%, and `trials_50` was longer than
`best6_50` on all four instances (mean +0.0474%). There GA-EAX did not converge on the 102 tours within its share
of the budget at any cut (about 400, 300 and 200 s; it was stopped at its deadline on all four instances), whereas
on the six tours it converged after 68 to 124 s. At 10,000 nodes GA-EAX converged on the 102 tours on 12, 10 and 5
of the 12 instances with shares of 80, 60 and 40 s, and on the six tours within about 4 s.

D2 was written for the population with the trial tours and held as written (a larger mean difference than at
10,000 nodes, -0.0189% against -0.0139%, on four instances). At 50,000 nodes, however, the six best tours alone
gave a larger difference from `lkh6` than the 102 tours, so the result at this size is not evidence for the trial
tours. One reading of these observations is that the larger population pays only when GA-EAX has the time to
converge on it. The run does not test this reading: the cuts changed the time of LKH-3 and the time of GA-EAX
together, and only two population sizes were used.

### Distance to the optimum on the TSPLIB instances (%)

| Instance | `lkh6` | `best6_50` | `trials_50` | `cascade` | `eax_scratch` |
|---|---|---|---|---|---|
| rl11849 | 0.0409 | 0.0160 | 0.0160 | 0.0087 | 0.0000 |
| usa13509 | 0.0039 | 0.0025 | 0.0010 | 0.0031 | 0.0005 |
| brd14051 | 0.0215 | 0.0119 | 0.0019 | 0.0036 | 0.0094 |
| d15112 | 0.0353 | 0.0184 | 0.0012 | 0.0174 | 0.0631 |
| d18512 | 0.0248 | 0.0112 | 0.0025 | 0.0003 | 0.0234 |
| mean | 0.0253 | 0.0120 | 0.0045 | 0.0066 | 0.0193 |

Neither an arm nor the reference had the shortest tour on all five instances: `eax_scratch` had it on rl11849 and
usa13509, `trials_50` on brd14051 and d15112, and `cascade` on d18512. Run v22 measured the settings of `lkh6` on
the same instances with the same budget per node on another machine type (`pop_full`, best of six seeds: 0.0250%
on average) and found 0.0144% for the restarted arm with BetaTour lists (`beta_tree`); the two runs are not paired.

### The known hybrid and GA-EAX from scratch

`cascade` was shorter than `lkh6` on 8 of the 12 instances of 10,000 nodes and on the five TSPLIB instances. It was
longer on the four instances of 50,000 nodes, where it had 94 first-trial tours after 202 s on average and its
GA-EAX did not converge. `trials_50` was shorter than `cascade` on 9 of 12, 3 of 5 and 4 of 4 instances (means
-0.0069%, -0.0021% and -0.0701%; sign test p = 0.15, 1.0 and 0.125).

GA-EAX from scratch, with six times the budget in one process, had not converged on any of the 12 instances of
10,000 nodes at its limit of 720 s and was longer than `lkh6` on 8 of them (mean +0.0189%; on the six uniform
instances it was longer on all six, mean +0.0377%). On TSPLIB it converged on 2 of 5 instances (limits of 1,422 to
2,221 s), was shorter than `lkh6` on 4 and reached the optimum of rl11849. In the probe, GA-EAX from scratch had
converged after 738 CPU-seconds on one uniform instance and had been 0.0276% shorter than the best of six
processes; the pilot did not repeat that reading at 10,000 nodes. Up to six reference runs shared the machine, so
the reference shows what GA-EAX from scratch reached within the wall-clock limit on this machine, not its result at
convergence.

### The primary arm in numbers (class means)

| Class | Trials per process before the cut | Union edges per node of the 102 tours | Share of GA-EAX (s) | GA-EAX ran for (s) | GA-EAX converged | Final length of `lkh6` reached at |
|---|---|---|---|---|---|---|
| 10,000 nodes | 124 | 1.464 | 60 | 45 | 10 / 12 | 0.60 of the budget |
| TSPLIB | 227 | 1.446 | 146 | 92 | 4 / 5 | 0.55 of the budget |
| 50,000 nodes | 58 | 1.567 | 299 | 300 (deadline) | 0 / 4 | 0.77 of the budget |

The time of GA-EAX includes writing its tour after the deadline; only tours found within its share count. On every
instance `trials_50` held a tour no longer than the final tour of `lkh6` before the budget ended. After the cut the
arm used one core, so its process-seconds were 0.55 to 0.58 of those of `lkh6`. In the second half of the budget
the six LKH-3 processes of `lkh6` shortened their best tour by 0.011% (10,000 nodes), 0.006% (TSPLIB) and 0.025%
(50,000 nodes).

### A combination chosen after the results (exploratory)

Keeping the shorter of `trials_50` and `best6_50` on each instance would have given -0.0139% (10,000 nodes),
-0.0208% (TSPLIB) and -0.0662% (50,000 nodes) against `lkh6`. Both populations could run at the same time on two of
the six cores that are free after the cut. This combination was chosen after seeing the results and was not an arm
of the run; if a confirmation run uses it, that run has to fix it beforehand.

### Limits

- The pilot is exploratory: 21 instances, comparisons without correction and no confirmatory test; D2 rests on four
  instances.
- `lkh6` and the merge arms share the recorded run up to the cut, so the comparisons measure what was done with the
  rest of the budget.
- `lkh6` is the shortest of six independent tours. An arm in which LKH-3 itself merges the six tours at the cut was
  not run (the probe tried it on one instance), so the run does not show that GA-EAX is needed for the difference.
- GA-EAX runs with a fixed random seed, so every population was recombined once.
- One LKH-3 version, one GA-EAX version, one machine type, and wall-clock time on a shared cloud machine. The six
  GA-EAX processes of the merge arms of an instance ran at the same time; the GA-EAX of `cascade` ran alone.
- No claim of novelty is made (see the protocol).

### What follows

By the rule a confirmation run is prepared: new instances, one primary comparison fixed beforehand, and a protocol
that is committed and pushed before the run starts. This is run v26; its result is in
`../v26_recombination_confirmation/README.md`.

`python analysis/v25_trial_tour_recombination.py results/v25/v25_trialtours21_results.zip` recomputes these numbers
from the recorded time courses, and with `--tours src [TSPLIB_DIR]` also the lengths of the archived tours. The
intervals above are those of the notebook (`summary/paired_stats.csv`, 5,000 resamples); the script draws 20,000
resamples of its own and can differ in the last digit. The report inside the ZIP is in Turkish.
