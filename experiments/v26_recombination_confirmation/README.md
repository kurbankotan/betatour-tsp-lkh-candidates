# Run v26: confirmation run on recombining the tours of LKH-3 processes with GA-EAX

Run v25 (pilot, `../v25_trial_tour_recombination/README.md`) stopped six LKH-3 processes at half of the wall-clock
budget and recombined their tours with GA-EAX (Nagata and Kobayashi, 2013); its rule was met on 21 of 21 instances.
This run repeats the comparison on 28 new synthetic instances with four hypotheses fixed beforehand, a primary arm
that runs both populations of the pilot at the same time, and a new control in which LKH-3 merges its own six tours.
The protocol is `../PROTOCOL_v26.md`; the notebook refuses to start unless that file and the notebook itself are on
the main branch on GitHub as built.

## Design

- 38 instances: 16 of 10,000 nodes (120 s), 6 of 20,000 nodes (240 s) and 6 of 50,000 nodes (600 s), all new and
  half uniform, half clustered; ten TSPLIB instances with known optimum (pcb3038, fl3795, fnl4461, rl5915, rl5934 and
  the five of the pilot; 0.02 s per node), which do not enter the tests.
- `lkh6`: six LKH-3.0.14 processes (POPMUSIC-5, seeds 201-206) for the whole budget; the shortest of their tours.
- `dual_50` (primary): the same processes taken as stopped at half of the budget; two GA-EAX processes run at the
  same time for the rest, one on the best tour and the last 16 trial tours of each process (`trials_50`, 102 tours),
  one on the six best tours (`best6_50`); the shorter result counts.
- `lkhmerge_50` (control): the six best tours at half of the budget given to six new LKH-3 processes as
  `MERGE_TOUR_FILE`s (LKH-3's own tour merging), for the rest of the budget.
- `cascade` (known hybrid): up to 100 first-trial tours of LKH-3, then GA-EAX, within the budget.
- `eax_scratch` (reference): GA-EAX from scratch with six times the budget in one process; 10,000 nodes and the five
  smaller TSPLIB instances only.
- Tests (28 synthetic instances, exact sign test, Holm over four tests): H1 `dual_50` / `lkh6`, H2 `dual_50` /
  `lkhmerge_50`, H3 `dual_50` / `cascade`; H4 `trials_50` / `best6_50` on the 16 instances of 10,000 nodes.
- All times are wall-clock times on one machine with 12 virtual CPUs.

## Files

- `byprod.py`: the module of run v25 with three additions: a mode of the recording patch that writes only the new
  best tours (`BETATOUR_BEST_ONLY`, for the merging processes), `MERGE_TOUR_FILE` lines in `write_par`, and helpers
  for common edges, distinct tours and the Holm correction. Pure Python and NumPy.
- `cells/`: the sources of the notebook cells (configuration, set-up with the check of the protocol on GitHub,
  instances, the arms, self-check, run loop, analysis, archive).
- `build_colab_notebook.py`: writes `notebooks/28_BetaTour_v26_recombination_confirmation.ipynb`, embeds `byprod.py`
  and writes the SHA-256 of the protocol and of the build into the first cell.
- `tests/test_byprod.py`: tests of `byprod.py` with the real LKH-3.0.14 and GA-EAX sources (the tests of run v25,
  plus the best-only recording, LKH-3's merging with the recorded candidate file and the helpers).
- `tests/run_notebook_locally.py`: runs the notebook's code cells outside Colab in test mode (two processes, four
  small instances) and imitates a lost session between or inside instances.

```
python experiments/v26_recombination_confirmation/tests/test_byprod.py --src src \
    --lkh-source /path/to/LKH-3.0.14 --eax-source /path/to/GA-EAX --work /tmp/v26_unit
BETATOUR_V26_TEST=1 BETATOUR_V26_REPO=. BETATOUR_V26_LKH=/path/to/LKH-3.0.14/LKH \
    BETATOUR_V26_EAX_SRC=/path/to/GA-EAX BETATOUR_V26_WORK=/tmp/v26_test \
    python experiments/v26_recombination_confirmation/tests/run_notebook_locally.py
```

## Running it

Commit and push the protocol and the notebook first. Then open the notebook in Colab with an L4 GPU and high RAM
(12 vCPUs, 53 GB; the GPU is not used) and run all cells. Estimated time: set-up 5-7 min; self-check 2 min; 16
instances of 10,000 nodes at about 6 min each, 1.6 h; 6 of 20,000 nodes at about 12 min, 1.2 h; 6 of 50,000 nodes at
about 31 min, 3.1 h; the five smaller TSPLIB instances 0.4 h; the five larger 1.2 h; the GA-EAX reference 0.8 h;
analysis and archive 1 min; about 8.5 h in total (at most about 9 h). Completed instances are kept on Drive
(`MyDrive/BetaTour/v26_recomb38`) and skipped when the notebook is run again; an instance interrupted by a change of
machine runs again from the start. `MAX_NEW_INSTANCES` in the first cell splits the run into sessions (22, 6 and
`None` give sessions of about 3, 3.1 and 2.5 h).

## Result (run on 20 and 21 November 2026)

All 38 instances and the 21 reference runs completed in one Colab session (NVIDIA L4, 12 vCPUs, 53 GB of RAM) in
7.7 h, from the first LKH-3 process (19:42 UTC on 20 November) to the last reference run (03:24 UTC on 21 November).
The protocol and the notebook were committed at 19:39:28 UTC (commit 9c777eb), and the notebook found both on GitHub
as built at 19:41:20 UTC, before the first instance. The self-check passed on the machine of the run: the recording
copy of LKH-3 gave the same cost at each of 25 trials as the unmodified LKH-3 while it wrote its candidate file,
GA-EAX with the grid search gave the same result as GA-EAX without it, and LKH-3's merging read the candidate file,
gave the same search with and without recording, kept the common edges and did not lengthen the tour. All 228
processes of `lkh6` were stopped from outside as designed, every population held at least two different tours, all
six merging processes recorded a tour within their time on every instance, and the report lists no warning. There
was no deviation from the protocol.

### Tests of the protocol

28 synthetic instances (H4: the 16 of 10,000 nodes); instances on which the first arm was shorter / equal / longer:

| Test | Instances | Mean difference (%) | 95% interval | Sign test p | Holm p | Supported |
|---|---|---|---|---|---|---|
| H1: `dual_50` / `lkh6` | 27 / 0 / 1 | -0.0240 | -0.0326 to -0.0161 | 2.2e-7 | 8.6e-7 | yes |
| H2: `dual_50` / `lkhmerge_50` | 18 / 0 / 10 | -0.0034 | -0.0060 to -0.0010 | 0.18 | 0.18 | no |
| H3: `dual_50` / `cascade` | 24 / 0 / 4 | -0.0315 | -0.0519 to -0.0144 | 1.8e-4 | 3.6e-4 | yes |
| H4: `trials_50` / `best6_50`, 10,000 nodes | 15 / 1 / 0 | -0.0067 | -0.0089 to -0.0045 | 6.1e-5 | 1.8e-4 | yes |

H1, H3 and H4 were supported and H2 was not. Stopping the six LKH-3 processes at half of the budget and recombining
their tours with two GA-EAX processes gave shorter tours than letting the processes run and shorter tours than the
known hybrid, and at 10,000 nodes the trial tours that LKH-3 discards added to the six best tours. LKH-3's own merging
of the six best tours at the same moment, however, gave nearly the same gain: `lkhmerge_50` was shorter than `lkh6` on
25 of the 28 instances (mean -0.0206%, against -0.0240% for `dual_50`). The mean difference between `dual_50` and
`lkhmerge_50` was small and its bootstrap interval excluded zero, but the sign test fixed in the protocol was not
significant. As stated there, GA-EAX is therefore not shown to be needed for the difference.

### By size

Instances on which the first arm was shorter / equal / longer, and the mean difference (%):

| Comparison | 10,000 nodes (16) | 20,000 nodes (6) | 50,000 nodes (6) |
|---|---|---|---|
| `dual_50` / `lkh6` | 15 / 0 / 1, -0.0125 | 6 / 0 / 0, -0.0152 | 6 / 0 / 0, -0.0637 |
| `lkhmerge_50` / `lkh6` | 13 / 0 / 3, -0.0068 | 6 / 0 / 0, -0.0175 | 6 / 0 / 0, -0.0605 |
| `dual_50` / `lkhmerge_50` | 13 / 0 / 3, -0.0057 | 1 / 0 / 5, +0.0023 | 4 / 0 / 2, -0.0032 |
| `trials_50` / `lkh6` | 15 / 0 / 1, -0.0125 | 6 / 0 / 0, -0.0120 | 6 / 0 / 0, -0.0191 |
| `best6_50` / `lkh6` | 13 / 0 / 3, -0.0058 | 6 / 0 / 0, -0.0132 | 6 / 0 / 0, -0.0637 |
| `best6_50` / `lkhmerge_50` | 4 / 2 / 10, +0.0010 | 0 / 0 / 6, +0.0043 | 4 / 0 / 2, -0.0032 |
| `trials_50` / `best6_50` | 15 / 1 / 0, -0.0067 | 2 / 0 / 4, +0.0012 | 0 / 0 / 6, +0.0447 |
| `cascade` / `lkh6` | 13 / 0 / 3, -0.0065 | 4 / 0 / 2, -0.0095 | 0 / 0 / 6, +0.0619 |
| `dual_50` / `cascade` | 14 / 0 / 2, -0.0060 | 4 / 0 / 2, -0.0057 | 6 / 0 / 0, -0.1256 |

At 10,000 nodes GA-EAX converged on the 102 tours on 14 of the 16 instances within its 60 s, these tours gave the
result of `dual_50` on 15 of them, and `dual_50` was shorter than LKH-3's merging on 13 of 16 instances. At 20,000 and
50,000 nodes GA-EAX did not converge on the 102 tours within its time (0 of 6 in each class), the six best tours gave
the result on 4 of 6 and 6 of 6 instances, and the two methods merged the six best tours about equally well (LKH-3
shorter on all six instances of 20,000 nodes, mean 0.0043%; GA-EAX shorter on 4 of 6 of 50,000 nodes). The difference
from `lkh6` grew with size for both merging arms; in the second half of the budget the processes of `lkh6` shortened
their best tour by 0.012%, 0.027% and 0.018%. On the one instance on which `dual_50` was longer than `lkh6`
(`c26_clu10k_003`, +0.022%) every other arm was longer than `lkh6` too.

### TSPLIB (descriptive)

On the five smaller instances (3,038 to 5,934 nodes) `lkh6` reached the optimum on four. The merging arms reached the
same lengths except on rl5915 (0.0085% above the optimum, against 0.0023% for `lkh6`); with 0.02 s per node these
instances do not separate the arms. On the five larger instances, the instances of the pilot with new seeds, the
distance to the optimum (%) was:

| Instance | `lkh6` | `best6_50` | `trials_50` | `dual_50` | `lkhmerge_50` | `cascade` |
|---|---|---|---|---|---|---|
| rl11849 | 0.0391 | 0.0168 | 0.0168 | 0.0168 | 0.0392 | 0.0000 |
| usa13509 | 0.0213 | 0.0047 | 0.0037 | 0.0037 | 0.0048 | 0.0023 |
| brd14051 | 0.0236 | 0.0089 | 0.0043 | 0.0043 | 0.0164 | 0.0013 |
| d15112 | 0.0135 | 0.0170 | 0.0041 | 0.0041 | 0.0129 | 0.0149 |
| d18512 | 0.0206 | 0.0135 | 0.0017 | 0.0017 | 0.0147 | 0.0160 |
| mean | 0.0236 | 0.0122 | 0.0061 | 0.0061 | 0.0176 | 0.0069 |

Here `dual_50` was shorter than LKH-3's merging on all five instances (mean -0.0115%), the 102 tours gave its result
on four, and the known hybrid was about as good on average (0.0069%; shortest of the arms on three instances). With
other seeds the pilot had found 0.0253% for `lkh6` and 0.0045% for `trials_50` on the same instances.

### Reference and cost

GA-EAX from scratch, with six times the budget in one process and up to six runs on the machine, converged on none of
the 16 instances of 10,000 nodes within 720 s and was longer than `lkh6` on 13 of them (mean +0.042%); `dual_50` was
shorter than it on 15 of 16 (mean -0.055%). On the five smaller TSPLIB instances it reached the optimum on four and
was 0.17% above it on fl3795. After the cut `dual_50` used two cores, so its process-seconds were 0.57, 0.59 and 0.61
of those of `lkh6` at 10,000, 20,000 and 50,000 nodes; `lkhmerge_50` used as many as `lkh6`.

### What the run shows

- Confirmed (H1, H3): on new instances of 10,000 to 50,000 nodes with 12 ms per node, merging the tours of six LKH-3
  processes at half of the budget gives shorter tours at the budget than letting the processes run, and the GA-EAX
  arm gives shorter tours than the known hybrid; the difference from the processes left running grows with size.
- Not confirmed (H2): the GA-EAX arm is not shown to be better than LKH-3's own tour merging. Most of the gain comes
  from merging the tours of the processes at all, which LKH-3 can do itself.
- Confirmed for 10,000 nodes only (H4): the discarded trial tours add to the six best tours where GA-EAX has time to
  converge on them; at 20,000 and 50,000 nodes with this budget they do not, and at 50,000 nodes they make the result
  worse (0 of 6).
- Limits: one budget per node, one cut, one LKH-3 version, one GA-EAX version with its fixed random seed, one
  machine type; no statement extends beyond them. Tour merging is old (Cook and Seymour, 2003) and is part of LKH-3;
  no claim of novelty follows from this run.

Run v27 (a pilot) followed with exact merging on the union graph of the six best tours, with and without edges ranked
by BetaTour; neither shortened the tours of `dual_50`. Its result is in `../v27_union_exact/README.md`.

`python analysis/v26_recombination_confirmation.py results/v26/v26_recomb38_results.zip` recomputes these numbers
from the recorded time courses, and with `--tours src [TSPLIB_DIR]` also the lengths of the archived tours. The
intervals above are those of the notebook (`summary/paired_stats.csv`, 5,000 resamples); the script draws 20,000
resamples of its own and can differ in the last digit. The report inside the ZIP is in Turkish.
