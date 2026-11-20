# Run v22: do the results of run v21 hold on the other sizes and on TSPLIB?

Run v21 found, on 10,000-node instances, that LKH-3 restarted with the BetaTour mixed lists in corrected candidate
files gave shorter tours than uninterrupted POPMUSIC, that lists from tour counting did as well, and that a restart
with a file equivalent to LKH's own cost nothing measurable at the budget. Its protocol required the comparison to be
repeated on the other instance classes before the paper states a gain. This run is that repetition, with the module
and the rules of run v21. The protocol is in `../PROTOCOL_v22.md` and was written before the run.

## Design

- 43 instances: 16 new instances of 1,000 nodes (budget 5 s), 16 new instances of 20,000 nodes (budget 300 s) and the
  11 TSPLIB instances of the earlier runs (budget max(10, 0.02 n) s); six LKH-3.0.14 processes (seeds 101-106).
- Four arms: uninterrupted POPMUSIC-5 (`pop_full`); restart with the seed's own lists, alpha-values and fathers
  (`pop_native`); restart with the BetaTour mixed lists (`beta_tree`) or the counting mixed lists (`freq_tree`) in
  the tree encoding.
- Primary comparisons, exact sign test over the 43 instances with Holm correction, on the mean of the six paired
  seeds at the budget: `beta_tree` versus `pop_full` and `beta_tree` versus `freq_tree`.

## Files

- `cells/`: the sources of the notebook cells. The experiment functions are those of run v21 without its two arms
  for the old file format; the instance cell adds the sizes and TSPLIB, the self-check cell a check of every TSPLIB
  instance before the run, the analysis cell the groups of this run.
- `build_colab_notebook.py`: writes `notebooks/24_BetaTour_v22_corrected_files_other_sizes.ipynb` and embeds
  `../v21_alpha_encoding/alphaenc.py` unchanged.
- `tests/run_notebook_locally.py`: runs the notebook's code cells outside Colab in test mode (two seeds, two small
  synthetic instances, two generated TSPLIB-format files, one of them with CEIL_2D distances) and imitates a lost
  session between or inside instances. The unit tests of the module are those of run v21.

```
BETATOUR_V22_TEST=1 BETATOUR_V22_REPO=. BETATOUR_V22_LKH=/path/to/LKH BETATOUR_V22_WORK=/tmp/v22_test \
    python experiments/v22_other_sizes/tests/run_notebook_locally.py
```

## Running it

Open the notebook in Colab with an A100 or L4 GPU and high RAM (12 vCPUs) and run all cells. Estimated time: set-up,
LKH build and TSPLIB download 4-6 min; warm-up, self-check and the TSPLIB check 2 min; 16 instances of 1,000 nodes
7 min; 16 instances of 20,000 nodes 5.1 h (19 min each); 11 TSPLIB instances 2.3 h (4 to 24 min each); analysis and
archives 1 min; about 7.5 h in total. Completed instances are kept on Drive (`MyDrive/BetaTour/v22_sizes43`) and
skipped when the notebook is run again; an instance interrupted by a change of machine runs again from the start.
`MAX_NEW_INSTANCES` in the first cell limits the number of new instances per session (26, then 9, then the rest gives
three sessions of about 2.5 h).

## Result (run on 19 and 20 November 2026)

All 43 instances completed in one Colab session (NVIDIA A100, 12 vCPUs; runs v21 and v23 used an L4 runtime with the
same number of vCPUs and the same reported CPU model). The self-check and the check of every TSPLIB instance passed
on the machine of the run, the first trial of every uninterrupted process equalled the shared first phase, and the
tree rule reproduced LKH's zero alpha-values for every seed.

Two deviations from the protocol:

1. The protocol file was written about an hour before the run started, but it was committed to the repository
   together with the results; unlike that of run v21, it has no public timestamp before the run. The primary
   comparisons and their endpoint are part of the analysis cell of the notebook, and the manifest that the notebook
   wrote before the first instance holds the SHA-256 over the notebook's code cells (`notebook_build`, 86a1b095...).
2. On the 16 instances of 20,000 nodes, LKH-3's clock (the CPU time of the process) ran about 1.5% behind the wall
   clock. Every process of these instances was therefore stopped by the outside deadline, 3 s of wall time after
   the budget, before its own clock reached the budget: its last recorded trial ended 1.3 to 5.5 s before the budget
   in the uninterrupted arm and up to 4.6 s before it in the restarted arms. On d18512 the processes were stopped in
   the same way (uninterrupted arm 1.0 to 2.0 s, restarted arms up to 1.2 s before the budget), and on d15112 one
   uninterrupted process by 0.2 s. All arms of an instance were stopped at the same wall time. In the last 10 s that
   were recorded at 20,000 nodes, tours improved by 0.00004 to 0.00013% per second, and with every arm evaluated at
   the earliest last recorded trial of the instance, the counts of the primary comparisons were 26 / 4 / 13
   (unchanged) and 21 / 5 / 17.

Neither hypothesis was supported (exact two-sided sign test over the 43 instances, Holm-adjusted; mean of six seeds).

- **P1 (H3).** LKH-3 restarted with the BetaTour lists found shorter tours than uninterrupted POPMUSIC on 26
  instances and longer tours on 13; on 4 the two arms did not differ (mean difference -0.0057%, 95% interval -0.0127
  to +0.0002, p = 0.053, adjusted p = 0.107). The direction was that of run v21.
- **P2 (H4).** The BetaTour lists gave shorter tours than the counting lists on 22 instances and longer tours on 16;
  on 5 the two arms did not differ (mean difference -0.0008%, 95% interval -0.0034 to +0.0015, p = 0.418).

`beta_tree` against `pop_full` by class (descriptive; instances on which the restarted arm was shorter / equal /
longer):

| Class | Instances | Best of six | Mean difference (%) | Mean of six | Mean difference (%) | 95% interval |
|---|---|---|---|---|---|---|
| 1,000 nodes, uniform | 8 | 2 / 6 / 0 | -0.0005 | 7 / 1 / 0 | -0.0067 | -0.0116 to -0.0027 |
| 1,000 nodes, clustered | 8 | 0 / 8 / 0 | 0.0000 | 2 / 3 / 3 | -0.0122 | -0.0372 to +0.0006 |
| 20,000 nodes, uniform | 8 | 4 / 0 / 4 | -0.0009 | 4 / 0 / 4 | +0.0006 | -0.0042 to +0.0052 |
| 20,000 nodes, clustered | 8 | 4 / 0 / 4 | +0.0011 | 4 / 0 / 4 | +0.0078 | -0.0029 to +0.0205 |
| TSPLIB | 11 | 8 / 3 / 0 | -0.0057 | 9 / 0 / 2 | -0.0146 | -0.0299 to -0.0038 |
| All | 43 | 18 / 17 / 8 | -0.0015 | 26 / 4 / 13 | -0.0057 | -0.0127 to +0.0002 |

On the 1,000-node instances the best of six was the same length in both arms on 14 of 16, as the protocol expected;
on the uniform ones more processes reached the shortest tour of the instance with the BetaTour lists (36 of 48,
against 22 of 48 for uninterrupted POPMUSIC; clustered: 35 against 32). At 20,000 nodes there was no difference in
either family. On TSPLIB the restarted arm was shorter on eight instances, equal on the three smallest (which every
arm solved to the optimal length) and longer on none; its mean gap to the optimal length was 0.0069% against
0.0126%, and it reached the optimal length on five instances against three (rl5915 and rl5934 in addition). On
fl3795 one of its six processes reached the optimum, against five and six of the two POPMUSIC arms.

Other comparisons at the budget (mean of six, descriptive):

| First / second arm | All 43 instances | Mean difference (%) | 95% interval | 22 instances of 7,000 nodes or more | Mean difference (%) | 95% interval |
|---|---|---|---|---|---|---|
| `pop_native` / `pop_full` | 18 / 5 / 20 | -0.0015 | -0.0072 to +0.0023 | 6 / 0 / 16 | +0.0040 | +0.0016 to +0.0063 |
| `beta_tree` / `pop_native` | 26 / 4 / 13 | -0.0042 | -0.0086 to -0.0002 | 15 / 0 / 7 | -0.0038 | -0.0099 to +0.0023 |
| `freq_tree` / `pop_full` | 26 / 4 / 13 | -0.0049 | -0.0131 to +0.0022 | 14 / 0 / 8 | +0.0019 | -0.0042 to +0.0106 |
| `freq_tree` / `pop_native` | 24 / 4 / 15 | -0.0034 | -0.0086 to +0.0017 | 15 / 0 / 7 | -0.0021 | -0.0086 to +0.0062 |
| `beta_tree` / `pop_full` | 26 / 4 / 13 | -0.0057 | -0.0127 to +0.0002 | 13 / 0 / 9 | +0.0002 | -0.0055 to +0.0065 |
| `beta_tree` / `freq_tree` | 22 / 5 / 16 | -0.0008 | -0.0034 to +0.0015 | 15 / 0 / 7 | -0.0017 | -0.0060 to +0.0022 |

On the large instances the restart itself had a cost in the seed mean (`pop_native` / `pop_full`: longer on 16 of
22), which run v21 had not found at 10,000 nodes (+0.0007%). The restarted search began 3.1 to 3.4% of the budget
after the first trial of the uninterrupted process had ended at 20,000 nodes, and 2.5 to 3.2% after it at 10,000
nodes in run v21. The effect of the lists (`beta_tree` / `pop_native`) on the large instances was largest early:
-0.0200% at a quarter of the budget (shorter on 18 of 22), -0.0067% at half, -0.0052% at three quarters and -0.0038%
at the budget. Model against counting at 20,000 nodes pointed in opposite directions in the two families (uniform:
counting shorter on 5 of 8, +0.0056%, interval +0.0011 to +0.0107; clustered: BetaTour shorter on 6 of 8, -0.0063%,
interval -0.0143 to -0.0003).

Time course of P1 over all instances (mean of six): 19 / 0 / 24 at a quarter of the budget (+0.031%), 29 / 1 / 13 at
half (-0.010%), 27 / 4 / 12 at three quarters (-0.006%) and 26 / 4 / 13 at the budget.

Trials completed within the budget (mean of seeds and instances; `pop_full`, `pop_native`, `beta_tree`,
`freq_tree`): 695, 631, 526, 532 (1,000 nodes, uniform); 354, 324, 239, 262 (1,000 nodes, clustered); 273, 270, 230,
238 (20,000 nodes, uniform); 211, 206, 177, 187 (20,000 nodes, clustered); 1106, 1069, 859, 897 (TSPLIB).

On three instances single processes were far from the others at the budget: one uninterrupted process on
`s22_clu1k_005` (0.57% above the shortest tour of the instance), processes of both POPMUSIC arms on rl5934 (up to
0.35%) and processes of the BetaTour and counting arms on `s22_clu20k_006` (up to 0.23%). They move the means and
intervals of the seed-mean endpoint in both directions; the sign test counts each instance once.

Consequences, as fixed in the protocol: H3 was not supported with the direction of run v21, so the gain over
uninterrupted POPMUSIC is stated for 10,000-node instances only (runs v21 and v23) and the result for the other
classes is inconclusive, with the intervals above. H4 was not supported, so the gain at 10,000 nodes is not
attributed to the learned model; lists from tour counting did as well.

`python analysis/v22_other_sizes.py results/v22/v22_sizes43_results.zip` recomputes these numbers from the instance
records. The intervals above are those of the notebook (`summary/paired_stats.csv`); the script draws its own
bootstrap resamples and can differ in the last digit. The report inside the ZIP is in Turkish.
