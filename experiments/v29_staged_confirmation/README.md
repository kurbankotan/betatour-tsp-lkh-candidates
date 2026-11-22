# Run v29 (confirmation run): staged merging of the tours at the cut

Run v28 (`../v28_staged_merging/README.md`), a pilot, found that merging the 102 tours of the six LKH-3 processes in
two stages (six GA-EAX groups at the same time, then the group results with the six best tours) gave shorter tours at
20,000 nodes than the GA-EAX arm of run v26 (10 of 12 instances) and than LKH-3's own merging (12 of 12). This run
repeats the comparison with the same settings on 18 new synthetic instances of 20,000 and 50,000 nodes, with two
hypotheses and the correction fixed before the run (`../PROTOCOL_v29.md`), and adds five TSPLIB instances with known
optimum as a descriptive check. The notebook refuses to start unless the protocol and the notebook itself are on the
main branch on GitHub as built.

## Design

- 23 instances: 12 new instances of 20,000 nodes (6 uniform, 6 clustered; 240 s), 6 new instances of 50,000 nodes
  (3 + 3; 600 s) and the TSPLIB instances rl11849, usa13509, brd14051, d15112 and d18512 (0.02 s per node,
  descriptive).
- One recorded run per instance and the arms of run v28, unchanged: `lkh6`, `trials_50`, `best6_50`, `dual_50`,
  `lkhmerge_50` and `staged_50`.
- Hypotheses (18 synthetic instances, exact sign test, Holm over the two): H1 `staged_50` / `lkhmerge_50`, H2
  `staged_50` / `dual_50`. Staged merging is recorded as confirmed only if both are supported.

## Files

- `cells/`: the sources of the notebook cells; they are those of run v28 with the instances of this run (including
  the TSPLIB instances, which the set-up downloads as run v26 did), the optimum in each instance record and the
  analysis of the two hypotheses with Holm correction. The recorded run, GA-EAX and LKH-3 merging use the code of run
  v26 unchanged (`../v26_recombination_confirmation/byprod.py`, embedded as `byprod_v26.py`).
- `build_colab_notebook.py`: writes `notebooks/31_BetaTour_v29_staged_confirmation.ipynb`, embeds the module and
  writes the SHA-256 of the protocol, of the module and of the build into the first cell.
- `tests/run_notebook_locally.py`: runs the notebook's code cells outside Colab in test mode (two processes, three
  small synthetic instances, one TSPLIB-format file written locally, two groups) and imitates a lost session between
  or inside instances.

```
BETATOUR_V29_TEST=1 BETATOUR_V29_REPO=. BETATOUR_V29_LKH=/path/to/LKH-3.0.14/LKH \
    BETATOUR_V29_EAX_SRC=/path/to/GA-EAX BETATOUR_V29_WORK=/tmp/v29_test \
    python experiments/v29_staged_confirmation/tests/run_notebook_locally.py
```

## Running it

Commit and push the protocol and the notebook first. Then open the notebook in Colab with an L4 GPU and high RAM
(12 vCPUs, 53 GB; the GPU is not used) and run all cells. Estimated time: set-up (two builds of LKH-3, four of GA-EAX
and the TSPLIB files) 5-7 min; self-check 2-3 min; 12 instances of 20,000 nodes at about 11.5 min each, 2.3 h; 6 of
50,000 nodes at about 28 min each, 2.8 h; 5 TSPLIB instances at 11-17 min each, 1.1 h; analysis and archive 1 min;
about 6.3 h in total. Completed instances are kept on Drive (`MyDrive/BetaTour/v29_staged23`) and skipped when the
notebook is run again; an instance interrupted by a change of machine runs again from the start. With
`MAX_NEW_INSTANCES = 12` in the first cell the run takes two sessions.

## Result (run on 22 November 2026)

All 23 instances completed in one Colab session (NVIDIA L4, not used; 12 vCPUs; 53 GB of RAM); they ran from 05:43 to
10:43 UTC (5.0 h). The protocol and the notebook were committed at 05:40:19 UTC (commit 90817ae), and the notebook found
both on GitHub as built at 05:42:16 UTC, before the first instance. The self-check passed on the machine of the run,
including the staged arm on a small instance. All 138 processes of `lkh6` were stopped from outside as designed, the
populations held 102 and 6 different tours and every stage-1 group 16, all six merging processes recorded a tour
within their time on every instance, and every stage value in the records is the best tour its process had found by
its deadline. There was no deviation from the protocol.

### Tests of the protocol

18 synthetic instances (12 of 20,000 and 6 of 50,000 nodes); instances on which the first arm was shorter / equal /
longer:

| Test | Instances | Mean difference (%) | 95% interval | Sign test p | Holm p | Supported |
|---|---|---|---|---|---|---|
| H1: `staged_50` / `lkhmerge_50` | 18 / 0 / 0 | -0.0213 | -0.0261 to -0.0166 | 7.6e-6 | 1.5e-5 | yes |
| H2: `staged_50` / `dual_50` | 17 / 0 / 1 | -0.0158 | -0.0203 to -0.0113 | 1.4e-4 | 1.4e-4 | yes |

Both hypotheses were supported, so by the decision rule staged merging is recorded as confirmed at these sizes and
budgets. On the one instance on which `staged_50` was longer than `dual_50` (`s29_uni20k_005`) the difference was
0.0002%.

### By size

Instances on which the first arm was shorter / equal / longer, and the mean difference (%):

| Comparison | 20,000 nodes (12) | 50,000 nodes (6) | TSPLIB (5, descriptive) |
|---|---|---|---|
| `staged_50` / `lkhmerge_50` | 12 / 0 / 0, -0.0183 | 6 / 0 / 0, -0.0274 | 4 / 1 / 0, -0.0061 |
| `staged_50` / `dual_50` | 11 / 0 / 1, -0.0102 | 6 / 0 / 0, -0.0270 | 0 / 1 / 4, +0.0015 |
| `staged_50` / `lkh6` | 12 / 0 / 0, -0.0324 | 6 / 0 / 0, -0.0856 | 5 / 0 / 0, -0.0150 |
| `staged_50` / `trials_50` | 11 / 0 / 1, -0.0102 | 6 / 0 / 0, -0.0689 | 0 / 1 / 4, +0.0015 |
| `dual_50` / `lkhmerge_50` | 10 / 0 / 2, -0.0081 | 3 / 0 / 3, -0.0004 | 4 / 1 / 0, -0.0076 |
| `trials_50` / `best6_50` | 12 / 0 / 0, -0.0084 | 0 / 0 / 6, +0.0420 | 4 / 1 / 0, -0.0054 |
| `dual_50` / `lkh6` | 12 / 0 / 0, -0.0222 | 6 / 0 / 0, -0.0587 | 5 / 0 / 0, -0.0164 |
| `lkhmerge_50` / `lkh6` | 12 / 0 / 0, -0.0141 | 6 / 0 / 0, -0.0583 | 4 / 0 / 1, -0.0088 |

At 20,000 nodes all 72 stage-1 groups converged, after 37 s on average; stage 2 converged after 11 s, and the arm
stopped after 48 s on average (28% to 49% of the 120 s left after the cut). At 50,000 nodes no stage-1 group converged
within its 180 s, stage 2 converged after 80 s on average, and the arm used 80% to 94% of the 300 s. GA-EAX on the 102
tours in one process converged on none of the 18 synthetic instances within its time. After the cut the staged arm
used 205 process-seconds at 20,000 nodes and 1,156 at 50,000 nodes on average, against 129 and 373 for `dual_50` and
732 and 1,811 for `lkhmerge_50`.

### TSPLIB (descriptive)

With 0.02 s per node, GA-EAX converged on the 102 tours within its time on all five TSPLIB instances, and so did the
staged arm, which stopped after 9% to 24% of the time left. Distance to the optimum (%):

| Instance | `lkh6` | `best6_50` | `trials_50` | `dual_50` | `lkhmerge_50` | `staged_50` |
|---|---|---|---|---|---|---|
| rl11849 | 0.0339 | 0.0168 | 0.0168 | 0.0168 | 0.0168 | 0.0168 |
| usa13509 | 0.0213 | 0.0041 | 0.0023 | 0.0023 | 0.0059 | 0.0026 |
| brd14051 | 0.0219 | 0.0102 | 0.0040 | 0.0040 | 0.0164 | 0.0053 |
| d15112 | 0.0135 | 0.0123 | 0.0039 | 0.0039 | 0.0144 | 0.0048 |
| d18512 | 0.0206 | 0.0127 | 0.0020 | 0.0020 | 0.0136 | 0.0068 |
| mean | 0.0223 | 0.0112 | 0.0058 | 0.0058 | 0.0134 | 0.0073 |

Here the staged arm was shorter than LKH-3's merging on four instances and equal on one, but one GA-EAX process on all
102 tours was shorter than the staged arm on four instances and equal on one.

### What the run shows

- Confirmed (H1, H2): on 18 new instances of 20,000 and 50,000 nodes with 12 ms per node, merging the 102 tours of the
  six LKH-3 processes in two stages at half of the budget gave shorter tours at the budget than LKH-3's own merging on
  all 18 instances and than the GA-EAX arm of run v26 on 17; the differences were larger at 50,000 nodes (0.027% from
  both) than at 20,000 nodes (0.018% and 0.010%).
- The staged arm finished its merge within the time at 20,000 nodes and its second stage at 50,000 nodes, while one
  GA-EAX process on the 102 tours finished on none of the synthetic instances. On the TSPLIB instances, where the
  budget per node was longer and that process finished too, it gave slightly shorter tours than the staged arm (mean
  0.0015%). The run does not test why.
- At 20,000 nodes and on TSPLIB the staged arm left most of the time after the cut unused; this run does not test
  what that time could add.
- Limits: one budget per node for the synthetic instances, one cut, one number of groups, one division of the time
  between the stages, one way of forming the groups, one LKH-3 version, one GA-EAX version with its fixed random seed,
  one machine type; the TSPLIB instances had been used before and are descriptive. Island models and parallel GA-EAX
  exist; no claim of novelty follows from this run.

`python analysis/v29_staged_confirmation.py results/v29/v29_staged23_results.zip` recomputes these numbers from the
records, and with `--tours src [TSPLIB_DIR]` also the lengths of the archived tours. The intervals above are those of
the notebook (`summary/paired_stats.csv`, 5,000 resamples); the script draws 20,000 resamples of its own and can differ
in the last digit. The report inside the ZIP is in Turkish.
