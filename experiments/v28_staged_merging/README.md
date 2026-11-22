# Run v28 (pilot): staged merging of the tours at the cut

Run v26 (`../v26_recombination_confirmation/README.md`) found that at 10,000 nodes GA-EAX gives shorter tours from the
102 tours of the six LKH-3 processes (the best tour and the last 16 trial tours of each) than from their six best
tours, but that at 20,000 and 50,000 nodes GA-EAX does not converge on the 102 tours in the time left after the cut.
Run v27 (`../v27_union_exact/README.md`) found that neither exact merging on the union graph of the six best tours nor
edges ranked by BetaTour shortened the GA-EAX merge. This pilot asks whether merging the 102 tours in two stages (six
groups at the same time, then the group results with the six best tours) gives shorter tours at 20,000 nodes than the
GA-EAX arm of run v26 and than LKH-3's own merging. The protocol is `../PROTOCOL_v28.md`; it states the evidence of two
technical checks before the run (exact repair of small windows gained far less than merging; staged merging matched
GA-EAX on the 102 tours on one instance of 20,000 nodes in less than half of the time) and the rule by which the run
decides whether a confirmation run is prepared. The notebook refuses to start unless that file and the notebook
itself are on the main branch on GitHub as built.

## Design

- 14 new instances: 12 of 20,000 nodes (6 uniform, 6 clustered; 240 s) for the readings and 2 of 50,000 nodes (1 + 1;
  600 s, descriptive).
- One recorded run per instance, as in runs v26 and v27: six POPMUSIC-5 LKH-3.0.14 processes (seeds 201-206) for the
  whole budget (`lkh6`). All other arms start from its tours at half of the budget.
- `trials_50`, `best6_50`, `dual_50`: GA-EAX on the 102 tours and on the six best tours, two processes at the same
  time, and the shorter result (run v26). `lkhmerge_50`: LKH-3's own merging of the six best tours (run v26).
- `staged_50`: stage 1 deals the 96 tours other than the six best tours in turn to six groups and runs six GA-EAX
  processes at the same time until 60% of the time left; stage 2 merges the six group results and the six best tours
  with one GA-EAX process until the budget.
- Readings (12 instances of 20,000 nodes, exact sign test, no correction): R1 `staged_50` / `dual_50`, R2 `staged_50`
  / `lkhmerge_50`. A confirmation run follows only if both hold.

## Files

- `cells/`: the sources of the notebook cells (configuration, set-up with the check of the protocol on GitHub,
  instances, the arms, self-check, run loop, analysis, archive). The recorded run, GA-EAX and LKH-3 merging use the
  code of run v26 unchanged (`../v26_recombination_confirmation/byprod.py`, embedded as `byprod_v26.py`); the staged
  arm is in `cells/c5_experiment.py` (`staged_groups`, `staged_merge`, `make_staged`), and the self-check runs it on a
  small instance.
- `build_colab_notebook.py`: writes `notebooks/30_BetaTour_v28_staged_merging_pilot.ipynb`, embeds the module and
  writes the SHA-256 of the protocol, of the module and of the build into the first cell.
- `tests/run_notebook_locally.py`: runs the notebook's code cells outside Colab in test mode (two processes, three
  small instances, two groups) and imitates a lost session between or inside instances.

```
BETATOUR_V28_TEST=1 BETATOUR_V28_REPO=. BETATOUR_V28_LKH=/path/to/LKH-3.0.14/LKH \
    BETATOUR_V28_EAX_SRC=/path/to/GA-EAX BETATOUR_V28_WORK=/tmp/v28_test \
    python experiments/v28_staged_merging/tests/run_notebook_locally.py
```

## Running it

Commit and push the protocol and the notebook first. Then open the notebook in Colab with an L4 GPU and high RAM
(12 vCPUs, 53 GB; the GPU is not used) and run all cells. Estimated time: set-up (two builds of LKH-3 and four of
GA-EAX) 5-7 min; self-check 2-3 min; 12 instances of 20,000 nodes at about 11.5 min each, 2.3 h; 2 of 50,000 nodes at
about 28 min each, 1 h; analysis and archive 1 min; about 3.4 h in total. Completed instances are kept on Drive
(`MyDrive/BetaTour/v28_staged14`) and skipped when the notebook is run again; an instance interrupted by a change of
machine runs again from the start. `MAX_NEW_INSTANCES` in the first cell splits the run into sessions.

## Result (run on 21 November 2026)

All 14 instances completed in one Colab session (NVIDIA L4, not used; 12 vCPUs; 53 GB of RAM); they ran from 19:26
to 22:03 UTC (2.6 h). The protocol and the notebook were committed at 19:19:36 UTC (commit d4244e7), and the notebook
found both on GitHub as built at 19:24:49 UTC, before the first instance. The self-check passed on the machine of the
run: besides the checks of run v26, the staged arm, run by the code of the experiment on a small instance (22 tours,
six groups), covered every tour but the best one exactly once, ran both stages and returned a valid tour not longer
than the best input. All 84 processes of `lkh6` were stopped from outside as designed, the populations held 102 and 6
different tours and every stage-1 group 16, all six merging processes recorded a tour within their time on every
instance, and every stage value in the records is the best tour its process had found by its deadline. There was no
deviation from the protocol.

### Readings of the protocol

12 instances of 20,000 nodes; instances on which the first arm was shorter / equal / longer:

| Reading | Instances | Mean difference (%) | 95% interval | Sign test p | Holds |
|---|---|---|---|---|---|
| R1: `staged_50` / `dual_50` | 10 / 0 / 2 | -0.0097 | -0.0146 to -0.0052 | 0.039 | yes |
| R2: `staged_50` / `lkhmerge_50` | 12 / 0 / 0 | -0.0131 | -0.0171 to -0.0093 | 0.00049 | yes |

Both readings held. On the two instances on which `staged_50` was longer than `dual_50`, the difference was 0.0004%.
By the decision rule a confirmation run of staged merging is prepared: new instances, including instances of 50,000
nodes, and a protocol pushed before the run.

### How the staged arm used its time

At 20,000 nodes all 72 stage-1 groups converged, after 38 s on average (their deadline was 72 s after the arm
started), stage 2 converged on all 12 instances after 11 s on average, and the arm stopped after 49 s on average,
between 31% and 49% of the 120 s left after the cut. GA-EAX on the 102 tours in one process converged on none of the
12 instances within the 120 s, and `staged_50` was shorter than it on 10 (mean -0.0101%). After the cut the staged arm
used 205 process-seconds on average, against 130 for `dual_50` and 732 for `lkhmerge_50`.

At 50,000 nodes no stage-1 group converged within its 180 s, stage 2 converged after 106 s on average, and the arm used
93% and 97% of the 300 s left. `staged_50` was the shortest arm on both instances: 0.038% and 0.030% shorter than
`dual_50` and 0.041% and 0.025% shorter than `lkhmerge_50`. GA-EAX on the 102 tours in one process was longer than on
the six best tours on both instances (mean +0.0475%), as at 50,000 nodes in run v26.

### Other comparisons (descriptive)

Instances on which the first arm was shorter / equal / longer, and the mean difference (%):

| Comparison | 20,000 nodes (12) | 50,000 nodes (2) |
|---|---|---|
| `staged_50` / `lkh6` | 12 / 0 / 0, -0.0302 | 2 / 0 / 0, -0.0955 |
| `staged_50` / `trials_50` | 10 / 0 / 2, -0.0101 | 2 / 0 / 0, -0.0812 |
| `dual_50` / `lkhmerge_50` | 8 / 0 / 4, -0.0033 | 1 / 0 / 1, +0.0006 |
| `trials_50` / `best6_50` | 11 / 0 / 1, -0.0051 | 0 / 0 / 2, +0.0475 |
| `dual_50` / `lkh6` | 12 / 0 / 0, -0.0205 | 2 / 0 / 0, -0.0618 |
| `lkhmerge_50` / `lkh6` | 11 / 0 / 1, -0.0171 | 2 / 0 / 0, -0.0624 |

As in run v26, the GA-EAX arm was not significantly shorter than LKH-3's merging (8 of 12 instances of 20,000 nodes,
p = 0.39). The staged arm was shorter than `dual_50` on 5 of the 6 uniform and 5 of the 6 clustered instances of
20,000 nodes (mean -0.0145% and -0.0050%), and shorter than `lkhmerge_50` on all 12.

### What the run shows

- Supported (R1, R2; pilot, no correction): at 20,000 nodes with 12 ms per node, merging the 102 tours of the six
  LKH-3 processes in two stages at half of the budget gave shorter tours at the budget than the GA-EAX arm of run v26
  (10 of 12 instances) and than LKH-3's own merging (12 of 12), and it stopped after less than half of the remaining
  time. On the two instances of 50,000 nodes it was the shortest arm by a wider margin, although its stage-1 groups
  did not converge.
- Limits: one budget per node, one cut, one number of groups, one division of the time between the stages, one way of
  forming the groups, one LKH-3 version, one GA-EAX version with its fixed random seed, one machine type; no statement
  extends beyond them. After the cut the arms use different numbers of cores (the staged arm six, then one). Island
  models and parallel GA-EAX exist; no claim of novelty follows from this run. The confirmation run decides whether
  the result holds on new instances.

The confirmation run (run v29) confirmed the result on 18 new instances of 20,000 and 50,000 nodes; it is in
`../v29_staged_confirmation/README.md`.

`python analysis/v28_staged_merging.py results/v28/v28_staged14_results.zip` recomputes these numbers from the
records, and with `--tours src` also the lengths of the archived tours. The intervals above are those of the notebook
(`summary/paired_stats.csv`, 5,000 resamples); the script draws 20,000 resamples of its own and can differ in the last
digit. The report inside the ZIP is in Turkish.
