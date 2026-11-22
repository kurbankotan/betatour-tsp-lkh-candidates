# Run v27 (pilot): exact merging on the union graph of the tours

Run v26 (`../v26_recombination_confirmation/README.md`) found that merging the tours of six LKH-3 processes at half of
the wall-clock budget gives shorter tours than letting the processes run, and that LKH-3's own merging captures most
of the gain of GA-EAX. This pilot asks two questions about the merging step. Is the shortest tour of the union graph
of the six tours (every edge used by at least one of them), computed exactly, shorter than the tour GA-EAX finds? And
does adding the BetaTour model's most probable edges outside the union shorten that exact merge? The protocol is
`../PROTOCOL_v27.md`; it states the evidence of a technical check before the run (GA-EAX reached the exact optimum of
the union graph on one instance of 3,000 nodes and went slightly beyond it on one of 10,000 nodes; the model's edges
were not used) and the rule by which the run decides whether a confirmation run is prepared. The notebook refuses to
start unless that file and the notebook itself are on the main branch on GitHub as built.

## Design

- 14 new instances: 12 of 10,000 nodes (6 uniform, 6 clustered; 120 s) for the readings and 2 of 20,000 nodes (1 + 1;
  240 s, descriptive).
- One recorded run per instance, as in run v26: six POPMUSIC-5 LKH-3.0.14 processes (seeds 201-206) for the whole
  budget (`lkh6`). All other arms start from its tours at half of the budget.
- `trials_50`, `best6_50`, `dual_50`: GA-EAX on 102 tours and on the six best tours, two processes at the same time,
  and the shorter result (run v26). `lkhmerge_50`: LKH-3's own merging of the six best tours (run v26).
- `exact6_50`: the shortest tour of the union graph of the six best tours, by branch and cut with SCIP 10.0;
  `exact6beta_50`: the same graph plus the 2% x n most probable BetaTour edges outside the union (the model reads the
  six first-trial tours); `exact6near_50`: plus as many shortest edges of the same model graph (control). One core
  each, the three at the same time, each with the rest of the budget minus its own preparation.
- Readings (12 instances of 10,000 nodes, exact sign test, no correction): R1 `exact6_50` / `best6_50`, R2
  `exact6_50` / `dual_50`, R3 `exact6beta_50` / `exact6_50`, R4 `exact6beta_50` / `exact6near_50`. A confirmation run
  follows only if R1 or R2 holds; the model's edges are taken further only if R3 and R4 hold.

## Files

- `unionx.py`: the union graph, the reduction (a node with two remaining edges uses both; a node with two fixed edges
  uses no other; an edge that would close a fixed path into a short cycle is removed), the branch and cut with SCIP
  (degree equations and subtour cuts from the connected components of LP and integral solutions, on the contracted
  fixed paths), the choice of extra edges and a command line that solves one graph in its own process with a
  wall-clock deadline. NumPy and PySCIPOpt.
- `cells/`: the sources of the notebook cells (configuration, set-up with the check of the protocol on GitHub,
  instances, the arms, self-check, run loop, analysis, archive). The recorded run, GA-EAX and LKH-3 merging use the
  code of run v26 unchanged (`../v26_recombination_confirmation/byprod.py`, embedded as `byprod_v26.py`).
- `build_colab_notebook.py`: writes `notebooks/29_BetaTour_v27_union_exact_pilot.ipynb`, embeds both modules and
  writes the SHA-256 of the protocol, of both modules and of the build into the first cell.
- `tests/test_unionx.py`: tests of `unionx.py` (the reduction and the exact solve against enumeration on small random
  graphs; a union of six 2-opt tours of 400 nodes; the time limit and the command line; the extra edges).
- `tests/run_notebook_locally.py`: runs the notebook's code cells outside Colab in test mode (two processes, three
  small instances, the model on the CPU) and imitates a lost session between or inside instances.

```
python experiments/v27_union_exact/tests/test_unionx.py
BETATOUR_V27_TEST=1 BETATOUR_V27_REPO=. BETATOUR_V27_LKH=/path/to/LKH-3.0.14/LKH \
    BETATOUR_V27_EAX_SRC=/path/to/GA-EAX BETATOUR_V27_WORK=/tmp/v27_test \
    python experiments/v27_union_exact/tests/run_notebook_locally.py
```

## Running it

Commit and push the protocol and the notebook first. Then open the notebook in Colab with an L4 GPU and high RAM
(12 vCPUs, 53 GB) and run all cells. Estimated time: set-up (PySCIPOpt, two builds of LKH-3 and four of GA-EAX, the
model) 6-8 min; self-check 2-3 min; 12 instances of 10,000 nodes at about 5 min each, 1 h; 2 of 20,000 nodes at about
10 min each, 20 min; analysis and archive 1 min; about 1.5 h in total. Completed instances are kept on Drive
(`MyDrive/BetaTour/v27_union14`) and skipped when the notebook is run again; an instance interrupted by a change of
machine runs again from the start. `MAX_NEW_INSTANCES` in the first cell splits the run into sessions.

## Result (run on 21 November 2026)

All 14 instances completed in one Colab session (NVIDIA L4, 12 vCPUs, 53 GB of RAM); they ran from 09:48 to 11:06 UTC
(1.3 h). The protocol and the notebook were committed at 09:45:26 UTC (commit 10643d6), and the notebook found both on
GitHub as built at 09:46:57 UTC, before the first instance. The self-check passed on the machine of the run: besides
the checks of run v26 (the recording copy of LKH-3, GA-EAX with and without the grid search, LKH-3's merging), the
reduction and the exact solve agreed with enumeration on small graphs, SCIP solved the union graph of six tours of an
instance of 2,000 nodes to optimality in 0.02 s with the same result in the notebook and through the command line,
and the model's graph and the graph of the control arm were the same. All 84 processes of `lkh6` were stopped from
outside as designed, the populations held 102 and 6 different tours, all six merging processes recorded a tour within
their time on every instance, and no exact arm returned a tour found after its deadline (the solver processes ended
at most 0.16 s after it). There was no deviation from the protocol.

### Readings of the protocol

12 instances of 10,000 nodes; instances on which the first arm was shorter / equal / longer:

| Reading | Instances | Mean difference (%) | 95% interval | Sign test p | Holds |
|---|---|---|---|---|---|
| R1: `exact6_50` / `best6_50` | 6 / 1 / 5 | +0.0050 | -0.0001 to +0.0116 | 1 | no |
| R2: `exact6_50` / `dual_50` | 1 / 0 / 11 | +0.0111 | +0.0048 to +0.0187 | 0.0063 | no |
| R3: `exact6beta_50` / `exact6_50` | 1 / 5 / 6 | +0.0066 | +0.0012 to +0.0119 | 0.125 | no |
| R4: `exact6beta_50` / `exact6near_50` | 0 / 5 / 7 | +0.0071 | +0.0028 to +0.0117 | 0.016 | no |

None of the four readings held. In R1 the exact arm was shorter on six instances and longer on five; in R2 and R4
the first arm was longer on most instances (11 and 7 of the 12) and shorter on one and on none. By the decision rule,
exact merging on the union graph is recorded as not shortening the tours of GA-EAX merging at these sizes and
budgets, no confirmation run is prepared, and the model's edges are recorded as not supported.

### The exact arms

Within the time left after the cut (about 60 s at 10,000 nodes, one core each), SCIP proved the shortest tour of the
union graph of the six best tours (`exact6_50`) on 5 of the 12 instances of 10,000 nodes, in 2.6 to 44.8 s, and on
neither instance of 20,000 nodes. It proved the larger graphs on 3 (`exact6beta_50`) and 7 (`exact6near_50`) of the
12. After the reduction a core of about 3,100 nodes remained at 10,000 nodes (6,700 at 20,000). Where SCIP did not
finish `exact6_50`, the gap between its tour and its lower bound was 0.18% at the median and at most 5.7%, and on 3
of the 14 instances it did not improve the tour it started from. On all five instances on which `exact6_50` was longer
than `best6_50`, SCIP had not proven the optimum; on the five on which it had, `exact6_50` was shorter than
`best6_50` on four and equal on one. The 200 model edges per instance (each with a predicted probability above 0.90)
appeared in the tour of `exact6beta_50` on 2 of the 12 instances (5 and 6 edges), and on both that tour was longer
than the tour of `exact6_50`. The 200 shortest edges of the control arm were never used.

### Where the shortest tour of the union graph is known

On 7 of the 12 instances of 10,000 nodes a record proves the shortest tour of the six-tour union graph: SCIP proved
`exact6_50` optimal on five, and on two it proved `exact6near_50` optimal with a tour that uses no edge outside the
union. Difference from that tour (%):

| Instance | Proven by | Length | `best6_50` | `dual_50` | `lkhmerge_50` | `exact6_50` |
|---|---|---|---|---|---|---|
| `u27_uni10k_000` | `exact6_50` | 71,739,558 | +0.00025 | -0.00077 | +0.00003 | equal |
| `u27_clu10k_000` | `exact6near_50` | 27,584,947 | +0.00017 | -0.01146 | -0.00003 | +0.00177 |
| `u27_clu10k_001` | `exact6near_50` | 41,293,190 | +0.00252 | -0.00128 | -0.00008 | +0.01268 |
| `u27_uni10k_003` | `exact6_50` | 72,096,726 | +0.00143 | -0.00949 | -0.00574 | equal |
| `u27_uni10k_004` | `exact6_50` | 71,710,836 | +0.00283 | -0.00131 | equal | equal |
| `u27_clu10k_004` | `exact6_50` | 39,040,935 | +0.00054 | -0.01070 | +0.01103 | equal |
| `u27_clu10k_005` | `exact6_50` | 30,333,000 | equal | -0.00013 | equal | equal |

GA-EAX on the same six tours (`best6_50`) reached this length on one instance and was at most 0.0028% longer on the
others (mean +0.0011%). LKH-3's merging was shorter on three instances, equal on two and longer on two. `dual_50` was
shorter on all seven (mean -0.0050%), so its tours used edges outside the union graph. On all 12 instances of 10,000
nodes `dual_50` took its tour from GA-EAX on the 102 tours (`trials_50`), whose tours had 18.7 edges outside the
six-tour union on average, against 1.5 for `best6_50` and 0.6 for `lkhmerge_50`.

### Other comparisons (descriptive)

Instances on which the first arm was shorter / equal / longer, and the mean difference (%):

| Comparison | 10,000 nodes (12) | 20,000 nodes (2) |
|---|---|---|
| `exact6_50` / `dual_50` | 1 / 0 / 11, +0.0111 | 0 / 0 / 2, +0.0366 |
| `exact6_50` / `lkhmerge_50` | 3 / 2 / 7, +0.0072 | 0 / 0 / 2, +0.0284 |
| `exact6_50` / `lkh6` | 9 / 0 / 3, -0.0059 | 0 / 0 / 2, +0.0175 |
| `exact6near_50` / `exact6_50` | 3 / 7 / 2, -0.0006 | 1 / 1 / 0, -0.0040 |
| `dual_50` / `lkhmerge_50` | 11 / 0 / 1, -0.0040 | 2 / 0 / 0, -0.0083 |
| `best6_50` / `lkhmerge_50` | 2 / 1 / 9, +0.0021 | 1 / 0 / 1, -0.0049 |
| `dual_50` / `lkh6` | 12 / 0 / 0, -0.0171 | 2 / 0 / 0, -0.0192 |

At 20,000 nodes the three exact arms were longer than every merging arm on both instances, and `exact6_50` was also
longer than `lkh6` on both. On all 14 instances `dual_50` was shorter than `lkh6` (mean -0.0174%) and on 13 of them
shorter than LKH-3's merging (mean -0.0046%), as at 10,000 nodes in run v26 (13 of 16); run v27 did not test these
comparisons. GA-EAX converged on the six tours on all 14 instances, and on the 102 tours on 11 of the 12 instances of
10,000 nodes and on neither of 20,000 nodes.

### What the run shows

- Not supported (R1, R2): with the time left after the cut and one core, exact merging on the union graph of the six
  best tours gave tours that were not shorter than those of GA-EAX on the same six tours and were longer than those
  of the GA-EAX arm of run v26 on 11 of 12 instances. Where the optimum of the union graph is known (7 instances),
  GA-EAX on the same six tours came within 0.003% of it, and the GA-EAX arm, which drew on 102 tours, was shorter
  than it on all seven. No confirmation run of exact merging is prepared.
- Not supported (R3, R4): adding the 200 edges that BetaTour ranked highest outside the union did not shorten the
  exact merge. The larger graph was proven optimal less often, and its tours were longer than those of the union
  alone on 6 and than those with as many shortest edges on 7 of the 12 instances.
- Limits: one budget per node, one cut, six tours in the union, one edge share and one way of choosing the model's
  edges, a branch and cut with degree equations and subtour cuts only, one SCIP version on one core per arm, one
  machine type; no statement extends beyond them. Exact tour merging (Cook and Seymour, 2003) and learned edge scores
  that restrict or extend a TSP graph exist; no claim of novelty follows from this run.

Run v28 (a pilot) followed with merging the 102 tours in two stages; its result is in `../v28_staged_merging/README.md`.

`python analysis/v27_union_exact.py results/v27/v27_union14_results.zip` recomputes these numbers from the records,
and with `--tours src` also the lengths of the archived tours, the union graphs, the edges outside them and the
consistency of the proven optima. The intervals above are those of the notebook (`summary/paired_stats.csv`, 5,000
resamples); the script draws 20,000 resamples of its own and can differ in the last digit. The report inside the ZIP
is in Turkish.
