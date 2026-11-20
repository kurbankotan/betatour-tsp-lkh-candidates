# Protocol for run v24: twice the budget at 20,000 nodes (written before the run)

## Why this run

With candidate files encoded as LKH-3 encodes its own, LKH-3 restarted with the BetaTour mixed lists found shorter
tours than uninterrupted POPMUSIC at 10,000 nodes (run v21: 18 of 24 instances on best of six; run v23: 10 of 12 new
instances; mean difference -0.006% in both). Run v22 repeated the comparison on the other classes with their budgets
of the paper. On its 16 instances of 20,000 nodes (budget 300 s) the two arms did not differ: shorter on 8 and longer
on 8, on best of six (mean +0.0001%) and on the mean of six (mean +0.0042%).

The time course of run v22 at 20,000 nodes is the reason for this run. It is exploratory (8 uniform and 8 clustered
instances, read after the run). Mean differences at 25, 50, 75 and 100% of the budget:

| Comparison (20,000 nodes) | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
| `beta_tree` / `pop_full`, best of six, uniform | +0.0160 | +0.0051 | +0.0030 | -0.0009 |
| `beta_tree` / `pop_full`, best of six, clustered | +0.0044 | +0.0046 | +0.0028 | +0.0011 |
| `pop_native` / `pop_full` (restart), mean of six, uniform | +0.0178 | +0.0110 | +0.0074 | +0.0037 |
| `pop_native` / `pop_full` (restart), mean of six, clustered | +0.0230 | -0.0020 | +0.0055 | +0.0038 |
| `beta_tree` / `pop_native` (lists), mean of six, uniform | -0.0072 | -0.0056 | -0.0041 | -0.0031 |
| `beta_tree` / `pop_native` (lists), mean of six, clustered | -0.0363 | -0.0030 | +0.0013 | +0.0040 |

On the uniform instances the restarted arm closed its distance to the uninterrupted arm throughout the budget, and
the cost of the restart itself (a file equivalent to LKH's own against no restart) was still falling at the end of
the budget; it was positive on all eight instances up to three quarters of the budget. On the 18 uniform instances of
10,000 nodes in runs v21 and v23 the same cost was +0.0084% at a quarter of the budget and +0.0011% at half. The
restarted search began 3.1 to 3.4% of the budget after the first trial of the uninterrupted process had ended at
20,000 nodes and 2.5 to 3.2% after it at 10,000 nodes, so the larger cost at 20,000 nodes is not explained by a longer
delay. This run asks whether the comparison at 20,000 nodes turns in favour of the BetaTour lists when the search
runs twice as long.

Stated beforehand: the evidence is a trend in an exploratory time course. In the same data the effect of the lists
was itself shrinking with time in both families, and on the clustered instances it had vanished by the end of the
budget. The outcome is open.

## Question

- **H5.** At 20,000 nodes and a budget of 600 s, LKH-3 restarted with the BetaTour mixed lists in corrected files
  finds shorter tours than uninterrupted POPMUSIC, all costs charged.

The result is reported in the paper whatever its direction.

## Design

- **Instances.** 20 new Euclidean instances of 20,000 nodes: 10 uniform (generator seed 8505) and 10 clustered
  (generator seed 8606), drawn with the generator of the earlier runs. Neither seed was used before. Budget 600 s,
  twice the budget of the 20,000-node class in the paper and in run v22.
- **Arms**, as in runs v21 and v22 (the module `alphaenc.py` of run v21 is embedded unchanged): `pop_full`
  (uninterrupted POPMUSIC-5), `pop_native` (restart with the seed's own lists, alpha-values and fathers) and
  `beta_tree` (restart with the BetaTour mixed lists, tree encoding, the seed's fathers).
- **Processes, shared first phase, time accounting.** As in run v21: six LKH-3.0.14 processes per arm (seeds 101 to
  106), ascent-free POPMUSIC, `MAX_CANDIDATES = 5`, `TRACE_LEVEL = 2`; a restarted arm is charged the wall time of
  the shared first phase and the time to build its lists.
- **Outside deadline.** Budget + 60 s (10% of the budget), instead of budget + 3 s. In run v22, LKH-3's clock (the CPU
  time of the process) ran about 1.5% behind the wall clock at 20,000 nodes, and the deadline stopped every process
  of that class before its own clock had reached the budget. Here the deadline is a safeguard only: the
  uninterrupted processes end at LKH's own limit, which does not count its preprocessing and therefore lies after
  the budget on the common clock, and the restarted processes end at theirs. The report lists every instance on
  which the last recorded trial of a process ended before the budget.
- **Order.** Instances alternate between the families. The order of the two restarted arms is a random permutation
  per instance (seed 24000 + position). The uninterrupted arm runs first on even and last on odd indices within a
  family. All arms of an instance run on one machine; an instance interrupted by a change of machine is run again
  from the start. A later session on another GPU type is refused unless `ALLOW_HARDWARE_CHANGE` is set; its use is
  a deviation and is reported.
- **Self-check.** The deterministic file check of run v21 on the machine of the run.

## Endpoints and tests

Percentage difference = 100 x (first arm - second arm) / second arm; negative values favour the first arm.

- **P1 (H5): `beta_tree` versus `pop_full`.** Per instance, the mean over the six seeds of the paired percentage
  difference of the best tour length at the budget (600 s).

Exact two-sided sign test over the 20 instances (ties excluded), alpha = 0.05; there is one primary comparison and no
correction. Mean, median and 95% bootstrap interval (5,000 resamples of instances) are reported with it. H5 is
supported if `beta_tree` is shorter on more instances and p < 0.05, that is, on at least 15 of the 20 instances when
none ties.

Secondary, descriptive (labelled exploratory): best of six for P1; P1 at 25, 50 and 75% of the budget, where 50%
(300 s) repeats the 20,000-node class of run v22 on new instances; `pop_native` / `pop_full` (cost of the restart)
and `beta_tree` / `pop_native` (effect of the lists) on both endpoints at all four times; the two families
separately; trials completed within the budget; processes stopped by the outside deadline. Runs v22 and v24 are
reported side by side and are not pooled.

## What each outcome means

- H5 supported: the paper states that at 20,000 nodes the restart with the BetaTour lists gave shorter tours at
  twice the budget and not at the budget of 300 s, and reports the time course of the cost of the restart and of
  the effect of the lists.
- H5 not supported with more instances shorter than longer: the result at 20,000 nodes is called inconclusive up to
  600 s, with the intervals; the gain is stated for 10,000 nodes.
- H5 not supported otherwise: the paper states that no gain was found at 20,000 nodes at either budget.

## Scope and limits

One size, two synthetic families, one LKH-3 version, six processes, 20 instances. The budget is twice that of the
paper for this size, so the result at 600 s is not a result at the paper's budget; the reading at 300 s is secondary.
The rule needs `beta_tree` to be shorter on at least 15 of the 20 instances: if it is truly shorter on 85% of
instances (on the mean of six it was shorter on 31 of the 36 instances of 10,000 nodes in runs v21 and v23), this
happens in about 93% of such runs, and with 75% in about 62%. No interim decision is taken: the run ends at 20
instances and the partial report printed during the run is exploratory. All 20 instances are analysed. An instance
whose uninterrupted arm does not reproduce the first-trial cost of the shared first phase is kept and flagged, and so
are an instance on which the tree rule does not reproduce LKH's zero alpha-values and an instance with a process
stopped before its clock reached the budget. Any deviation from this protocol is listed in
`v24_longer_budget/README.md`.

## Common rules

As in run v21: LKH-3.0.14 from the official archive or from a pinned copy checked by SHA-256; the checkpoint
`betatour_v2.pt` (SHA-256 2322f66c...3ebd70) and repository commit d63f7e17; the experiment module embedded in the
notebook, compared with its SHA-256, copied to the result archive and recorded in the manifest together with one
SHA-256 over all code cells. A changed setting or code requires a new experiment name. This protocol is committed and
pushed to the repository before the notebook is started.
