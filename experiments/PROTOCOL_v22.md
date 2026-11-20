# Protocol for run v22: corrected candidate files on the other sizes and on TSPLIB (written before the run)

## Why this run

Run v21 (24 instances of 10,000 nodes) found that the candidate files of runs v1 to v8 handicapped restarted LKH-3 and
that, with files encoded as LKH-3 encodes its own, LKH-3 restarted with the BetaTour mixed lists found shorter tours
than uninterrupted POPMUSIC (18 of 24 instances on best of six, adjusted p = 0.023; 22 of 24 on the mean of six). Lists
from tour counting did as well as the BetaTour lists (12 against 12), and a restart with an equivalent file cost
nothing measurable at the budget. The protocol of run v21 fixed that the comparison is repeated on the other sizes and
on TSPLIB before the paper states a gain. This run does that with the same code and rules.

## Questions

- **H3 (other sizes).** On the instance classes of the paper other than 10,000 nodes, LKH-3 restarted with the
  BetaTour mixed lists in corrected files finds shorter tours within the budget than uninterrupted POPMUSIC, all
  costs charged.
- **H4 (model against counting).** The BetaTour mixed lists give shorter tours than the counting mixed lists.

Stated beforehand: in run v5 the best of six of all five arms was the same length on all eight uniform and on five
of the eight clustered 1,000-node instances, and the optimal length on the three TSPLIB instances below 5,000 nodes.
On best of six, ties are therefore expected on the small instances. The mean of six still differs there, because it
reflects how many of the six processes reach that length within the budget.

The results are reported in the paper whatever their direction.

## Design

- **Instances (43).** 16 new instances of 1,000 nodes (8 uniform, generator seed 6101; 8 clustered, seed 6202; budget
  5 s), 16 new instances of 20,000 nodes (8 uniform, seed 6505; 8 clustered, seed 6606; budget 300 s), and the 11
  TSPLIB instances of the earlier runs (pcb3038 to d18512; budget max(10, 0.02 n) s). The budgets are those of the
  earlier runs. None of the four seeds was used before. The 10,000-node class is run v21.
- **Arms**, as in run v21: `pop_full` (uninterrupted POPMUSIC-5), `pop_native` (restart with the seed's own lists,
  alpha-values and fathers), `beta_tree` (restart with the BetaTour mixed lists, tree encoding, the seed's fathers)
  and `freq_tree` (the same with the counting mixed lists). The module `alphaenc.py` of run v21 is embedded
  unchanged.
- **Processes, shared first phase, time accounting.** As in run v21: six LKH-3.0.14 processes per arm (seeds 101 to
  106), ascent-free POPMUSIC, `MAX_CANDIDATES = 5`, `TRACE_LEVEL = 2`; a restarted arm is charged the wall time of
  the shared first phase and the time to build its lists.
- **Order.** The 1,000-node instances run first, then the 20,000-node and TSPLIB instances in turn. The order of the
  three restarted arms is a random permutation per instance (seed 22000 + position); the uninterrupted arm runs
  first on even and last on odd indices within a group. All arms of an instance run on one machine; an instance
  interrupted by a change of machine is run again from the start. A later session on another GPU type is refused
  unless `ALLOW_HARDWARE_CHANGE` is set; its use is a deviation and is reported.
- **Checks before the first instance (not measured).** The deterministic file check of run v21; one first trial per
  TSPLIB instance (file, distance type, LKH's reported length against the recomputed length, shape of the candidate
  file); one model inference on the largest instance.

## Endpoints and tests

Percentage difference = 100 x (first arm - second arm) / second arm; negative values favour the first arm.

- **P1 (H3): `beta_tree` versus `pop_full`.** Per instance, the mean over the six seeds of the paired percentage
  difference of the best tour length at the budget.
- **P2 (H4): `beta_tree` versus `freq_tree`.** The same endpoint.

Both: exact two-sided sign test over the 43 instances (ties excluded), Holm correction over the two tests,
alpha = 0.05; mean, median and 95% bootstrap interval (5,000 resamples of instances) are reported with them. H3 is
supported if `beta_tree` is shorter on more instances and the adjusted p < 0.05; H4 likewise.

The endpoint is the mean of six because best of six is expected to tie on about half of the instances (the
1,000-node instances and the small TSPLIB instances), which would leave about 22 instances for the sign test, and
because it uses all six paired processes of an instance. In run v21 the mean of six favoured `beta_tree` on 22 of the
24 instances and best of six on 18. With 22 instances and three wins in four, a sign test reaches an adjusted
p < 0.05 in about half of such runs.

Secondary, descriptive (no correction, labelled exploratory): best of six for P1 and P2 (the endpoint of P2 in run
v21); both endpoints for `pop_native` / `pop_full`, `beta_tree` / `pop_native`, `freq_tree` / `pop_full` and
`freq_tree` / `pop_native`; every group separately; the instances with at least 7,000 nodes and the smaller ones
separately; the same at 25, 50 and 75% of the budget; trials completed within the budget; the share of the
first-trial tour's edges with a zero alpha-value; for TSPLIB, the gap of every arm to the optimal length. Runs v21
and v22 are reported side by side and are not pooled into one test.

## What each outcome means

- H3 supported: the paper states that, with corrected files, the restart with BetaTour lists gave shorter tours than
  uninterrupted POPMUSIC on the classes of this run as well, with the sizes of the differences per class and the
  ties on the small instances.
- H3 not supported with the same direction: the gain is stated for 10,000 nodes only, with the intervals of this
  run, and the result is called inconclusive for the other classes. H3 not supported with the opposite direction:
  the paper states that the gain of run v21 did not carry over.
- H4 supported: the learned model adds to counting in the solver. H4 not supported: the paper states that the
  solver gain came from the tour ensemble and did not need the learned model, as at 10,000 nodes.

## Scope and limits

Two synthetic families at two sizes and eleven TSPLIB instances, one budget per size, one LKH-3 version, six
processes. No interim decision is taken: the run ends at 43 instances and the partial report printed during the run
is exploratory. All 43 instances are analysed. An instance whose uninterrupted arm does not reproduce the first-trial
cost of the shared first phase is kept and flagged, and so is an instance on which the tree rule, applied to LKH's
own lists, does not reproduce LKH's zero alpha-values. Any deviation from this protocol is listed in
`v22_other_sizes/README.md`.

## Common rules

As in run v21: LKH-3.0.14 from the official archive or from a pinned copy checked by SHA-256; the checkpoint
`betatour_v2.pt` (SHA-256 2322f66c...3ebd70) and repository commit d63f7e17; the experiment module embedded in the
notebook, compared with its SHA-256, copied to the result archive and recorded in the manifest together with one
SHA-256 over all code cells. A changed setting or code requires a new experiment name.
