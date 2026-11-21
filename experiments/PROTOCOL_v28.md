# Protocol for run v28 (pilot): staged merging of the tours at the cut (written before the run)

## Why this run

Run v26 (`v26_recombination_confirmation/README.md`) stopped six LKH-3 processes at half of the wall-clock budget and
merged their tours with GA-EAX. At 10,000 nodes the 102 tours (the best tour and the last 16 trial tours of each
process) gave shorter tours than the six best tours alone on 15 of 16 instances. At 20,000 and 50,000 nodes GA-EAX did
not converge on the 102 tours within the time left (0 of 6 in each class), the 102 tours gave longer tours than the
six best tours on all six instances of 50,000 nodes, and at 20,000 nodes LKH-3's own merging was shorter than the
GA-EAX arm on 5 of 6 instances. Run v27 (`v27_union_exact/README.md`) found that neither the exact shortest tour of the
union graph of the six best tours nor edges ranked by BetaTour shortened the tours of GA-EAX merging. This run asks
whether merging the 102 tours in two stages (in six groups at the same time, then the group results with the six best
tours) recovers the advantage of the 102 tours at a size at which one GA-EAX process does not converge on them in
time.

## Evidence before the run

Two technical checks during development (not results of this study):

- Exact window repair, considered after run v27. On the clustered instance of 10,000 nodes of the check of run v27,
  solving every 200-node window around the nodes at which the six best tours disagree, with the rest of the tour
  fixed, by the branch and cut of run v27 (SCIP) shortened the best tour at the cut by 0.0069%, against 0.023% for
  LKH-3's merging and 0.025% for GA-EAX on the 102 tours on the same instance; windows of 1,000 and 2,500 nodes were
  not solved within 15 and 30 s. This line was not pursued.
- Staged merging. One uniform instance of 20,000 nodes (generator seed 777); six recording LKH-3 processes with a
  budget of 240 s each, run two at a time on a two-core machine; cut at 120 s of each process. In the 120 s after the
  cut, measured from the best tour at the cut: LKH-3's merging (two processes) −0.0505%; GA-EAX on the six best tours
  −0.0469% (converged after 8 s); GA-EAX on the 102 tours −0.0665% (not converged); staged merging with six groups of
  16 trial tours each and then the six group results with the six best tours −0.0692%. The groups ran one at a time;
  on six cores the staged arm would have taken the longest group (38 s) plus the second stage (11 s). The six
  processes left running reached −0.0328% at 240 s.

These observations suggest that staged merging reaches at least what GA-EAX reaches on the 102 tours, in less than
half of the time, when six cores are available. This pilot checks it on 12 new instances of 20,000 nodes with the
times and the machine of runs v26 and v27, on which GA-EAX on the 102 tours did not converge.

## Questions and readings

The readings use the 12 instances of 20,000 nodes. Each is an exact two-sided sign test over instances (ties
excluded) at the budget, on the tour lengths; there is no correction (pilot).

- **R1.** `staged_50` versus `dual_50` (the primary arm of run v26).
- **R2.** `staged_50` versus `lkhmerge_50` (LKH-3's own merging of the six best tours).

A reading holds if the first arm is shorter on more instances than it is longer and p < 0.05 (with 12 instances
and no ties: at least 10 shorter).

## Decision rule

- If R1 and R2 both hold, a confirmation run of staged merging is prepared (new instances, including 50,000 nodes,
  protocol pushed before the run). Only after a confirmation would a learned choice of the groups (BetaTour) be
  considered.
- Otherwise staged merging is recorded as not supported at these sizes and budgets, and no confirmation run is
  prepared.
- The two instances of 50,000 nodes, the time the staged arm used, the convergence of GA-EAX in each stage and the
  comparisons with `lkh6`, `trials_50` and `best6_50` are descriptive.

## Design

- Instances: 12 new instances of 20,000 nodes (6 uniform, generator seed 12111; 6 clustered, 12212; 240 s) and 2 new
  instances of 50,000 nodes (1 uniform, 12515; 1 clustered, 12616; 600 s), from the generator of the earlier runs
  (12 ms per node). None of these seeds was used before.
- One recorded run per instance, as in runs v26 and v27: six ascent-free POPMUSIC-5 LKH-3.0.14 processes (seeds
  201-206) for the whole budget, recorded by the copy of LKH-3 with the recording calls of run v26 (same search). All
  arms are derived from it; the cut is at half of the budget.
- Arms: `lkh6` (the shortest tour of the six processes at the budget); `trials_50`, `best6_50` and `dual_50` (GA-EAX
  on the 102 tours and on the six best tours, two processes at the same time, and the shorter result; code and
  settings of run v26); `lkhmerge_50` (LKH-3's merging of the six best tours with six processes, seeds 301-306; as
  run v26); `staged_50` (below).
- `staged_50` uses the 102 tours of `trials_50`. Stage 1: the tours other than the best tours of the six processes
  are dealt in turn to six groups (16 tours each, from every process), and six GA-EAX processes run at the same time,
  one per group, until 60% of the time left after the cut and the arm's own preparation. Stage 2: the six group
  results and the six best tours (12 tours) go to one GA-EAX process until the end of the budget. A tour counts if
  its process found it within the deadline of its stage; the result of the arm is the shortest of the best tour at
  the cut and these tours. GA-EAX version, number of children (30), patches and fixed random seed are those of run
  v26 ("Normal" up to 20,000 nodes, "Large" above).
- The order of the three derived blocks (GA-EAX, LKH-3 merging, staged merging) rotates with the instance index.
- Endpoint: the length of the tour at the budget, paired by instance; percentage differences
  100 × (first − second) / second, with 5,000 bootstrap resamples over instances for 95% intervals.
- Machine: Colab L4 GPU with high RAM (12 vCPUs, 53 GB), as runs v26 and v27. The GPU is not used.

## Scope and limits

- A pilot: 12 instances for the readings, no correction, no confirmatory test; two instances of 50,000 nodes.
- One budget per node, one cut, one number of groups, one division of the time between the stages (60% and 40%), one
  way of forming the groups, one LKH-3 version, one GA-EAX version with its fixed random seed, one machine type.
- After the cut `dual_50` uses two cores, `lkhmerge_50` six, and `staged_50` six in stage 1 and one in stage 2. The
  staged arm stops early when stage 2 converges; the time it used is recorded, but the comparison is at the budget.
- Island models and parallel genetic algorithms, including parallel GA-EAX on very large instances, exist; no claim
  of novelty follows from this run.
- No interim decision: the run ends when all 14 instances are done. Every deviation from this protocol is listed in
  `v28_staged_merging/README.md`.

## Common rules

LKH-3.0.14 from the official archive or the pinned copy checked by SHA-256, with the recording patch of run v26;
GA-EAX at the pinned commit of run v26 with its patches; repository commit d63f7e17 for `tsplab.py` (checked out
before the run). The module `byprod.py` (run v26, unchanged) is embedded in the notebook, compared with its SHA-256,
copied to the result archive and recorded in the manifest together with the SHA-256 of this protocol and one SHA-256
over all code cells, the module and this protocol. The notebook does not start unless this protocol and the notebook
are on the main branch on GitHub as built. A changed setting or code requires a new experiment name. The repository is
updated with the result after the run.
