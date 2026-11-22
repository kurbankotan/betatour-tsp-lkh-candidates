# Protocol for run v29 (confirmation run): staged merging of the tours at the cut (written before the run)

## Why this run

Run v28 (`v28_staged_merging/README.md`), a pilot, merged the 102 tours of six LKH-3 processes at half of the
wall-clock budget in two stages: six GA-EAX processes at the same time on six groups of trial tours, then one GA-EAX
process on the six group results and the six best tours. On 12 new instances of 20,000 nodes this arm gave shorter
tours at the budget than the GA-EAX arm of run v26 on 10 instances (mean 0.0097%, p = 0.039) and than LKH-3's own
merging on all 12 (mean 0.0131%, p = 0.0005), without correction; on two instances of 50,000 nodes it was the
shortest arm. By the rule of that protocol a confirmation run is prepared: new instances, the settings of run v28
unchanged, and hypotheses fixed before the run.

## Hypotheses

The tests use the 18 new synthetic instances (20,000 and 50,000 nodes). Each is an exact two-sided sign test over
instances (ties excluded) on the tour lengths at the budget, with Holm correction over the two tests.

- **H1.** `staged_50` gives shorter tours than `lkhmerge_50` (LKH-3's own merging of the six best tours).
- **H2.** `staged_50` gives shorter tours than `dual_50` (the primary arm of run v26).

A hypothesis is supported if the first arm is shorter on more instances than it is longer and its Holm-adjusted p is
below 0.05 (with 18 instances and no ties: at least 15 shorter for the smaller p, at least 14 for the larger).

## Decision rule

- If H1 and H2 are both supported, staged merging is recorded as confirmed at these sizes and budgets, and the
  manuscript reports runs v28 and v29. Only then would a learned choice of the groups (BetaTour) be considered, in a
  new pilot with its own protocol.
- If only H1 is supported, staged merging is recorded as shorter than LKH-3's merging but not shown to be shorter than
  the GA-EAX arm; if only H2 or neither, as not confirmed.
- Everything else is descriptive: each size and family, the five TSPLIB instances (distance to the optimum), the time
  the staged arm used, the convergence of GA-EAX in each stage, and the comparisons with `lkh6`, `trials_50` and
  `best6_50`.

## Design

- Instances: 12 new instances of 20,000 nodes (6 uniform, generator seed 13111; 6 clustered, 13212; 240 s) and 6 new
  instances of 50,000 nodes (3 uniform, 13515; 3 clustered, 13616; 600 s), from the generator of the earlier runs
  (12 ms per node); none of these seeds was used before. Five TSPLIB instances with known optimum (rl11849, usa13509,
  brd14051, d15112, d18512; used in runs v25 and v26) with a budget of 0.02 s per node, as in run v26; they do not
  enter the tests.
- One recorded run per instance, as in runs v26 to v28: six ascent-free POPMUSIC-5 LKH-3.0.14 processes (seeds
  201-206) for the whole budget, recorded by the copy of LKH-3 with the recording calls of run v26 (same search). All
  arms are derived from it; the cut is at half of the budget.
- Arms, all as in run v28: `lkh6` (the shortest tour of the six processes at the budget); `trials_50`, `best6_50` and
  `dual_50` (GA-EAX on the 102 tours and on the six best tours, two processes at the same time, and the shorter
  result); `lkhmerge_50` (LKH-3's merging of the six best tours with six processes, seeds 301-306); `staged_50`
  (stage 1: the tours other than the six best tours dealt in turn to six groups, six GA-EAX processes at the same
  time until 60% of the time left after the cut and the arm's preparation; stage 2: the six group results and the six
  best tours in one GA-EAX process until the budget; a tour counts if found within the deadline of its stage).
  GA-EAX version, number of children (30), patches and fixed random seed as in run v26 ("Normal" up to 20,000 nodes,
  "Large" above).
- The order of the three derived blocks (GA-EAX, LKH-3 merging, staged merging) rotates with the instance index.
- Endpoint: the length of the tour at the budget, paired by instance; percentage differences
  100 × (first − second) / second, with 5,000 bootstrap resamples over instances for 95% intervals.
- Machine: Colab L4 GPU with high RAM (12 vCPUs, 53 GB), as runs v26 to v28. The GPU is not used.

## Scope and limits

- One budget per node, one cut, one number of groups, one division of the time between the stages (60% and 40%), one
  way of forming the groups, one LKH-3 version, one GA-EAX version with its fixed random seed, one machine type.
- After the cut `dual_50` uses two cores, `lkhmerge_50` six, and `staged_50` six in stage 1 and one in stage 2; the
  staged arm stops early when stage 2 converges, and the comparison is at the budget.
- The TSPLIB instances were used before (runs v25 and v26) and are descriptive only.
- Island models and parallel genetic algorithms, including parallel GA-EAX on very large instances, exist; no claim
  of novelty follows from this run.
- No interim decision: the run ends when all 23 instances are done. Every deviation from this protocol is listed in
  `v29_staged_confirmation/README.md`.

## Common rules

LKH-3.0.14 from the official archive or the pinned copy checked by SHA-256, with the recording patch of run v26;
GA-EAX at the pinned commit of run v26 with its patches; repository commit d63f7e17 for `tsplab.py` (instances and
TSPLIB helpers); the TSPLIB files from the TSPLIB website. The module `byprod.py` (run v26, unchanged) is embedded in
the notebook, compared with its SHA-256, copied to the result archive and recorded in the manifest together with the
SHA-256 of this protocol and one SHA-256 over all code cells, the module and this protocol. The notebook does not
start unless this protocol and the notebook are on the main branch on GitHub as built. A changed setting or code
requires a new experiment name. The repository is updated with the result after the run.
