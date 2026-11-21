# Protocol for run v27 (pilot): exact merging on the union graph of the tours (written before the run)

## Why this run

Run v26 (`v26_recombination_confirmation/README.md`) stopped six LKH-3 processes at half of the wall-clock budget and
merged their tours. GA-EAX on the 102 tours and on the six best tours (`dual_50`) gave shorter tours than the
processes left running on 27 of 28 instances, but LKH-3's own merging of the six best tours (`lkhmerge_50`) gave
nearly the same gain, and the difference between the two was not significant. A literature scan after that run
proposed a mathematical and a learned component for the merging step:

- **Exact merging.** The union graph of the six tours (every edge that at least one of them uses) is very sparse. Its
  shortest tour can be found exactly (tour merging, Cook and Seymour, 2003). Here SCIP solves it by branch and cut
  after a reduction that fixes and removes edges (`v27_union_exact/unionx.py`).
- **Model-ranked edges.** The shortest tour of the union graph cannot use an edge that no tour has. BetaTour ranks
  the edges of its graph; its most probable edges outside the union could let the exact merge find shorter tours.

## Evidence before the run

A technical check during development (one instance of 3,000 nodes with a budget of 12 s and one clustered instance
of 10,000 nodes with 60 s, six processes run two at a time on a two-core machine, cut at half of each process's
time; not a result of this study) gave:

- 3,000 nodes: the exact optimum over the union of the six best tours was 0.050% shorter than the best of them, and
  GA-EAX on the same six tours found the same tour. Over the union of 30 tours (best tour and four trial tours per
  process) the exact optimum and GA-EAX on 102 tours reached the same length (0.054%). The 102-tour union was not
  solved within 60 s.
- 10,000 nodes: the exact optimum over the six-tour union was 0.0195% shorter than the best tour (solved in 23 s);
  GA-EAX on the same six tours was 0.0230% shorter, with one edge outside the union.
- 3,000 nodes, BetaTour edges: adding the 60 or 150 most probable model edges outside the six-tour union (2% and 5%
  of n), or as many shortest edges, did not change the exact optimum; the optimal tour used none of them.

These observations suggest that GA-EAX already reaches the exact optimum of the union graph, or goes beyond it, and
that the added edges are not used. This pilot checks them on 12 new instances of 10,000 nodes, with the times and
the machine of run v26, before the line is closed or pursued.

## Questions and readings

The readings use the 12 instances of 10,000 nodes. Each is an exact two-sided sign test over instances (ties
excluded) at the budget, on the tour lengths; there is no correction (pilot).

- **R1.** `exact6_50` versus `best6_50`: the exact optimum over the union of the six best tours versus GA-EAX on the
  same six tours.
- **R2.** `exact6_50` versus `dual_50` (the primary arm of run v26).
- **R3.** `exact6beta_50` versus `exact6_50`: does adding the model's edges shorten the exact merge?
- **R4.** `exact6beta_50` versus `exact6near_50`: are the model's edges better than as many shortest edges?

A reading holds if the first arm is shorter on more instances than it is longer and p < 0.05 (with 12 instances
and no ties: at least 10 shorter).

## Decision rule

- If R1 or R2 holds, a confirmation run of exact merging is prepared (new instances, protocol pushed before the
  run). If both fail, exact merging on the union graph is recorded as not shortening the tours of GA-EAX merging at
  these sizes and budgets, and no confirmation run is prepared.
- The model's edges are taken further only if R3 and R4 both hold. Otherwise they are recorded as not supported.
- The two instances of 20,000 nodes, solve times, the share of proven optima, the edges of the GA-EAX tours outside
  the union and the comparisons with `lkhmerge_50` and `lkh6` are descriptive.

## Design

- Instances: 12 new instances of 10,000 nodes (6 uniform, generator seed 11111; 6 clustered, 11212; 120 s) and 2 new
  instances of 20,000 nodes (1 uniform, 11313; 1 clustered, 11414; 240 s), from the generator of the earlier runs.
  None of these seeds was used before.
- One recorded run per instance, as in run v26: six ascent-free POPMUSIC-5 LKH-3.0.14 processes (seeds 201-206)
  for the whole budget, recorded by the copy of LKH-3 with the recording calls of run v26 (same search). All arms
  are derived from it; the cut is at half of the budget.
- Arms: `lkh6` (the shortest tour of the six processes at the budget); `trials_50`, `best6_50` and `dual_50` (GA-EAX
  on 102 tours and on the six best tours, two processes at the same time, and the shorter result; code and settings
  of run v26); `lkhmerge_50` (LKH-3's merging of the six best tours with six processes, seeds 301-306; as run v26);
  `exact6_50`, `exact6beta_50`, `exact6near_50` (below).
- Exact arms: the union graph of the six best tours at the cut; for `exact6beta_50` plus the k = round(0.02 n) edges
  of the BetaTour graph (16 nearest neighbours and the model's ensemble edges) that are absent from the union and
  have the highest predicted probability; for `exact6near_50` plus the k shortest such edges. The model reads the six
  first-trial tours of the recorded run as its ensemble (M = 6, as in runs v4 to v8 and v21 to v23), with the
  checkpoint of runs v2 to v23. The shortest tour of each graph is computed by SCIP 10.0 (PySCIPOpt 6.3.0) with the
  reduction and the subtour cuts of `unionx.py`, single-threaded, with the best of the six tours as its first
  solution and a time limit equal to the budget minus the cut and the arm's own preparation (the union; the model's
  inference for `exact6beta_50`; the model graph for `exact6near_50`). The three exact arms run at the same time in
  separate processes, one core each. A result counts if SCIP found it within the arm's deadline; otherwise the best
  of the six tours counts.
- The order of the three derived blocks (GA-EAX, LKH-3 merging, exact arms) rotates with the instance index.
- Endpoint: the length of the tour at the budget, paired by instance; percentage differences
  100 × (first − second) / second, with 5,000 bootstrap resamples over instances for 95% intervals.
- Machine: Colab L4 GPU with high RAM (12 vCPUs, 53 GB), as run v26. The GPU is used only for the model's inference.

## Scope and limits

- A pilot: 12 instances for the readings, no correction, no confirmatory test; two instances of 20,000 nodes.
- One budget per node, one cut, six tours in the union (the 102-tour union is not solved; it was not solved within
  60 s at 3,000 nodes), one edge share (2% of n) and one way of choosing the model's edges, one LKH-3 version, one
  GA-EAX version with its fixed random seed, one SCIP version, one machine type.
- The exact arms use one core each after the cut, `dual_50` two and `lkhmerge_50` six.
- An exact merge cannot use an edge outside its graph, while GA-EAX can add edges when it merges subtours; the run
  measures the resulting tour lengths, not the reason for a difference.
- Exact tour merging (Cook and Seymour, 2003) and the use of learned edge scores to restrict or extend a TSP graph
  exist; no claim of novelty follows from this run.
- No interim decision: the run ends when all 14 instances are done. Every deviation from this protocol is listed in
  `v27_union_exact/README.md`.

## Common rules

LKH-3.0.14 from the official archive or the pinned copy checked by SHA-256, with the recording patch of run v26;
GA-EAX at the pinned commit of run v26 with its patches; repository commit d63f7e17 for `tsplab.py`, `tsp_model.py`
and the checkpoint (checked by SHA-256); PySCIPOpt 6.3.0 from PyPI. The modules `byprod.py` (run v26, unchanged) and
`unionx.py` are embedded in the notebook, compared with their SHA-256 values, copied to the result archive and
recorded in the manifest together with the SHA-256 of this protocol and one SHA-256 over all code cells, both modules
and this protocol. The notebook does not start unless this protocol and the notebook are on the main branch on GitHub
as built. A changed setting or code requires a new experiment name. The repository is updated with the result after
the run.
