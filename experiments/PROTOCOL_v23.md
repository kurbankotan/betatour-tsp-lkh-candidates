# Protocol for run v23: restart markings in candidate files (exploratory; written before the run)

## Why this run

Run v21 showed that the alpha-values of a candidate file act on a restarted LKH-3 search. An entry with alpha = 0
marks an edge that every trial after the first keeps when the edge is in the best or next best tour
(`ChooseInitialTour.c`, case C); elsewhere the tour is rebuilt by a random walk. With restricted search (the default),
the first edge that a move removes is not an edge of the best tour (`LinKernighan.c`), so the marks also decide where
the search of a trial can start. LKH-3 marks the edges of its minimum 1-tree, and the corrected files of run v21 used
the same marks (tree encoding). This run asks whether marks taken from the tour ensemble give shorter tours within
the budget.

Evidence available before this run (development sandbox, 19 November 2026; exploratory, used only to design it):

- **Offline analysis**, three instances of 10,000 nodes (two uniform, one clustered), six first-trial tours each. An
  edge of a first-trial tour was called wrong if it was not in the shorter of two independent 150-s LKH-3 tours of
  the instance; the first-trial tours were 0.24 to 1.13% longer than that tour, and 9.0 to 13.2% of their edges were
  wrong. LKH's own marks left 23.9 to 24.9% of a tour's edges unprotected, and these held 58.6 to 61.5% of the wrong
  edges. Protecting the edges found in at least five of the six tours (count marking) left 16.4 to 18.6% unprotected,
  and these held 78.1 to 83.2% of the wrong edges. A model marking with the same number of protected edges held 72.3
  to 81.1% of them. The two markings protected largely the same edges: 95.3 to 96.8% of the edges protected by the
  count marking were also protected by the model marking.
- **Probe with LKH-3**, four instances (the three of the offline analysis and one more uniform instance) with two
  seeds each, one process per arm, LKH's CPU clock, POPMUSIC-5 lists and fathers of LKH's own file, only the marks
  changed. Difference to LKH's
  own marks after 59 s: model marking (threshold set to the share of unprotected edges of the count marking)
  -0.009%, shorter in 5 of 8 comparisons; count marking with five of six tours +0.030%, shorter in 1 of 6 (one
  comparison at +0.15%); count marking with four of six tours -0.004%, shorter in 4 of 8. Trials completed: 318
  (model marking), 278 (count marking, five of six), 257 (LKH's marks). No difference could be told from noise.

The idea therefore has a mechanism and an offline indication, and no evidence in the solver. This run is exploratory:
it decides whether a confirmation run is prepared.

## Questions

- **Q1.** With the BetaTour mixed lists, does the model marking give shorter tours than the tree marking of run v21?
- **Q2.** Does the count marking?
- **Q3.** With LKH's own POPMUSIC lists, does the model marking give shorter tours than LKH's own marks?

Stated beforehand: the offline analysis gives no reason to expect the model marking to do better than the count
marking, because it found fewer of the wrong edges at the same number of protected edges. The run and its result are
recorded in the repository whatever their direction.

## Design

- **Instances.** 12 new Euclidean instances of 10,000 nodes: 6 uniform (generator seed 7111) and 6 clustered
  (generator seed 7212), drawn with the generator of the earlier runs. Neither seed was used before, and the
  instances of the offline analysis and of the probe are not among them. Budget 120 s.
- **Processes, shared first phase, time accounting.** As in run v21: six LKH-3.0.14 processes per arm (seeds 101 to
  106), ascent-free POPMUSIC, `MAX_CANDIDATES = 5`, `TRACE_LEVEL = 2`; per seed one first trial, after which LKH
  writes its candidate file; a restarted arm is charged the wall time of the shared first phase and the time to
  build its own file (successor arrays, model inference, mixing, marking).
- **Arms.**

  | Arm | Candidate lists | Entries with alpha = 0 |
  |---|---|---|
  | `pop_full` | POPMUSIC-5, uninterrupted | LKH's own (no file) |
  | `pop_native` | the seed's POPMUSIC-5 lists, restart | LKH's own (minimum 1-tree) |
  | `pop_model` | the same lists, restart | model marking |
  | `beta_tree` | BetaTour mixed lists (model top 3, then POPMUSIC), restart | tree marking (the arm of run v21) |
  | `beta_model` | the same BetaTour lists, restart | model marking |
  | `beta_count` | the same BetaTour lists, restart | count marking |

  Every file holds the fathers of the seed's own POPMUSIC file.
- **Markings.**
  *Tree marking:* an entry whose edge has alpha = 0 in the seed's own POPMUSIC lists.
  *Count marking:* an entry whose edge is in at least five of the six first-trial tours (the seed's own tour
  included).
  *Model marking:* the probability of an undirected edge is the larger of the model's two directed outputs on its
  graph. Per seed, the threshold is the probability of the k-th most probable edge of the seed's first-trial tour,
  where k is the number of edges of that tour that the count marking protects; an entry is marked if its probability
  is at least the threshold. The model marking and the count marking thus protect the same number of edges of the
  first tour and differ in which edges they protect.
  The other entries keep their order: on the POPMUSIC lists they keep LKH's alpha-values (0 becomes 1), on the
  BetaTour lists they get 100, 200, ... by list position, as in the tree encoding of run v21.
- **Order.** Instances alternate between the families. The order of the five restarted arms is a random permutation
  per instance (seed 23000 + position). The uninterrupted arm runs first on even and last on odd indices within a
  family. All arms of an instance run on one machine; an instance interrupted by a change of machine is run again
  from the start. A later session on another GPU type is refused unless `ALLOW_HARDWARE_CHANGE` is set; its use is
  a deviation and is reported.
- **Self-check.** The deterministic file check of run v21 on the machine of the run. Per instance the notebook also
  records whether the marking writer, given LKH's own marks, returns LKH's own alpha-values.

## Readings and decision rule

Percentage difference = 100 x (first arm - second arm) / second arm; negative values favour the first arm.

- **M1 (Q1):** `beta_model` versus `beta_tree`. **M2 (Q2):** `beta_count` versus `beta_tree`. **M3 (Q3):**
  `pop_model` versus `pop_native`.
- Endpoint of all three: per instance, the mean over the six seeds of the paired percentage difference of the best
  tour length at the budget (same seed, same first tour, same lists, same fathers).
- Reported for each: wins, ties and losses over the 12 instances, the exact two-sided sign test (ties excluded),
  mean, median and 95% bootstrap interval (5,000 resamples of instances). No correction for multiple comparisons is
  applied, and no result of this run is a confirmatory test.
- **Decision rule.** A marking is taken to a confirmation run (new instances, its own protocol, one primary
  comparison) if it is shorter on more instances and the sign test gives p < 0.05, that is, on at least 10 of the 12
  instances when none ties. Otherwise no confirmation run is prepared, and the README records that the marking was
  not supported at this size. With three readings the chance that at least one of them passes when no marking has
  an effect is at most about 6% (3 x 79 / 4096), which is why a confirmation run is required before any statement
  in the paper.

Secondary, descriptive: both endpoints (best of six, mean of six) for the three readings and for `beta_model` /
`beta_count` (model against counting at an equal number of protected edges), every restarted arm against
`pop_full`, `beta_tree` / `pop_native` and `beta_model` / `pop_model` (effect of the lists under each marking); the
same at 25, 50 and 75% of the budget; the two families separately; trials completed within the budget; the share of
the first-trial tour's edges with a zero alpha-value; the thresholds of the model marking; the number of instances
on which each arm had the shortest tour. `beta_tree` / `pop_full` repeats the second primary comparison of run v21
on new instances; it is reported next to run v21 and not pooled with it.

## What each outcome means

- A marking passes the rule: a confirmation run at 10,000 nodes is prepared and, if it confirms the gain, a run on
  the other sizes. The paper says nothing about markings before that.
- No marking passes: the paper keeps the tree encoding of run v21 and states in one sentence, with the intervals,
  that two ensemble markings were tried without a detectable gain.
- A marking is longer on more instances with p < 0.05: the README records that the marking was harmful at this size.

## Scope and limits

One size, two synthetic families, one budget, one LKH-3 version, 12 instances. The rule needs a marking to be
shorter on at least 10 of the 12 instances: if it is truly shorter on three instances in four, this happens in about
39% of such runs, and with nine instances in ten in about 89%; a real but less regular gain can be missed. The rule
uses the sign test because, in run v21, one instance on which three of the six processes of one arm stayed on a
plateau raised the between-instance standard deviation of that arm's paired mean-of-six differences from between
0.006 and 0.008% to between 0.03 and 0.04%. A marking
changes two things at once: which edges of the best tour a trial keeps, and the order of the entries of each list,
because LKH sorts a list by alpha-value first. The run does not separate the two. The threshold of the model marking
is tied to the count marking, so other numbers of protected edges are not tested. The count marking can protect only
edges that occur in the first-trial tours, whereas the model marking can protect any edge of the model's graph; the
share of the current best tour that each marking protects can therefore develop differently during the search, and
the run records the number of trials, not this share. All 12 instances are analysed; flagged instances (first trial
of the uninterrupted arm differs from the shared first phase, tree rule or marking writer does not reproduce LKH's
values) are kept. Any deviation from this protocol is listed in `v23_restart_marking/README.md`.

## Common rules

As in run v21: LKH-3.0.14 from the official archive or from a pinned copy checked by SHA-256; the checkpoint
`betatour_v2.pt` (SHA-256 2322f66c...3ebd70) and repository commit d63f7e17. The module of run v21 (`alphaenc.py`)
and the marking module of this run (`marking.py`) are embedded in the notebook, compared with their SHA-256 values,
copied to the result archive and recorded in the manifest together with one SHA-256 over all code cells. A changed
setting or code requires a new experiment name.
