# Protocol for run v21: candidate-file encoding control (written before the run)

## Why this run

In runs v1 to v8, every arm that gave LKH-3 its candidate lists from outside (the restarted POPMUSIC controls, the
counting arms and the BetaTour arms) passed them through a candidate file written by `tsplab.write_candidates`. The
source code of LKH-3.0.14 shows that two fields of such a file act on the search after the file has been read, and
the files of runs v1 to v8 did not reproduce either of them.

1. **Alpha-values.** Every trial after the first starts from a tour built by a random walk
   (`ChooseInitialTour.c`). The walk follows an edge of the best or next best tour by rule only if the edge is a
   candidate whose alpha-value is zero (case C of the function); otherwise it moves to a random unvisited candidate
   (case D). With `CANDIDATE_SET_TYPE = POPMUSIC` and `SUBGRADIENT = NO`, LKH-3 gives alpha = 0 to the edges of its
   minimum 1-tree, about two candidates per node. The files of runs v1 to v8 held alpha = 0, 100, 200, ... in list
   order, so one candidate per node had alpha = 0.
2. **Father of the node in the minimum 1-tree (Dad).** Until the first tour of a run is recorded, `Near(a, b)`
   (`LKH.h`) is true for the edges of the minimum 1-tree, and the first trial does not break these edges first
   (`LinKernighan.c`, restricted search). The files of runs v1 to v8 held no father.

Evidence available before this run (exploratory, used only to design it):

- Deterministic check with a fixed number of trials (`alphaenc.encoding_selfcheck`; six instances of 500 to 2,000
  nodes, 10 to 25 trials). On all six, a file rewritten with LKH's own alpha-values and fathers gave the same cost
  at every trial as the file LKH wrote itself, and with alpha = 0, 100, 200, ... the search differed from the second
  trial on. Without the fathers the search differed from the first trial on in one instance and not at all in the
  other five.
- Probe in the development sandbox (`v21_alpha_encoding/probe/`): 18 uniform instances of 10,000 nodes, one process
  per arm, LKH's CPU clock, 60 s after the first trial. The share of first-trial tour edges with a zero alpha-value
  was 0.757 under LKH's own alpha-values and 0.444 under 0, 100, 200, ... The rank-encoded continuation completed
  171 trials against 271, and its tour was 0.021% longer after 10 s (14 of 18 instances, p = 0.031) and 0.007% longer
  after 59 s (95% CI -0.001 to +0.015). In the probe both files held the fathers.

Concerned: the restarted POPMUSIC controls, the counting arms and the BetaTour arms of runs v1 to v8. Not concerned:
the uninterrupted POPMUSIC arms, the candidate-coverage results, and runs v9 to v12, which wrote no candidate file
of their own.

## Questions

- **H1 (encoding).** With the candidate file of runs v1 to v8, restarted LKH-3 finds longer tours within the budget
  than with a file equivalent to its own.
- **H2 (BetaTour with corrected files).** With alpha-values and fathers encoded as LKH-3 does, restarting LKH-3 with
  the BetaTour mixed lists gives shorter tours within the budget than uninterrupted POPMUSIC, all costs charged.

The results are reported in the paper whatever their direction.

## Design

- **Instances.** 24 new Euclidean instances of 10,000 nodes: 12 uniform (generator seed 5111) and 12 clustered
  (generator seed 5212), drawn with the generator of the earlier runs and scaled to the integer grid of side 10^6.
  Neither seed was used before. Budget 120 s, the budget of the 10,000-node instances in the earlier runs.
- **Processes.** Per instance and arm, six LKH-3.0.14 processes (seeds 101 to 106) in parallel on a machine with 12
  virtual CPUs; ascent-free POPMUSIC, `MAX_CANDIDATES = 5`, `RUNS = 1`, `TRACE_LEVEL = 2` in every process (one line
  per completed trial).
- **Shared first phase.** Per seed, POPMUSIC-5 lists and one trial; LKH writes its candidate file (lists,
  alpha-values, fathers). The six first-trial tours are the BetaTour and counting ensembles (M = 6), as in run v5.
- **Arms.**

  | Arm | Candidate lists | Alpha-values | Fathers |
  |---|---|---|---|
  | `pop_full` | POPMUSIC-5, uninterrupted | LKH's own (no file) | LKH's own |
  | `pop_native` | the seed's POPMUSIC-5 lists, restart | LKH's own | LKH's own |
  | `pop_rank` | the same lists, restart | 0, 100, 200, ... | none (`restart_pop_raw` of run v5) |
  | `beta_tree` | BetaTour mixed lists (model top 3, then POPMUSIC), restart | tree encoding | LKH's own |
  | `beta_rank` | the same lists, restart | 0, 100, 200, ... | none (`betatour` of run v5) |
  | `freq_tree` | counting mixed lists, restart | tree encoding | LKH's own |

  *Tree encoding:* an entry whose edge has alpha = 0 in the same seed's own POPMUSIC lists gets alpha = 0; the other
  entries get 100, 200, ... by list position. LKH sorts each list by alpha-value and then by length when it reads
  the file.
- **Time accounting (as in run v5).** One clock per instance and arm; for a restarted arm it starts with the shared
  first phase. The uninterrupted arm is stopped from outside at budget + 3 s and only tours found within the budget
  count. A
  restarted arm is charged the wall time of the shared first phase and the time to build its lists (successor
  arrays, model inference, mixing, encoding); the remainder is LKH's `TIME_LIMIT`, and tours count at first-phase
  time + list time + start-up delay + LKH's own time.
- **Order.** Instances alternate between the families. The order of the five restarted arms is a random permutation
  per instance (seed 21000 + position). The uninterrupted arm runs first on even and last on odd indices within a
  family. All arms of an instance run on one machine; an instance interrupted by a change of machine is run again
  from the start. A later session on another GPU type is refused unless `ALLOW_HARDWARE_CHANGE` is set; its use
  is a deviation and is reported.
- **Model.** Checkpoint `betatour_v2.pt` (SHA-256 2322f66c...3ebd70), repository commit d63f7e17. One inference on a
  random 10,000-node input precedes the first instance so that it does not carry the one-time GPU start-up.
- **Self-check.** Before the first instance the notebook repeats the deterministic check above on the machine of the
  run. It stops if the file rewritten with LKH's own alpha-values and fathers is not equivalent to LKH's own, or if
  the file with alpha = 0, 100, 200, ... does not change the search from the second trial on.

## Endpoints and tests

Percentage difference = 100 x (first arm - second arm) / second arm; negative values favour the first arm.

- **P1 (H1): `pop_rank` versus `pop_native`.** Per instance, the mean over the six seeds of the paired percentage
  difference of the best tour length at the budget (same seed, same first tour, same lists).
- **P2 (H2): `beta_tree` versus `pop_full`.** Per instance, the percentage difference of the best of six tour
  lengths at the budget.

Both: exact two-sided sign test over the 24 instances (ties excluded), Holm correction over the two tests,
alpha = 0.05; mean, median and 95% bootstrap interval (5,000 resamples of instances) are reported with them. H1 is
supported if `pop_rank` is longer on more instances and the adjusted p < 0.05. H2 is supported if `beta_tree` is
shorter on more instances and the adjusted p < 0.05.

Secondary, descriptive (no correction, labelled exploratory): both endpoints for every pair below; the same at 25,
50 and 75% of the budget; the two families separately.

| Pair | Reads as |
|---|---|
| `pop_native` / `pop_full` | cost of the restart itself with an equivalent file |
| `pop_rank` / `pop_full` | the restart comparison of run v5 |
| `beta_tree` / `pop_native` | effect of the lists, corrected files |
| `beta_rank` / `pop_rank` | effect of the lists, files of run v5 |
| `beta_rank` / `beta_tree` | effect of the encoding on the BetaTour lists |
| `beta_tree` / `freq_tree` | model versus counting |
| `freq_tree` / `pop_full` | counting versus uninterrupted POPMUSIC |
| `beta_rank` / `pop_full` | the end-to-end comparison of run v5 |

Mechanism measures per arm: trials completed within the budget; share of the first-trial tour's edges with a zero
alpha-value in the arm's file; list-building time.

## What each outcome means

- H1 supported: the restart penalty and the restart-matched comparisons of runs v1 to v8 were affected by the file
  encoding. The paper replaces them with the comparisons of this run and states the earlier values with this
  caveat.
- H1 not supported: the paper reports the encoding difference, the interval of P1 as the bound of its effect at the
  budget, and the time course.
- H2 supported: the statement that better edge predictions did not make LKH-3 faster no longer holds for this size.
  Before the paper claims a gain, the comparison is repeated on the other sizes and on TSPLIB (a new run with its
  own protocol), and `beta_tree` / `freq_tree` decides whether the gain needs the model.
- H2 not supported: the statement stands for this size on corrected files.

## Scope and limits

One size (10,000 nodes), two synthetic families, one budget, one LKH-3 version. P1 measures the two fields of the
file together; the probe measured the alpha-values alone. No interim decision is taken: the run ends at 24 instances
and the partial report printed during the run is exploratory. All 24 instances are analysed; an instance whose
uninterrupted arm does not reproduce the first-trial cost of the shared first phase is kept and flagged, and so is
an instance on which the tree rule, applied to LKH's own lists, does not reproduce LKH's zero alpha-values. Any
deviation from this protocol is listed in `v21_alpha_encoding/README.md`.

## Common rules

LKH-3.0.14 is built from the official archive or from a pinned copy checked by SHA-256
(4c4b9e6a...69e1cf33); the notebook checks that the compiled source contains the two rules under test. The
experiment code (`alphaenc.py`) is embedded in the notebook, compared with its SHA-256, copied to the result archive
and recorded in the manifest together with one SHA-256 over all code cells. A changed setting or code requires a new
experiment name.
