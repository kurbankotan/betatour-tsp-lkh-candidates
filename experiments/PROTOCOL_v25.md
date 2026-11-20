# Protocol for run v25: recombining the trial tours that LKH-3 discards (pilot; written before the run)

## Why this run

Every trial of LKH-3 ends in a locally optimal tour. LKH-3 merges it with its best tour and keeps the best tour;
the trial tour itself is discarded. A probe in the development sandbox (20 November 2026; one run per number;
`v25_trial_tour_recombination/probe/README.md`) recorded these tours with a copy of LKH-3.0.14 that runs the same
search, and gave them to GA-EAX (Nagata and Kobayashi, 2013) as its population:

- One uniform instance of 10,000 nodes, six processes, 120 s. The shortest of the six tours was the reference.
  GA-EAX on the six best tours: -0.0126%. GA-EAX on the last 17 trial tours of each process: -0.0229% (23 CPU-s of
  GA-EAX). The union of the six best tours lacked 93 of the 10,000 edges of the shortest tour known for the
  instance; the union of the 102 trial tours lacked none. For comparison, 100 first-trial tours followed by GA-EAX
  gave -0.0271%, and GA-EAX from scratch -0.0276% with about the total CPU time of the six processes.
- One uniform instance of 50,000 nodes: GA-EAX on the trial tours of the first 60 s was 0.058% shorter than the
  best of six after 120 s, but GA-EAX had run for 721 CPU-s on one core and had not converged, and no LKH-3 run of
  equal cost was made.

The probe therefore shows that the discarded tours hold information that the six best tours lack, and it does not
show that using them pays at equal cost. This pilot measures that on new instances, with every arm limited to the
same wall-clock budget on the same machine. It is exploratory: it decides whether a confirmation run is prepared.

## Questions

- **Q1.** At 10,000 nodes, does stopping the six LKH-3 processes at half of the budget and recombining their best
  tours and last trial tours with GA-EAX give a shorter tour at the budget than letting the six processes run on?
- **Q2.** Is the difference larger at 50,000 nodes?
- **Q3.** Do the discarded trial tours matter, that is, is the result better than recombining the six best tours
  only?

Reported without a rule: the comparison with the known hybrid (first-trial tours followed by GA-EAX), with GA-EAX
from scratch at equal total CPU time, and the distance to the optimum on five TSPLIB instances.

## Design

- **Instances (21).** 12 new instances of 10,000 nodes (6 uniform, generator seed 9111; 6 clustered, seed 9212),
  budget 120 s. 4 new instances of 50,000 nodes (2 uniform, seed 9515; 2 clustered, seed 9616), budget 600 s
  (12 ms per node, as at 10,000 nodes). The five TSPLIB instances of 11,849 to 18,512 nodes of the earlier runs,
  all EUC_2D with known optimum (rl11849, usa13509, brd14051, d15112, d18512), budget 0.02 s per node as in run
  v22. No seed was used before, in a run or in the probe.
- **Machine and clock.** One Colab machine with 12 virtual CPUs; all arms of an instance run on one machine. All
  times are wall-clock times from the start of the arm. LKH's own clock is not used.
- **`lkh6` (comparison arm).** Six processes of the recording copy of LKH-3.0.14 (seeds 101 to 106), ascent-free
  POPMUSIC with `MAX_CANDIDATES = 5` (the settings of `pop_full` in runs v21 to v24), stopped from outside 2 s after
  the budget. The recording copy writes the tour of every trial and every new best tour with a wall-clock stamp;
  the self-check of the notebook requires that it gives the same cost at each of 25 trials as the unmodified
  LKH-3. Result: the shortest best tour recorded up to the budget.
- **Merge arms.** For each cut at 1/3, 1/2 and 2/3 of the budget, two populations are built from the same recorded
  run: `trials` (per process its best tour at the cut and its last 16 distinct trial tours that ended before the
  cut; 102 tours) and `best6` (the six best tours at the cut). GA-EAX (commit 015dfbe of the author's repository,
  30 offspring per pair, version `Normal` up to 20,000 nodes and `Large` above) starts from the population in one
  process and is stopped at a wall-clock deadline: budget minus cut minus the time to select and write the
  population. The start-up of GA-EAX (reading the instance, lists of nearest cities) is inside this time. Its
  lists of nearest cities are found with a grid search that returns the lists of the original loops (checked entry
  by entry in the tests, and by an identical GA-EAX result in the self-check of the notebook); the original loops
  take about a minute at 50,000 nodes. Result: the shorter of the best tour at the cut and the best tour of GA-EAX at its deadline. The six
  GA-EAX processes of an instance run in parallel. After the cut the other five cores are idle in these arms.
- **`cascade` (known hybrid).** Six workers produce first-trial tours of LKH-3 in parallel (a worker's first run
  builds the POPMUSIC-5 candidate file, its later runs read it; seeds 2001 onwards) until 100 tours exist or 40% of
  the budget has passed; GA-EAX starts from them and is stopped at the budget.
- **`eax_scratch` (reference, not an arm).** GA-EAX with the author's default settings (population 300, 30
  offspring), one process, stopped after six times the budget (the total CPU time of six processes); up to six
  instances in parallel; instances of at most 20,000 nodes only.
- **Order.** 10,000-node instances first (families alternating), then TSPLIB, then 50,000 nodes. Within a group
  the recorded run and its merges precede the cascade on even positions and follow it on odd positions.
- **Checks.** Every tour that enters a population and every result tour is checked to be a tour, and its length is
  recomputed with `tsplab.tour_length`; the length reported by GA-EAX must equal it. The final tour of every arm is
  stored in the result archive.

## Readings and decision rule

Percentage difference = 100 x (first arm - second arm) / second arm at the budget; negative values favour the first
arm. One value per instance. Reported per size class with wins, ties and losses, the exact two-sided sign test, mean,
median and 95% bootstrap interval (5,000 resamples of instances). No correction for multiple comparisons; nothing in
this run is a confirmatory test.

- **D1 (Q1).** On the 12 instances of 10,000 nodes, `trials_50` is shorter than `lkh6` on more instances and the
  sign test gives p < 0.05 (at least 10 of 12 when none ties).
- **D2 (Q2).** `trials_50` is shorter than `lkh6` on all four instances of 50,000 nodes, and the mean difference
  there is larger in size than at 10,000 nodes.
- **D3 (Q3, informative).** On the 12 instances of 10,000 nodes, `trials_50` is shorter than `best6_50` on more
  instances with p < 0.05.

If D1 and D2 both hold, a confirmation run is prepared: new instances, one primary comparison, and a protocol that
is committed and pushed before that run starts. Otherwise no confirmation run is prepared and the pilot is recorded
as not supporting the idea. The cuts at 1/3 and 2/3 are secondary and do not enter the rule.

## Scope and limits

- 21 instances, one LKH-3 version, one GA-EAX version with its fixed random seed, one machine type.
- `lkh6` and the merge arms come from the same recorded run, so they share the search up to the cut; the
  comparison measures what is done with the second half of the budget.
- GA-EAX is single-threaded. The merge arms and the cascade therefore use one core where `lkh6` uses six; an
  implementation that used the idle cores could do better, and this pilot does not measure it.
- At 50,000 nodes GA-EAX may not converge within its share of the budget; its best tour at the deadline counts.
- With four instances, D2 cannot be tested for significance; it is a threshold for going on.
- The idea of merging LKH tours is old (Cook and Seymour, 2003), partition crossover has been combined with
  multi-trial LKH (Hains, Whitley and Howe), and a cascade of local search and EAX exists (UNiCS,
  arXiv:2501.14285). A search of the literature on 20 November 2026 found no work that pools the discarded trial
  tours as a population, which is not proof that none exists. No claim of novelty is made here.
- This protocol was written before the run. By the author's decision the repository is updated after the result,
  so the protocol has no public time stamp that precedes the run.

## Common rules

LKH-3.0.14 from the official archive or from the pinned copy checked by SHA-256; `FindTour.c` must have the SHA-256
recorded in `byprod.py` before it is patched. GA-EAX from https://github.com/nagata-yuichi/GA-EAX at the pinned
commit, checked by a SHA-256 over its source files. Repository commit d63f7e17 for `tsplab.py`. The module
`byprod.py` is embedded in the notebook, compared with its SHA-256, copied to the result archive and recorded in the
manifest together with one SHA-256 over all code cells. A changed setting or code requires a new experiment name.
