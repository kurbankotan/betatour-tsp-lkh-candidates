# Protocol for run v26: confirmation run on recombining the tours of LKH-3 processes with GA-EAX (written before the run)

## Why this run

Run v25 (pilot, 20 November 2026; `v25_trial_tour_recombination/README.md`) stopped six LKH-3 processes at half of
the wall-clock budget and gave their tours to GA-EAX (Nagata and Kobayashi, 2013) for the rest of the budget. Its
rule, written before the run, was met: GA-EAX on the best tour and the last 16 trial tours of each process (102
tours) gave a shorter tour at the budget than the six processes left running on all 21 instances (12 of 10,000 nodes:
mean -0.014%; 4 of 50,000 nodes: -0.019%; 5 TSPLIB instances: -0.021%). Two observations of the pilot shape this run:

- At 10,000 nodes the 102 tours gave shorter tours than the six best tours alone on 12 of 12 instances, and on the
  TSPLIB instances on 4 of 5 with one equal. At 50,000 nodes the order was reversed: the six best tours gave -0.066%
  and the 102 tours -0.019%, and GA-EAX did not converge on the 102 tours within its time. Keeping the shorter of the
  two results would have given -0.014% (10,000 nodes), -0.021% (TSPLIB) and -0.066% (50,000 nodes). This
  combination was chosen after the results; this run fixes it beforehand as its primary arm.
- The pilot had no arm in which LKH-3 merges its own six tours, so it does not show that GA-EAX is needed for the
  difference. This run adds that arm.

## Questions

The protocol of the pilot foresaw one primary comparison for a confirmation run. This run keeps that comparison as
H1 and adds three hypotheses; the four are tested as one family with Holm correction:

- **H1.** The primary arm (`dual_50`) gives shorter tours at the budget than the six LKH-3 processes left running
  (`lkh6`).
- **H2.** The primary arm gives shorter tours than LKH-3's own merging of the six tours at the same moment
  (`lkhmerge_50`).
- **H3.** The primary arm gives shorter tours than the known hybrid of first-trial tours and GA-EAX (`cascade`).
- **H4.** At 10,000 nodes, GA-EAX on the 102 tours gives shorter tours than GA-EAX on the six best tours
  (`trials_50` versus `best6_50`).

The results are reported whatever their direction.

## Design

- **Instances (38).** 28 new synthetic instances, generated with the generator of the earlier runs and scaled to the
  integer grid of side 10^6, with a budget of 12 ms per node: 16 of 10,000 nodes (8 uniform, generator seed 10111;
  8 clustered, seed 10212; 120 s), 6 of 20,000 nodes (3 uniform, seed 10313; 3 clustered, seed 10414; 240 s) and 6
  of 50,000 nodes (3 uniform, seed 10515; 3 clustered, seed 10616; 600 s). No generator seed was used before, in a
  run or in a probe. Ten TSPLIB instances, all EUC_2D with known optimum, with a budget of 0.02 s per node as in runs
  v22 and v25: pcb3038, fl3795, fnl4461, rl5915 and rl5934 (`tsplib_new`; not used with this idea before, used in
  run v22 with other arms) and rl11849, usa13509, brd14051, d15112 and d18512 (`tsplib_pilot`; the TSPLIB instances
  of the pilot, here with new seeds of LKH-3). The TSPLIB instances do not enter the tests.
- **Machine and clock.** One Colab machine with 12 virtual CPUs (L4 GPU and high RAM, as in run v25; the GPU is not
  used). The arms of an instance run one after another on one machine. All times are wall-clock times from the start
  of the arm; LKH's own clock is not used.
- **`lkh6` (baseline).** Six processes of the recording copy of LKH-3.0.14 (seeds 201 to 206; the earlier runs used
  101 to 106), ascent-free POPMUSIC with `MAX_CANDIDATES = 5` (the settings of `pop_full` in runs v21 to v25),
  stopped from outside 2 s after the budget. Each process writes the tour of every trial and every new best tour with
  a wall-clock stamp, and its candidate file. Result: the shortest best tour recorded up to the budget.
- **The cut.** At half of the budget (rounded down to the millisecond) the processes of `lkh6` are taken as stopped.
  The arms below start from the tours that the processes had recorded by then; `lkh6` is the continuation of the
  same processes. All arms of an instance except `cascade` thus share one recorded run up to the cut.
- **`trials_50` and `best6_50`.** Two populations at the cut: per process its best tour and its last 16 distinct
  trial tours that ended before the cut (102 tours), and the six best tours. GA-EAX (commit 015dfbe of the author's
  repository, 30 offspring per pair, version `Normal` up to 20,000 nodes and `Large` above, its fixed random seed,
  and the grid search for the lists of nearest cities of run v25) runs on each population in one process. The two
  processes run at the same time on two cores; each is stopped at a wall-clock deadline: budget minus cut minus the
  time to select and write its population. The start-up of GA-EAX is inside this time. Result of each: the shorter of
  the best tour at the cut and the best tour of GA-EAX within its time. If a population holds fewer than two
  different tours, GA-EAX is not run and the result is the best tour at the cut.
- **`dual_50` (primary arm).** The shorter of the results of `trials_50` and `best6_50`. After the cut it uses two
  cores where `lkh6` uses six.
- **`lkhmerge_50` (control: LKH-3's own tour merging).** At the cut, the six best tours are given to six new LKH-3
  processes (seeds 301 to 306) as `MERGE_TOUR_FILE`s. With two or more such files, LKH-3 treats the edges that all
  of them share like fixed edges and drops from its candidate lists the edges that cannot be combined with them
  (`IsCommonEdge.c`, `IsPossibleCandidate.c`); the search runs on the rest. Each process starts from the best of the
  six tours (`INITIAL_TOUR_FILE`) and reads the candidate file of the recorded process with the same position, so
  the preprocessing is not repeated. The processes are stopped from outside at the end of the remaining time (budget
  minus cut minus the time to write the tours); only tours that they recorded within that time count. Result: the
  shorter of the best tour at the cut and the best tour that the six processes recorded. If the six tours are all the
  same, the result is that tour.
- **`cascade` (known hybrid).** As in run v25: six workers produce first-trial tours of LKH-3 (seeds 3001 onwards; a
  worker's first run builds the POPMUSIC-5 candidate file, its later runs read it) until 100 tours exist or 40% of
  the budget has passed; GA-EAX starts from them and is stopped at the budget.
- **`eax_scratch` (reference, not an arm).** GA-EAX with the author's default settings (population 300, 30
  offspring), one process, stopped after six times the budget (the total CPU time of six processes); up to six
  instances in parallel after all instances are done; on the 16 instances of 10,000 nodes and on `tsplib_new` only.
- **Order.** Instances by size (10,000, 20,000 and 50,000 nodes, families alternating within a size), then
  `tsplib_new`, then `tsplib_pilot`. Within a family or a TSPLIB set, the recorded run with its derived arms precedes
  `cascade` on positions 0, 1, 4, 5, ... and follows it on positions 2, 3, 6, 7, ...; the two GA-EAX processes
  precede LKH-3's merging on even positions and follow it on odd positions. An instance interrupted by a change of
  machine is run again from the start. A later session on another machine type is refused unless
  `ALLOW_HARDWARE_CHANGE` is set; its use is a deviation and is reported.
- **Checks.** Before the first instance the notebook checks on the machine of the run that the recording copy of
  LKH-3, while it writes its candidate file, gives the same cost at each of 25 trials as the unmodified LKH-3; that
  GA-EAX with the grid search gives the same result as GA-EAX without it; and that LKH-3's merging reads the
  candidate file, keeps the common edges, gives the same search with and without recording and does not lengthen
  the tour. Every tour that enters a population or a merge and every result tour is checked to be a tour, and its
  length is recomputed with `tsplab.tour_length`; the length reported by GA-EAX must equal it. The final tour of
  every arm and reference run is stored in the result archive.
- **Protocol on GitHub.** The notebook refuses to start unless this file, on the main branch on GitHub, is byte for
  byte the protocol it was built for and the notebook on GitHub carries the same build hash. It writes the commit
  and the commit time of this file, and the time of the check, to the manifest before the first instance.

## Endpoints and tests

Percentage difference = 100 x (first arm - second arm) / second arm, of the results at the budget; negative values
favour the first arm. One value per instance.

| Test | First arm / second arm | Instances |
|---|---|---|
| H1 | `dual_50` / `lkh6` | the 28 synthetic instances |
| H2 | `dual_50` / `lkhmerge_50` | the 28 synthetic instances |
| H3 | `dual_50` / `cascade` | the 28 synthetic instances |
| H4 | `trials_50` / `best6_50` | the 16 instances of 10,000 nodes |

Each test: exact two-sided sign test over the instances (ties excluded); Holm correction over the four tests,
alpha = 0.05; the mean, the median and a 95% bootstrap interval (5,000 resamples of instances) are reported with
them. A hypothesis is supported if the first arm is shorter on more instances and the adjusted p < 0.05.

Secondary readings, descriptive (no correction, labelled exploratory): the four comparisons per size class, per
family and on each TSPLIB set; the pairs below; the distance to the optimum on the TSPLIB instances.

| Pair | Reads as |
|---|---|
| `trials_50` / `lkh6`, `best6_50` / `lkh6` | each population alone |
| `lkhmerge_50` / `lkh6` | LKH-3's own merging against no merging |
| `best6_50` / `lkhmerge_50` | the same six tours, merged by GA-EAX and by LKH-3 |
| `cascade` / `lkh6` | the known hybrid against no merging |
| `trials_50` / `best6_50` at 20,000 and 50,000 nodes | the reversal seen in the pilot |
| `eax_scratch` / `lkh6`, `dual_50` / `eax_scratch` | GA-EAX from scratch at equal total CPU time |

Mechanism measures: trials per process before the cut; union and common edges of the populations; time and
convergence of GA-EAX; which population gave the result of `dual_50`; the moment at which `dual_50` reached the final
length of `lkh6`; processes of `lkhmerge_50` with a tour within the time, their trials and whether their tours keep
the common edges; process-seconds of each arm as a share of those of `lkh6`.

## What each outcome means

- H1 supported: on new instances of 10,000 to 50,000 nodes with 12 ms per node, stopping six LKH-3 processes at half
  of the budget and recombining their tours with two GA-EAX processes gives shorter tours than letting the six
  processes run, at equal wall-clock time on one machine. Not supported: the result of the pilot is not confirmed,
  and the paper reports it as exploratory.
- H2 supported: the difference is not obtained by LKH-3's own merging of the same tours in the same time. Not
  supported: GA-EAX is not shown to be needed, and the paper says so.
- H3 supported: the arm gives shorter tours than the known hybrid. Not supported: no improvement on it is claimed.
- H4 supported: at 10,000 nodes the trial tours that LKH-3 discards add to the six best tours. Not supported: no such
  contribution is claimed.
- In every case the size classes are reported separately, and no statement extends beyond the sizes, families,
  budget and solver versions of this run.

## Scope and limits

- Three sizes, two synthetic families, one budget per node, one cut, one LKH-3 version, one GA-EAX version with its
  fixed random seed (each population is recombined once), one machine type.
- `lkh6`, `trials_50`, `best6_50`, `dual_50` and `lkhmerge_50` share the recorded run up to the cut, so the
  comparisons among them measure what is done with the second half of the budget.
- GA-EAX is single-threaded: `dual_50` uses two cores after the cut, `lkh6` and `lkhmerge_50` six.
- LKH-3's merging is given the six best tours, as LKH-3 is used for tour merging; it is not given the 102 tours.
  LKH-3's genetic mode (`POPULATION_SIZE`), which recombines the tours of successive runs within one process, is not
  run.
- Wall-clock time on a shared cloud machine; the self-check and the machine type are recorded.
- The ten TSPLIB instances do not enter the tests; five of them were used in the pilot.
- Tour merging of LKH tours (Cook and Seymour, 2003), partition crossover with multi-trial LKH (Hains, Whitley and
  Howe) and a cascade of local search and EAX (UNiCS, arXiv:2501.14285) exist. The run measures the gain of this
  combination on these instances; no claim of novelty follows from it.
- No interim decision is taken: the run ends when all 38 instances are done, and a partial report printed during the
  run is exploratory. All instances are analysed. The notebook stops if a process of `lkh6` has no tour at the cut;
  such a stop and every other deviation from this protocol are listed in `v26_recombination_confirmation/README.md`.

## Common rules

LKH-3.0.14 from the official archive or from the pinned copy checked by SHA-256; `FindTour.c` must have the SHA-256
recorded in `byprod.py` before it is patched. GA-EAX from https://github.com/nagata-yuichi/GA-EAX at the pinned
commit, checked by a SHA-256 over its source files. Repository commit d63f7e17 for `tsplab.py`. The module
`byprod.py` is embedded in the notebook, compared with its SHA-256, copied to the result archive and recorded in the
manifest together with the SHA-256 of this protocol and one SHA-256 over all code cells, the module and this
protocol. A changed setting or code requires a new experiment name.
