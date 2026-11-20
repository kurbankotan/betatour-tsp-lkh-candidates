# Development probe, 20 November 2026: combining LKH-3 tours with EAX (exploratory)

This probe was run in the development sandbox to decide what to test next. It is not part of any protocol, each
number comes from one run, and nothing here is a result for the paper. A Colab pilot with new instances and a rule
written beforehand is needed before any statement.

## Question

Six LKH-3 processes are run independently and the shortest of their six tours is taken. How much shorter is the tour
when their tours are combined, and how does a solver built on recombination (GA-EAX, Nagata and Kobayashi, 2013)
compare at equal total CPU time?

## Set-up

- Machine: Intel Xeon 2.30 GHz, 2 cores, one process per core unless noted. Times are CPU seconds of the processes.
- LKH-3.0.14, settings of the `pop_full` arm of runs v21 to v24: `CANDIDATE_SET_TYPE = POPMUSIC`, `SUBGRADIENT = NO`,
  `MAX_CANDIDATES = 5`, seeds 101 to 106, `TIME_LIMIT = 120` (preprocessing of 6.2 to 6.8 s not counted by LKH).
- GA-EAX from https://github.com/nagata-yuichi/GA-EAX (commit 015dfbe, MIT licence), version `Normal`, compiled with
  `g++ -O3`; one print statement was changed to show the CPU time per generation. Population 300 and 30 offspring
  when started from scratch; with a given population, the population size is the number of tours. The time printed
  by GA-EAX does not include its set-up (about 3 s at 10,000 nodes).
- Instance A: new synthetic uniform instance, 10,000 nodes (generator of the repository, seed 990117).
- Instance B: `usa13509` from TSPLIB (optimum 19,982,859). No LKH-3 run was made here; the LKH-3 values are those of
  run v22 on Colab (six processes, 270 s each), so this comparison is across machines.
- All tours were checked and their lengths recomputed with `tsplab.tour_length`.

## Instance A (uniform, 10,000 nodes)

Six LKH-3 tours: 71,864,895; 71,864,593; 71,861,449; 71,847,996; 71,853,250; 71,853,777 (best 71,847,996, mean
71,857,660; 762 CPU-s in total).

| Method | CPU-s | Length | Difference to the best of six |
|---|---|---|---|
| Six LKH-3 processes, best | 762 | 71,847,996 | 0 |
| One LKH-3 process (seed 101), `TIME_LIMIT = 720` | 727 | 71,854,096 | +0.0085% |
| Six tours, pairwise GPX2 (LKH's operator) | 762 + 0.3 | 71,844,821 | -0.0044% |
| Six tours, LKH-3 with `MERGE_TOUR_FILE` (10 s) | 762 + 13 | 71,843,431 | -0.0064% |
| Six tours, shortest tour in their union graph (HiGHS, proven) | 762 + 40 | 71,842,772 | -0.0073% |
| Six tours as the population of GA-EAX | 762 + 3 | 71,838,938 | -0.0126% |
| GA-EAX from scratch | 738 | 71,828,196 | -0.0276% |
| 30 LKH-3 first-trial tours, then GA-EAX | 68 | 71,844,734 | -0.0045% |
| 60 first-trial tours, then GA-EAX | 149 | 71,831,553 | -0.0229% |
| 100 first-trial tours (1 to 100), then GA-EAX | 216 | 71,828,555 | -0.0271% |
| 100 first-trial tours (101 to 200), then GA-EAX | 233 | 71,828,304 | -0.0274% |
| 200 first-trial tours, then GA-EAX | 389 | 71,829,475 | -0.0258% |

First-trial tours: one LKH-3 run writes the POPMUSIC-5 candidate file, the other runs read it (seeds 2001 onwards,
`MAX_TRIALS = 1`); 1.3 s per tour, mean length 72,041,265, 87% of the edges shared by two tours. The GA-EAX runs
with 30, 60 and 100 tours shared their core with another process, which lengthens their CPU time.

Against the GA-EAX tour found from scratch: the union of the six LKH-3 tours holds 9,907 of its 10,000 edges; a
LKH-3 tour has 537 to 851 edges that are not in it, and 86 to 91% of these are edges on which the six tours
disagree. Of the 8,504 edges common to the six tours, 76 are not in it.

GA-EAX from scratch was at 75,098,735 (+4.5%) after 120 CPU-s; it needs its whole run.

## Instance B (usa13509)

| Method | CPU-s | Above the optimum |
|---|---|---|
| Run v22, `pop_full`, best of six / mean of six | about 1,620 (Colab) | 0.0039% / 0.0241% |
| Run v22, `beta_tree`, best of six / mean of six | about 1,620 (Colab) | 0.0013% / 0.0093% |
| 100 first-trial tours, then GA-EAX | 384 | 0.0096% |
| 300 first-trial tours, then GA-EAX | 1,113 | 0.0034% |

On this instance the combination did not give a shorter tour than the six LKH-3 processes of run v22.

## Limits

One run per row, two instances, CPU time on a sandbox machine. GA-EAX is single-threaded, so in the wall-time
protocol of the paper (six processes in parallel) only the generation of the tours runs in parallel. The idea is not
new: tour merging of LKH tours (Cook and Seymour, 2003) and a cascade of local search and EAX (UNiCS,
arXiv:2501.14285) exist; no claim of novelty follows from this probe.

## Second part: the trial tours that LKH-3 discards

Every trial of LKH-3 ends in a locally optimal tour. LKH-3 merges it with its best tour and keeps only the best
tour. A copy of LKH-3.0.14 with one added block in `FindTour.c` wrote the tour of every trial to a file (after
`LinKernighan()`, before the merge; no search state is changed). On instance A the six runs of this copy ended with
the same six tour lengths as the unmodified LKH-3 and completed 553 to 604 trials in 120 s (575 to 632 without the
recording).

### Instance A: GA-EAX on recorded trial tours of the six processes

| Population | Tours | GA-EAX CPU-s | Length | Difference to the best of six | Edges of the GA-EAX tour found from scratch that are missing from the union |
|---|---|---|---|---|---|
| The six best tours (first part) | 6 | 3 | 71,838,938 | -0.0126% | 93 |
| Last 5 trial tours of each process at 120 s | 30 | 9 | 71,838,919 | -0.0126% | 25 |
| Last 10 of each at 120 s | 60 | 23 | 71,834,520 | -0.0188% | 3 |
| Last 17 of each at 120 s | 102 | 23 | 71,831,533 | -0.0229% | 0 |
| 17 of each, evenly spaced over the 120 s | 102 | 39 | 71,833,509 | -0.0202% | 0 |
| Last 17 of each at 20 s | 102 | 51 | 71,833,572 | -0.0201% | 3 |
| Last 34 of each at 20 s | 204 | 77 | 71,833,572 | -0.0201% | 1 |
| First 17 trials of each (first 7 s) | 102 | 49 | 71,840,135 | -0.0109% | 4 |
| One process only (seed 101), last 100 at 120 s | 100 | 32 | 71,858,396 | +0.0145% | 40 |

The one-process population gave a tour 0.0090% shorter than that process's own best tour (71,864,895). For
comparison, 100 first-trial tours followed by GA-EAX gave -0.0271% with 216 CPU-s in total (first part). The union
of the 102 tours of the fourth row has 14,656 edges, 6,138 of them common to all tours.

Further checks on instance A:

- Shortest tour in the union graph of the 102 tours (fourth and sixth row) with HiGHS and subtour cuts: not solved
  within 420 s (the tour returned was longer than that of GA-EAX). With six tours the same code had finished in 40 s.
- Second round: the six processes were continued from the merged tour of the fourth row for 20 s. One of them
  improved it (71,831,039). GA-EAX on the last 17 trial tours of each gave 71,830,812 (12 CPU-s).

### Instance C: new synthetic uniform instance, 50,000 nodes (seed 990250)

Six runs of the recording copy, POPMUSIC-5, 120 s (preprocessing 13.5 to 14.2 s, 43 to 47 trials). Best of six after
60 s: 160,077,903; after 120 s: 160,037,383 (mean of six 160,069,162). GA-EAX (version `Large`) on the last 17
trial tours of each process, read after 721 CPU-s of GA-EAX, when it had not yet converged:

| Population | LKH-3 CPU-s | Length after 721 CPU-s of GA-EAX | Difference to the best of six at 120 s |
|---|---|---|---|
| Trial tours up to 60 s | 444 | 159,945,236 | -0.0576% |
| Trial tours up to 120 s | 804 | 159,928,150 | -0.0683% |

No LKH-3 run with the same total CPU time or the same wall time was made on this instance, so these two rows are
not a comparison at equal cost: GA-EAX is single-threaded and took longer than the LKH-3 runs themselves. The
lengths of this table are those printed by GA-EAX during its run (the runs were stopped before they wrote their
tours). At generation 0 its best and mean lengths equalled the values computed with `tsplab.tour_length` for the 102
input tours. All tours of instances A and B were recomputed.

### Candidate files of NeuroLKH

The public NeuroLKH code (https://github.com/liangxinedu/NeuroLKH, commit 32ab884, `test.py` lines 62 to 71 and
`tsplib_test.py` lines 47 to 56) writes each line of its candidate file as node, father 0, five candidates with
alpha-values 0, 100, 200, 300, 400. This is the convention of the files of runs v1 to v8, whose effect on LKH-3.0.14
run v21 measured (longer tours on 19 of 24 instances, +0.0065%; about one third fewer trials). NeuroLKH was not run
here, and its own LKH source differs from LKH-3.0.14, so the size of the effect in its pipeline is not known.

### Literature checked for the second part (three searches, two pages read)

- Hains, Whitley and Howe, "Improving Lin-Kernighan-Helsgaun with crossover on clustered instances of the TSP":
  combines the partition crossover GPX with multi-trial LKH-2 (abstract only was read).
- Heins, Schäpermeier, Kerschke and Whitley, arXiv:2407.03927: candidate lists and LKH on instances of up to 2,000
  nodes; one candidate set is given through a candidate file, the alpha-values written are not described; "the
  impact of initial tours on EAX" is named as future work.
- No work was found that pools the discarded trial tours of LKH and recombines them as a population; this search is
  too small to conclude that none exists.

## Files

`tools/`: the scripts used (instance, six LKH runs, merge probe, first-trial tours, GA-EAX runs, edge comparison,
length check; for the second part `bp_pop.py`, `bp_batch.sh`, `run_dump.sh`, `run_round.sh`, `run_big.sh`,
`big_batch.sh`, `union_exact.py`, `best_at.py` and `FindTour_recording.patch.txt`). `logs/`: LKH-3 and GA-EAX
outputs of the rows above.
