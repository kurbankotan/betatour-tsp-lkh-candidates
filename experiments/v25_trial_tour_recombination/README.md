# Run v25 (pilot): recombining the trial tours that LKH-3 discards

Every trial of LKH-3 ends in a locally optimal tour that is merged with the best tour and then discarded. This
pilot records these tours and gives them to GA-EAX (Nagata and Kobayashi, 2013) as its population, and it compares
the result with the six LKH-3 processes left running, at the same wall-clock budget. The protocol is in
`../PROTOCOL_v25.md`; the probe that led to the pilot is in `probe/README.md`. The pilot is exploratory: it decides
whether a confirmation run is prepared.

## Design

- 21 instances: 12 new instances of 10,000 nodes (budget 120 s), five TSPLIB instances with known optimum (rl11849,
  usa13509, brd14051, d15112, d18512; budget 0.02 s per node) and 4 new instances of 50,000 nodes (budget 600 s).
- `lkh6`: six LKH-3.0.14 processes (POPMUSIC-5, seeds 101-106) for the whole budget; the shortest of their tours.
- `trials_33`, `trials_50`, `trials_67`: the same processes stopped at that share of the budget; per process its
  best tour and its last 16 trial tours (102 tours) are recombined by GA-EAX in the remaining time.
  `trials_50` is the primary arm.
- `best6_33`, `best6_50`, `best6_67`: the same with the six best tours only (control).
- `cascade`: the known hybrid; up to 100 first-trial tours of LKH-3, then GA-EAX, within the budget.
- `eax_scratch`: reference; GA-EAX from scratch in one process with six times the budget (instances of at most
  20,000 nodes).
- All times are wall-clock times on one machine with 12 virtual CPUs.

## Files

- `byprod.py`: the three source patches (the recording calls in `FindTour.c` of LKH-3.0.14; time stamps and a
  wall-clock deadline in `env.cpp` of GA-EAX; a grid search for the lists of nearest cities in `evaluator.cpp` of
  GA-EAX, which returns the lists of the original loops in less than a second where the loops take about a minute
  at 50,000 nodes), reading of the recorded tours, populations, the GA-EAX runner and the paired statistics. Pure
  Python and NumPy.
- `cells/`: the sources of the notebook cells (configuration, set-up with both builds, instances, the arms,
  self-check, run loop, analysis, archive).
- `build_colab_notebook.py`: writes `notebooks/27_BetaTour_v25_trial_tour_recombination.ipynb` and embeds
  `byprod.py`.
- `tests/test_byprod.py`: tests of `byprod.py` with the real LKH-3.0.14 and GA-EAX sources (same search with and
  without recording, valid records, GA-EAX from recorded tours, deadline, the grid search against GA-EAX's own
  lists entry by entry, and the same GA-EAX result with and without it).
- `tests/run_notebook_locally.py`: runs the notebook's code cells outside Colab in test mode (two processes, three
  small instances) and imitates a lost session between or inside instances.
- `probe/`: record, scripts and logs of the development probe of 20 November 2026.

```
python experiments/v25_trial_tour_recombination/tests/test_byprod.py --src src \
    --lkh-source /path/to/LKH-3.0.14 --eax-source /path/to/GA-EAX --work /tmp/v25_unit
BETATOUR_V25_TEST=1 BETATOUR_V25_REPO=. BETATOUR_V25_LKH=/path/to/LKH-3.0.14/LKH \
    BETATOUR_V25_EAX_SRC=/path/to/GA-EAX BETATOUR_V25_WORK=/tmp/v25_test \
    python experiments/v25_trial_tour_recombination/tests/run_notebook_locally.py
```

## Running it

Open the notebook in Colab with an L4 GPU and high RAM (12 vCPUs, 53 GB; the GPU is not used) and run all cells.
Estimated time: set-up with two builds of LKH-3 and two of GA-EAX 4-6 min; self-check 1 min; 12 instances of 10,000
nodes at about 6 min each, 1.2 h; five TSPLIB instances at 11 to 17 min each, 1.1 h; four instances of 50,000 nodes
at about 28 min each, 1.9 h; the GA-EAX reference in three groups, 1.0 h; analysis and archive 1 min; about 5.3 h in
total. Completed instances are kept on Drive (`MyDrive/BetaTour/v25_trialtours21`) and skipped when the notebook is
run again; an instance interrupted by a change of machine runs again from the start. `MAX_NEW_INSTANCES = 12` in the
first cell ends the first session after the instances of 10,000 nodes (about 1.3 h); `RUN_REFERENCE = False` leaves
out the reference.

## Result

Not run yet.
