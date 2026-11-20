# Run v26: confirmation run on recombining the tours of LKH-3 processes with GA-EAX

Run v25 (pilot, `../v25_trial_tour_recombination/README.md`) stopped six LKH-3 processes at half of the wall-clock
budget and recombined their tours with GA-EAX (Nagata and Kobayashi, 2013); its rule was met on 21 of 21 instances.
This run repeats the comparison on 28 new synthetic instances with four hypotheses fixed beforehand, a primary arm
that runs both populations of the pilot at the same time, and a new control in which LKH-3 merges its own six tours.
The protocol is `../PROTOCOL_v26.md`; the notebook refuses to start unless that file and the notebook itself are on
the main branch on GitHub as built.

## Design

- 38 instances: 16 of 10,000 nodes (120 s), 6 of 20,000 nodes (240 s) and 6 of 50,000 nodes (600 s), all new and
  half uniform, half clustered; ten TSPLIB instances with known optimum (pcb3038, fl3795, fnl4461, rl5915, rl5934 and
  the five of the pilot; 0.02 s per node), which do not enter the tests.
- `lkh6`: six LKH-3.0.14 processes (POPMUSIC-5, seeds 201-206) for the whole budget; the shortest of their tours.
- `dual_50` (primary): the same processes taken as stopped at half of the budget; two GA-EAX processes run at the
  same time for the rest, one on the best tour and the last 16 trial tours of each process (`trials_50`, 102 tours),
  one on the six best tours (`best6_50`); the shorter result counts.
- `lkhmerge_50` (control): the six best tours at half of the budget given to six new LKH-3 processes as
  `MERGE_TOUR_FILE`s (LKH-3's own tour merging), for the rest of the budget.
- `cascade` (known hybrid): up to 100 first-trial tours of LKH-3, then GA-EAX, within the budget.
- `eax_scratch` (reference): GA-EAX from scratch with six times the budget in one process; 10,000 nodes and the five
  smaller TSPLIB instances only.
- Tests (28 synthetic instances, exact sign test, Holm over four tests): H1 `dual_50` / `lkh6`, H2 `dual_50` /
  `lkhmerge_50`, H3 `dual_50` / `cascade`; H4 `trials_50` / `best6_50` on the 16 instances of 10,000 nodes.
- All times are wall-clock times on one machine with 12 virtual CPUs.

## Files

- `byprod.py`: the module of run v25 with three additions: a mode of the recording patch that writes only the new
  best tours (`BETATOUR_BEST_ONLY`, for the merging processes), `MERGE_TOUR_FILE` lines in `write_par`, and helpers
  for common edges, distinct tours and the Holm correction. Pure Python and NumPy.
- `cells/`: the sources of the notebook cells (configuration, set-up with the check of the protocol on GitHub,
  instances, the arms, self-check, run loop, analysis, archive).
- `build_colab_notebook.py`: writes `notebooks/28_BetaTour_v26_recombination_confirmation.ipynb`, embeds `byprod.py`
  and writes the SHA-256 of the protocol and of the build into the first cell.
- `tests/test_byprod.py`: tests of `byprod.py` with the real LKH-3.0.14 and GA-EAX sources (the tests of run v25,
  plus the best-only recording, LKH-3's merging with the recorded candidate file and the helpers).
- `tests/run_notebook_locally.py`: runs the notebook's code cells outside Colab in test mode (two processes, four
  small instances) and imitates a lost session between or inside instances.

```
python experiments/v26_recombination_confirmation/tests/test_byprod.py --src src \
    --lkh-source /path/to/LKH-3.0.14 --eax-source /path/to/GA-EAX --work /tmp/v26_unit
BETATOUR_V26_TEST=1 BETATOUR_V26_REPO=. BETATOUR_V26_LKH=/path/to/LKH-3.0.14/LKH \
    BETATOUR_V26_EAX_SRC=/path/to/GA-EAX BETATOUR_V26_WORK=/tmp/v26_test \
    python experiments/v26_recombination_confirmation/tests/run_notebook_locally.py
```

## Running it

Commit and push the protocol and the notebook first. Then open the notebook in Colab with an L4 GPU and high RAM
(12 vCPUs, 53 GB; the GPU is not used) and run all cells. Estimated time: set-up 5-7 min; self-check 2 min; 16
instances of 10,000 nodes at about 6 min each, 1.6 h; 6 of 20,000 nodes at about 12 min, 1.2 h; 6 of 50,000 nodes at
about 31 min, 3.1 h; the five smaller TSPLIB instances 0.4 h; the five larger 1.2 h; the GA-EAX reference 0.8 h;
analysis and archive 1 min; about 8.5 h in total (at most about 9 h). Completed instances are kept on Drive
(`MyDrive/BetaTour/v26_recomb38`) and skipped when the notebook is run again; an instance interrupted by a change of
machine runs again from the start. `MAX_NEW_INSTANCES` in the first cell splits the run into sessions (22, 6 and
`None` give sessions of about 3, 3.1 and 2.5 h).

## Result

Not run yet.
