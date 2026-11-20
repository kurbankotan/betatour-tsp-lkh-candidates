# Run v24: does the gain appear at 20,000 nodes with twice the budget?

With corrected candidate files, LKH-3 restarted with the BetaTour mixed lists gave shorter tours than uninterrupted
POPMUSIC at 10,000 nodes (runs v21 and v23) and no difference at 20,000 nodes with the budget of 300 s (run v22). In
run v22 the restarted arm closed its distance to the uninterrupted arm throughout the budget on the uniform
instances, and the cost of the restart itself was still falling at the end. This run repeats the comparison on new
20,000-node instances with a budget of 600 s. The protocol is in `../PROTOCOL_v24.md`; it was written before the run
and states the exploratory evidence behind it.

## Design

- 20 new instances of 20,000 nodes (10 uniform, 10 clustered), budget 600 s, six LKH-3.0.14 processes (seeds
  101-106).
- Three arms, as in runs v21 and v22: uninterrupted POPMUSIC-5 (`pop_full`); restart with the seed's own lists,
  alpha-values and fathers (`pop_native`); restart with the BetaTour mixed lists in the tree encoding (`beta_tree`).
- One primary comparison: `beta_tree` versus `pop_full` on the mean of the six paired seeds at the budget, exact
  two-sided sign test over the 20 instances.
- The outside deadline is 60 s after the budget (3 s in the earlier runs) and is a safeguard only: LKH-3's clock ran
  about 1.5% behind the wall clock at 20,000 nodes in run v22, and the deadline came before the processes' own clock
  reached the budget.

## Files

- `cells/`: the sources of the notebook cells. They are those of run v21 with the settings of this run, without its
  arms for the old file format and for counting; the analysis cell reports the comparison at the budget and at
  half of it (the budget of run v22), and flags processes whose last recorded trial ended before the budget.
- `build_colab_notebook.py`: writes `notebooks/26_BetaTour_v24_longer_budget_20k.ipynb` and embeds
  `../v21_alpha_encoding/alphaenc.py` unchanged.
- `tests/run_notebook_locally.py`: runs the notebook's code cells outside Colab in test mode (two seeds, two
  instances of 1,000 nodes, budget 6 s) and imitates a lost session between or inside instances. The unit tests of
  the module are those of run v21.

```
BETATOUR_V24_TEST=1 BETATOUR_V24_REPO=. BETATOUR_V24_LKH=/path/to/LKH BETATOUR_V24_WORK=/tmp/v24_test \
    python experiments/v24_longer_budget/tests/run_notebook_locally.py
```

## Running it

Commit and push `../PROTOCOL_v24.md` first. Then open the notebook in Colab with an L4 or A100 GPU and high RAM (12
vCPUs) and run all cells. Estimated time: set-up and LKH build 3-5 min; warm-up and self-check 1-2 min; 20 instances
at about 30 min each (shared first phase 33 s, uninterrupted arm 10.4 min, two restarted arms 19.3 min), 10 h;
analysis and archives 1 min. Completed instances are kept on Drive (`MyDrive/BetaTour/v24_long20k20`) and skipped
when the notebook is run again; an instance interrupted by a change of machine runs again from the start.
`MAX_NEW_INSTANCES = 7` in the first cell gives three sessions of about 3.5 h, and `10` two sessions of about 5 h.

## Result

Not run yet.
