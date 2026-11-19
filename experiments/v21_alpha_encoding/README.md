# Run v21: do the candidate files of the restarted arms reproduce LKH-3's own?

Runs v1 to v8 stopped LKH-3 after its first trial and started it again from a candidate file. LKH-3 uses two fields of
that file during the search, and the files of those runs reproduced neither: the alpha-values (the restart of every
later trial keeps an edge of the best tour only if its alpha-value is zero) and the father of each node in the minimum
1-tree (the first trial does not break 1-tree edges first). This run measures what the difference cost the restarted
arms and repeats the comparison between BetaTour lists and POPMUSIC with files encoded as LKH-3 encodes its own. The
protocol is in `../PROTOCOL_v21.md` and was written before the run.

Run numbers follow the laboratory record. Numbers 13 to 20 belong to a planned hard-instance run and to exploratory
solver pilots that are not part of the paper and are not in this repository.

## Design

- 24 new instances of 10,000 nodes (12 uniform, 12 clustered), budget 120 s, six LKH-3.0.14 processes (seeds 101-106).
- Six arms: uninterrupted POPMUSIC-5 (`pop_full`); restart with the seed's own lists and LKH's own alpha-values and
  fathers (`pop_native`) or with the file of run v5 (`pop_rank`); restart with the BetaTour mixed lists in the tree
  encoding (`beta_tree`) or with the file of run v5 (`beta_rank`); restart with the counting mixed lists in the tree
  encoding (`freq_tree`).
- Tree encoding: alpha = 0 for the entries whose edge has alpha = 0 in the same seed's own POPMUSIC lists, 100, 200,
  ... by position for the others, and the fathers of the seed's own file.
- Time accounting as in run v5: a restarted arm is charged the shared first phase and the time to build its lists.
- Primary comparisons, exact sign test over instances with Holm correction: `pop_rank` versus `pop_native` (mean of
  the six paired seeds) and `beta_tree` versus `pop_full` (best of six), both at the budget.

## Files

- `alphaenc.py`: encodings, LKH task functions that return alpha-values, fathers and trial end times, the share of
  tour edges that the restart can keep, the deterministic self-check and the paired statistics.
- `cells/`: the sources of the notebook cells (configuration, set-up, instances, experiment, self-check, run loop,
  analysis, archive).
- `build_colab_notebook.py`: writes `notebooks/23_BetaTour_v21_alpha_encoding_control.ipynb`. The module is embedded
  in a `%%writefile` cell and compared with its SHA-256 before it is imported.
- `tests/test_alphaenc.py`: unit tests of the module; with `--lkh` also the tests that run LKH-3.
- `tests/run_notebook_locally.py`: runs the notebook's code cells outside Colab in test mode (two seeds, two
  1,000-node instances, 6 s budget) and imitates a lost session between or inside instances.
- `probe/`: the exploratory measurement that preceded the protocol (scripts, per-instance table, summary).

```
python experiments/v21_alpha_encoding/tests/test_alphaenc.py --src src --lkh /path/to/LKH
BETATOUR_V21_TEST=1 BETATOUR_V21_REPO=. BETATOUR_V21_LKH=/path/to/LKH BETATOUR_V21_WORK=/tmp/v21_test \
    python experiments/v21_alpha_encoding/tests/run_notebook_locally.py
```

## Running it

Open the notebook in Colab with an A100 or L4 GPU and high RAM (12 vCPUs) and run all cells. Estimated time: set-up
and LKH build 3-5 min; warm-up and self-check 1 min; per instance about 11.5 min (shared first phase 15 s,
uninterrupted arm 2 min, five restarted arms 9 min); 24 instances about 4.6 h; analysis and archives 1 min. Completed
instances are kept on Drive (`MyDrive/BetaTour/v21_alphaenc24`) and skipped when the notebook is run again; an
instance interrupted by a change of machine runs again from the start. `MAX_NEW_INSTANCES` in the first cell limits
the number of new instances per session.

## Result

Not run yet.
