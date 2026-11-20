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

## Result (run on 19 November 2026)

All 24 instances completed in one Colab session (NVIDIA L4, 12 vCPUs). The protocol was pushed at 12:59 UTC (commit
3223ac8) and the first instance finished at 13:15 UTC. There was no deviation from the protocol. The self-check passed
on the machine of the run, the first trial of every uninterrupted process equalled the shared first phase, the tree
rule reproduced LKH's zero alpha-values for every seed, and 2 of the 720 restarted processes were stopped by the
outside deadline before their own time limit.

Both primary comparisons supported their hypotheses (exact sign test over the 24 instances, Holm-adjusted).

- **P1.** With the file of runs v1 to v8, restarted LKH-3 found longer tours than with the file equivalent to its own
  on 19 instances and shorter tours on 5 (mean of six seeds; mean difference +0.0065%, 95% interval +0.0027 to
  +0.0103, adjusted p = 0.013).
- **P2.** LKH-3 restarted with the BetaTour lists in the tree encoding found shorter tours than uninterrupted POPMUSIC
  on 18 instances and longer tours on 6 (best of six; mean difference -0.0059%, 95% interval -0.0093 to -0.0028,
  adjusted p = 0.023).

Secondary comparisons at the budget (best of six, descriptive; instances on which the first arm was shorter / equal /
longer):

| First / second arm | Instances | Mean difference (%) | 95% interval |
|---|---|---|---|
| `pop_native` / `pop_full` | 12 / 0 / 12 | -0.0019 | -0.0059 to +0.0020 |
| `pop_rank` / `pop_full` | 3 / 0 / 21 | +0.0077 | +0.0048 to +0.0105 |
| `beta_tree` / `pop_native` | 17 / 0 / 7 | -0.0040 | -0.0072 to -0.0008 |
| `beta_rank` / `pop_rank` | 21 / 0 / 3 | -0.0089 | -0.0115 to -0.0061 |
| `beta_rank` / `beta_tree` | 5 / 0 / 19 | +0.0047 | +0.0020 to +0.0079 |
| `beta_tree` / `freq_tree` | 12 / 0 / 12 | +0.0004 | -0.0029 to +0.0038 |
| `freq_tree` / `pop_full` | 17 / 0 / 7 | -0.0062 | -0.0097 to -0.0027 |
| `beta_rank` / `pop_full` | 12 / 0 / 12 | -0.0012 | -0.0043 to +0.0020 |

With an equivalent file the restart itself cost nothing measurable at the budget (`pop_native` / `pop_full`). The
file of run v5 reproduced the restart penalty of that run (`pop_rank` / `pop_full`: +0.0077% here, +0.0080% on the
sixteen 10,000-node instances of run v5), and with it the BetaTour arm tied uninterrupted POPMUSIC as it did there
(`beta_rank` / `pop_full`: 12 / 0 / 12 here, 9 / 0 / 7 there). Lists from tour counting did as well as the BetaTour
lists (`beta_tree` / `freq_tree`), so at this size the gain over POPMUSIC did not need the learned model.

Trials completed within the budget (mean of seeds and instances): 316 (`pop_full`), 308 (`pop_native`), 205
(`pop_rank`), 259 (`beta_tree`), 200 (`beta_rank`), 267 (`freq_tree`). Share of the first-trial tour's edges with a
zero alpha-value: 0.755 in the three corrected files, 0.443 (`pop_rank`) and 0.490 (`beta_rank`).

The difference between `pop_rank` and `pop_native` was largest early (+0.018% at a quarter of the budget, mean of six
seeds) and was still present at the budget. `beta_tree` was behind `pop_full` at a quarter of the budget (+0.006%,
11 / 0 / 13), ahead at half of it (-0.006%, 17 / 0 / 7) and furthest ahead at three quarters (-0.007%, 20 / 0 / 4).

On one clustered instance (`enc_clu10k_006`), three of the six `beta_tree` runs and one run each of `pop_full`,
`pop_native` and `pop_rank` were still about 0.36% above the shortest tour at the budget. This widens the intervals of
the seed-mean endpoint in the pairs with `beta_tree`; best of six is not affected.

Consequences, as fixed in the protocol: the restart penalty and the restart-matched comparisons of runs v1 to v8 were
affected by the file encoding, and the statement that the better lists did not make LKH-3 faster no longer holds for
10,000-node instances. Run v22 repeated the comparison on the other sizes and on TSPLIB
(`../v22_other_sizes/README.md`), and run v23 repeated it on 12 further 10,000-node instances
(`../v23_restart_marking/README.md`).

`python analysis/v21_file_encoding.py results/v21/v21_alphaenc24_results.zip results/v5/paper51_v1_results.zip`
recomputes these numbers from the instance records. The intervals above are those of the notebook
(`summary/paired_stats.csv`); the script draws its own bootstrap resamples and can differ in the last digit.
