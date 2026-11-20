# Run v23: restart markings in candidate files (exploratory)

In a candidate file of LKH-3, the entries with alpha = 0 decide which edges of the best tour every trial after the
first keeps and where its search can start. LKH-3 marks the edges of its minimum 1-tree, and the corrected files of
run v21 used the same marks. This run tries marks taken from the six first-trial tours: by counting (an edge found in
at least five of the six tours) and by the BetaTour model (an edge whose predicted probability passes a threshold that
protects as many edges of the seed's first tour as the count marking). The protocol is in `../PROTOCOL_v23.md` and
was written before the run. It states the evidence before the run (an offline analysis in favour of the idea, a
small probe with LKH-3 that showed no clear gain) and the rule by which the run decides whether a confirmation run
is prepared. Nothing in this run is a confirmatory test.

## Design

- 12 new instances of 10,000 nodes (6 uniform, 6 clustered), budget 120 s, six LKH-3.0.14 processes (seeds 101-106).
- Six arms: uninterrupted POPMUSIC-5 (`pop_full`); restart with the seed's own lists and LKH's own marks
  (`pop_native`) or the model marking (`pop_model`); restart with the BetaTour mixed lists and the tree marking of
  run v21 (`beta_tree`), the model marking (`beta_model`) or the count marking (`beta_count`). Every file holds the
  fathers of the seed's own POPMUSIC file.
- Three readings on the mean of the six paired seeds at the budget: `beta_model` versus `beta_tree`, `beta_count`
  versus `beta_tree`, and `pop_model` versus `pop_native`. A marking is taken to a confirmation run if it is shorter
  on more instances and the exact two-sided sign test over the 12 instances gives p < 0.05.

## Files

- `marking.py`: the markings (edge counts over tours, undirected edge probabilities from the model output, the
  threshold of the model marking, the alpha-values of a marking). Pure NumPy.
- `cells/`: the sources of the notebook cells. The set-up, instance, self-check, run and archive cells are those of
  run v21 with the settings of this run (the set-up cell also checks the second embedded module); the experiment
  cell builds the five candidate files per seed, and the analysis cell reports the readings.
- `build_colab_notebook.py`: writes `notebooks/25_BetaTour_v23_restart_marking.ipynb` and embeds
  `../v21_alpha_encoding/alphaenc.py` (unchanged) and `marking.py`.
- `tests/test_marking.py`: unit tests of `marking.py`; with `--lkh` and `--checkpoint` also the markings on the
  candidate file of a real LKH-3 run and on the output of the model.
- `tests/run_notebook_locally.py`: runs the notebook's code cells outside Colab in test mode (two seeds, two
  instances of 1,000 nodes, budget 6 s) and imitates a lost session between or inside instances.

```
python experiments/v23_restart_marking/tests/test_marking.py --src src --lkh /path/to/LKH \
    --checkpoint checkpoints/betatour_v2.pt
BETATOUR_V23_TEST=1 BETATOUR_V23_REPO=. BETATOUR_V23_LKH=/path/to/LKH BETATOUR_V23_WORK=/tmp/v23_test \
    python experiments/v23_restart_marking/tests/run_notebook_locally.py
```

## Running it

Open the notebook in Colab with an A100 or L4 GPU and high RAM (12 vCPUs) and run all cells. Estimated time: set-up
and LKH build 3-5 min; warm-up and self-check 1 min; 12 instances at about 11.5 min each (shared first phase 15 s,
uninterrupted arm 2 min, five restarted arms 9 min), 2.3 h; analysis and archives 1 min. Completed instances are
kept on Drive (`MyDrive/BetaTour/v23_marking12`) and skipped when the notebook is run again; an instance interrupted
by a change of machine runs again from the start. `MAX_NEW_INSTANCES = 6` in the first cell gives two sessions of
about 1.2 h.

## Result (run on 19 November 2026)

All 12 instances completed in one Colab session (NVIDIA L4, 12 vCPUs). The self-check passed on the machine of the
run, the first trial of every uninterrupted process equalled the shared first phase, the tree rule and the marking
writer reproduced LKH's values for every seed, and 1 of the 360 restarted processes was stopped by the outside
deadline before its own time limit. One deviation: the protocol file was written about half an hour before the run
started, but it was committed to the repository together with the results. The readings and the decision rule are
part of the analysis cell of the notebook, and the manifest that the notebook wrote before the first instance holds
the SHA-256 over the notebook's code cells (`notebook_build`, 4fbd5d8b...).

No marking passed the decision rule (mean of six seeds at the budget; instances on which the first arm was shorter /
equal / longer):

| Reading | Instances | Mean difference (%) | 95% interval | Sign test p |
|---|---|---|---|---|
| M1 `beta_model` / `beta_tree` | 7 / 0 / 5 | -0.0012 | -0.0039 to +0.0022 | 0.77 |
| M2 `beta_count` / `beta_tree` | 6 / 0 / 6 | -0.0022 | -0.0064 to +0.0024 | 1.00 |
| M3 `pop_model` / `pop_native` | 6 / 0 / 6 | -0.0030 | -0.0094 to +0.0028 | 1.00 |

As fixed in the protocol, no confirmation run is prepared, and the two markings are recorded as not supported at
this size. The intervals leave room for a gain of at most 0.004% (M1), 0.006% (M2) and 0.009% (M3).

The markings changed the files as intended. The share of the first-trial tour's edges with a zero alpha-value rose
from 0.757 (tree marking) to 0.823 (`pop_model`), 0.828 (`beta_model`) and 0.827 (`beta_count`). The thresholds of
the model marking were between 0.942 and 0.955, and the model marking protected 96% of the edges that the count
marking protected. The restarted processes completed more trials within the budget (`pop_model` 414 against 313 for
`pop_native`; `beta_model` 364 and `beta_count` 327 against 257 for `beta_tree`), and their tours were not
detectably shorter. The model marking and the count marking did not differ either (`beta_model` / `beta_count`:
4 / 0 / 8, +0.0011%, interval -0.0019 to +0.0037).

Comparisons that this run shares with run v21 (new instances in each run, not pooled):

| First / second arm | Endpoint | Run v21 (24 instances) | Run v23 (12 instances) |
|---|---|---|---|
| `beta_tree` / `pop_full` | best of six | 18 / 0 / 6, -0.0059% | 10 / 0 / 2, -0.0060% (-0.0103 to -0.0022, p = 0.039) |
| `beta_tree` / `pop_full` | mean of six | 22 / 0 / 2, -0.0028% | 9 / 0 / 3, -0.0018% (-0.0067 to +0.0053) |
| `beta_tree` / `pop_native` | best of six | 17 / 0 / 7, -0.0040% | 10 / 0 / 2, -0.0038% (-0.0072 to -0.0006) |
| `pop_native` / `pop_full` | best of six | 12 / 0 / 12, -0.0019% | 9 / 0 / 3, -0.0023% (-0.0048 to +0.0007) |

The second primary comparison of run v21 (`beta_tree` / `pop_full`, best of six) gave nearly the same mean difference
on the 12 new instances. The uninterrupted arm had the shortest tour of the six arms on none of the 12 instances
(best of six; `pop_native` on 1, `pop_model` on 3, `beta_tree` on 3, `beta_model` on 2, `beta_count` on 3).

On one clustered instance (`mark_clu10k_004`) the worst process of every arm was 0.12 to 0.18% above the shortest
tour at the budget; this widens the intervals of the seed-mean endpoint.

`python analysis/v23_restart_marking.py results/v23/v23_marking12_results.zip results/v21/v21_alphaenc24_results.zip`
recomputes these numbers from the instance records. The intervals above are those of the notebook
(`summary/paired_stats.csv`); the script draws its own bootstrap resamples and can differ in the last digit. The
report inside the ZIP is in Turkish.
