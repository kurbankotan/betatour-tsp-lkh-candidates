# Results of the four runs

Each folder holds the outputs of one notebook exactly as it wrote them to Google Drive. During development the model was called TourFold; see the naming note in the main README.

| Folder | Notebook | Content |
| --- | --- | --- |
| `v1/` | 01 | data generation, BetaTour-kNN and single-instance training, first LKH tests (Appendix A.6) |
| `v2/` | 02 | BetaTour training, coverage study, first single-core comparison (LKH-3 default, edge fixing, 16-tour ensembles) |
| `v3/` | 03 | coverage of mixed lists and smaller ensembles, single-core comparison with control and mixed lists |
| `v4/` | 04 | multi-core multi-start comparison |

Files in each folder:

- `RAPOR*.md`: the automatic report of the run (Turkish), including the run time of every stage.
- `config*.json`: the settings of the run.
- `log*.txt`, `status*.json`: the progress log and the final status.
- `results/`: result tables (`.csv`) and summaries (`.json`).
  - `lkh_eval_parts/` (v1 to v3): one record per instance and configuration with the improvement trace of the LKH-3 run (`events`: pairs of tour length and wall-clock seconds since the configuration started), the overhead charged to the budget (`offset`, `ens_time`) and, in run v3, the list length.
  - `instances/` (v4): one file per instance with the traces of the six baseline runs and the six BetaTour runs and the timing of the pause.
  - `*_wtl.csv`: wins, ties and losses of each configuration against the others.
  - `coverage_v3.*`, `decision*.json`: candidate coverage (missing reference-tour edges per 1,000 nodes, share covered by the first two and three candidates).
- `figures/`: the figures drawn by the notebook (gap over time per instance set, coverage).

Frequently used Turkish terms in file and column names: `topluluk` ensemble, `sayım` counting, `omurga sabit` edge fixing, `süzülmüş omurga` filtered edge fixing, `varsayılan` default, `ascent yok` no ascent, `ilk denemeler` first trials, `devam anı` resume time, `bütçe payı` share of the budget, `örnek` instance.
