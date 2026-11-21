# Results of the runs

## Runs v1 to v4

Each of the folders `v1/` to `v4/` holds the outputs of one notebook exactly as it wrote them to Google Drive. During development the model was called TourFold; see the naming note in the main README.

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

## Runs v5 to v12, v21 to v23 and v25 to v27

From run v5 on, each notebook stored its records under `MyDrive/BetaTour/<experiment>` and exported a compact result ZIP at the end. The folders below hold these ZIPs unchanged, except for run v7 (see the note). Each ZIP contains `manifest.json` (settings, seeds, hardware, source commit, checkpoint and code hashes), `progress.json`, one record per instance or per instance and arm, and the `summary/` tables that the paper quotes. From run v9 on, the ZIPs also contain the experiment code that ran (`code/`) and the per-seed records written by the LKH hook and the coordinator (for example `run.par`, `applied.stats`, `best.events` and, in runs v11 and v11r2, `telemetry.txt` and `action_telemetry.tsv`). Raw tours, candidate snapshots and stored model outputs are in separate audit ZIPs that stayed on Drive (Appendix A of the paper). Run v12 stores one record per instance with the edge accounting of every seed; its audit ZIP (initial lists, final tours and best-tour edge sets) also stayed on Drive. Run v21 stores one record per instance with, for every arm and seed, the improvement trace and the end time of every trial; its audit ZIP (first-trial tours, LKH's own candidate lists with alpha-values and fathers, and the candidate files of the restarted arms) also stayed on Drive. Runs v22 and v23 store the same kind of record and audit ZIP as run v21. Run v25 (a pilot that does not use the BetaTour checkpoint and is not part of the paper) stores one record per instance with the time courses of the six LKH-3 processes and of every GA-EAX run, one record per reference run, and the final tour of every arm and of every reference run (`tours/`); it has no audit ZIP. Run v26 stores the same kind of records, with those of the merging processes of LKH-3 and the hash and commit of its protocol in the manifest. Run v27 (a pilot that is not part of the paper and uses the checkpoint in one arm) stores the records of run v26 without the known hybrid and the reference runs, adds those of its three exact arms (the graph, its reduction and the statistics of SCIP), and stores the six tours at the cut next to the final tours (`tours/`).

| Folder | Run | Notebook | File | Bytes | Entries | SHA-256 |
| --- | --- | --- | --- | ---: | ---: | --- |
| `v5/` | v5 | 05 | `paper51_v1_results.zip` | 1,440,440 | 466 | `64cf6adebb342bc30a24ae14c7abef8caa8d90c4e1fc01811e0f86d7bf091866` |
| `v6/` | v6 | 06 | `adaptive40_v1_results.zip` | 1,313,017 | 407 | `a2c0a8e55a9cd4781ec02659a25c31f689d1c69809b7e8c16159527a9c7080c9` |
| `v7/` | v7 | 07 | `residual40_v1_results_without_arrays.zip` | 1,686,897 | 368 | `2dac95a535ab03c18495503a7de80e99b44948d62ab83015b71634e8c0c26752` |
| `v8/` | v8 | 08 | `augment40_v1_results.zip` | 1,737,366 | 409 | `00174a890bd74d3209d5efbc3f8115e9dcb2e8dfe0d3e19fda7b0aa6858b3caa` |
| `v9/` | v9 | 09 | `v9_live_pilot_results.zip` | 9,740,597 | 971 | `9205b03a8860006f60aae06184f3ca1d8575c511ba23881fdb971065f5f90522` |
| `v10/` | v10 | 10 | `v10_repair_pilot_results.zip` | 9,373,505 | 1,205 | `0bb5ca73a8543f952f49d7e636ce5a303eda9f7e81851001d0fcdc0dc9cd1ede` |
| `v11/` | v11 | 11 | `v11_pop6_candidate_mechanism_results.zip` | 17,657,885 | 629 | `cd61292532cb59b62408876ee37281700787ba439be01c228b0859509bdb11ca` |
| `v11r2/` | v11r2 | 12 | `v11r2_pop6_candidate_mechanism_results.zip` | 18,309,325 | 632 | `c4e7f22afe8ee4aeea7e4634be66ab1d469762ad1b91fce0d28161197907c843` |
| `v12/` | v12 | 13 | `v12_selfcorrect43_results.zip` | 112,734 | 51 | `245a81787c05b574060785a0f3e4eefd29c22d6882e9a1758b738a19acf0e0b5` |
| `v21/` | v21 | 23 | `v21_alphaenc24_results.zip` | 1,285,161 | 34 | `427cfd80c44f0b8c689c9beb74dd11d899dd26f14a7edab8c0f2edababb20c15` |
| `v22/` | v22 | 24 | `v22_sizes43_results.zip` | 1,856,457 | 53 | `a3cc61578af7f434125e7b44791465780a04dd72cd5f65903880f775d1e40e29` |
| `v23/` | v23 | 25 | `v23_marking12_results.zip` | 761,818 | 23 | `7c9418d76ed25f26901f9003894e4a89fb2cf8e0960a2d1a6215f93bc087c893` |
| `v25/` | v25 | 27 | `v25_trialtours21_results.zip` | 8,687,144 | 84 | `0af69c8a30daa808db7d82f9a69fc58103c3a4ed4ae71055a98316e293d2d327` |
| `v26/` | v26 | 28 | `v26_recomb38_results.zip` | 9,420,519 | 126 | `f6c9d0447d8e10dacb66f66df9024c0c0937380fa1710508b7eac0d5ec69573c` |
| `v27/` | v27 | 29 | `v27_union14_results.zip` | 4,770,815 | 38 | `12f350bebee8b8bd195d7743b7a12f161cc349b2f68b7ac9c11ee710a8fa7205` |

**Run v7.** The complete result ZIP (`residual40_v1_results.zip`, 255,303,068 bytes, SHA-256 `ace84bc19efdfecad01ebac4cb9a778b38f5f463572ff8e15bfe1014587bbc62`) exceeds GitHub's file-size limit because it holds 40 candidate arrays and 40 first-phase arrays (`.npz`, 254 MB). The file here is that archive with the 80 `.npz` files removed and every other entry unchanged; it contains all instance records and summaries. The complete archive is available from the corresponding author.

The summary tables use Turkish column and comparison names in runs v5 and v6 (`eşit-5 POP` equal-five POP restart, `tur sıklığı` tour frequency, `kesintisiz POP` uninterrupted POPMUSIC, `uyarlamalı` adaptive, `sabit` fixed, `birincil` primary). `python analysis/v11_mechanism_pilot.py <zip>` recomputes the run-v11 and run-v11r2 statistics from their ZIPs, and `python analysis/v12_self_correction.py <zip>` the run-v12 statistics (Table 10). The reports and the comparison names of runs v21 to v23 and v25 to v27 are in Turkish; `python analysis/v21_file_encoding.py <zip>`, `python analysis/v22_other_sizes.py <zip>`, `python analysis/v23_restart_marking.py <zip>`, `python analysis/v25_trial_tour_recombination.py <zip>`, `python analysis/v26_recombination_confirmation.py <zip>` and `python analysis/v27_union_exact.py <zip>` recompute their comparisons and print them in English.
