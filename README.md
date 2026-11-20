# BetaTour: learning near-optimal TSP candidate edges from LKH-3 tour ensembles

Code, notebooks and results for the manuscript

> K. Kotan, "Learning Near-Optimal TSP Edges from Tour Ensembles: Prediction Quality versus Solver Performance in LKH-3," manuscript submitted for publication, 2026.

BetaTour is a graph neural network with 1.9 million parameters. It reads an ensemble of cheap single-trial LKH-3 tours, in the way protein structure predictors read a multiple sequence alignment, and ranks the candidate edges of every node for LKH-3. The study asks two questions: does such a model rank the edges of near-optimal tours better than LKH's own candidate generators, and do the better lists make LKH-3 faster when every cost (ensemble, model inference, list construction and LKH preprocessing) is charged to the time budget?

## Main results

> **Correction in progress (runs v21 to v23, 19 and 20 November 2026).** In runs v1 to v8, the candidate files that restarted LKH-3 did not encode two fields as LKH-3 does: the alpha-values, on which the restart of every later trial depends, and the father of each node in the minimum 1-tree. Every arm of those runs that gave LKH-3 its lists through such a file was concerned (BetaTour, counting and the restarted POPMUSIC controls). Run v21 measured the effect on 24 new 10,000-node instances with a protocol pushed before the run. With the old file, restarted LKH-3 found longer tours than with a file equivalent to its own on 19 of the 24 instances (adjusted p = 0.013). With corrected files, LKH-3 restarted with the BetaTour lists found shorter tours than uninterrupted POPMUSIC on 18 of the 24 instances (mean 0.006%, adjusted p = 0.023), and on 10 of 12 further 10,000-node instances in run v23 (mean 0.006%). Run v22 repeated the comparison on 43 instances of the other classes (1,000 and 20,000 nodes, TSPLIB): the restarted arm was shorter on 26 instances and longer on 13, which did not reach significance (mean of six seeds; mean 0.006%, adjusted p = 0.107); on the best of six seeds it was shorter on 8 and equal on 3 of the 11 TSPLIB instances and showed no difference at 20,000 nodes (shorter on 8, longer on 8). In runs v21 and v22, lists from tour counting did as well as the BetaTour lists, so the gain is not attributed to the learned model. Run v23 (exploratory) also tried two other rules for the zero alpha-values of the file and found no gain. Run v24 will repeat the 20,000-node comparison with twice the budget; its protocol is `experiments/PROTOCOL_v24.md` and it has not been run yet. The solver comparisons of runs v1 to v8 below were therefore affected, and the statement that the better lists did not make LKH-3 faster no longer holds for 10,000-node instances; the text below and the manuscript have not yet been revised. The coverage results and runs v9 to v12 did not use such files. Details: `experiments/PROTOCOL_v21.md` to `PROTOCOL_v23.md` and the README files of `experiments/v21_alpha_encoding/`, `experiments/v22_other_sizes/` and `experiments/v23_restart_marking/`.

Reference-tour edges missing from the first five candidates of each node, per 1,000 nodes (lower is better; BetaTour and counting read the same eight-tour ensemble):

| Candidate list | Validation | Uniform 1k | Clustered 1k | Uniform 10k | Clustered 10k | TSPLIB |
| --- | --- | --- | --- | --- | --- | --- |
| POPMUSIC | 3.95 | 5.23 | 5.62 | 5.60 | 6.18 | 7.25 |
| α-nearness after ascent | 4.56 | 2.27 | 3.89 | 2.79 | 2.46 | 4.69 |
| Counting, M = 8 | 1.32 | 1.59 | 1.86 | 2.08 | 2.69 | 3.93 |
| **BetaTour, M = 8** | **0.48** | **0.31** | **0.61** | **0.42** | **0.52** | **2.55** |

The better lists did not make LKH-3 faster. Win/tie/loss counts below are by instance and refer to BetaTour.

- With all overhead charged, BetaTour-guided LKH-3 tied ascent-free LKH-3 with POPMUSIC candidates on one core (24/4/23 over 51 instances of 1,000 to 20,000 nodes) and in a six-core multi-start setting (18/15/18). The ascent-free baseline itself beat LKH-3's default configuration on 41 of the 51 instances, so a comparison with the default configuration alone mixes the effect of the candidates with the time saved by skipping the subgradient ascent.
- With matched restarts on the same 51 instances (run v5), BetaTour beat an equal-five POPMUSIC restart (26/18/7) but tied uninterrupted POPMUSIC (18/17/16). Confidence-adaptive allocation did not improve fixed BetaTour on 40 new instances (run v6, 12/14/14).
- On two further sets of 40 new instances, model-ranked edges beat tour-frequency and nearest-neighbour edges within restarts (run v7: 33/0/7; run v8: 28/0/12 and 32/0/8) but lost to uninterrupted POPMUSIC-5 (8/0/32 and 12/0/28) and POPMUSIC-6 (run v8: 9/0/31).
- Three exploratory pilots on reused instances changed running LKH processes instead of restarting them. Live candidate additions (run v9) beat POPMUSIC-5 on six of eight instances but split four to four against POPMUSIC-6. Model-guided external repair of paused tours (run v10) shortened some of them, but the final LKH tour was shorter than the repaired tour in all 48 seeds of every repair arm, and the BetaTour repair arm lost to POPMUSIC-6 on seven of eight instances and tied on one. An instrumented pilot (run v11r2, four reused instances) found that LKH used model-ranked additions more often than equally many frequency or random additions (19.0 against 16.1 and 10.7 entries into the current tour per added edge). The model additions ended shorter than random additions on all four instances and shorter than frequency additions on three, but all three kinds of addition reduced the number of completed trials, and the guided search tied a sham run (2/0/2) and lost narrowly to POPMUSIC-6 (1/0/3). With four instances, none of the run-v11r2 differences was statistically significant.
- A diagnostic run on unmodified LKH-3 (run v12, 43 instances) measured how much of the candidate deficit of POPMUSIC lists LKH-3 repairs by itself. LKH-3 appends the edges of every new best tour to its candidate lists; this added 66% of the reference edges missing from the initial POPMUSIC-5 lists to the candidate set within a quarter of the budget and 78% by the end. At the budget the candidate set still missed more reference edges than the BetaTour lists did from the start (1.36 against 0.88 per 1,000 nodes; 33 of 43 instances, p = 0.0003), but edges missing from the initial lists made up only 4.9% of the reference edges absent from the final tours.

## Runs

| Run | Notebook | Question | Instances | Results |
| --- | --- | --- | --- | --- |
| v1 | `01_BetaTour_v1_data_training_first_tests` | data, BetaTour-kNN training, first LKH tests | 30,000 instances (29,000 for training, 1,000 for validation); small LKH tests | `results/v1/` |
| v2 | `02_BetaTour_v2_training_coverage_single_core` | BetaTour training, coverage, single-core LKH | coverage sets; 51 LKH instances | `results/v2/` |
| v3 | `03_BetaTour_v3_controls_mixed_lists` | controls, mixed lists, smaller ensembles | same | `results/v3/` |
| v4 | `04_BetaTour_v4_multi_core` | six-core multi-start LKH | 51 | `results/v4/` |
| v5 | `05_BetaTour_v5_controlled_follow_up` | matched restart controls | 51 (reused) | `results/v5/` |
| v6 | `06_BetaTour_v6_confidence_adaptive_holdout` | confidence-adaptive candidate allocation | 40 new | `results/v6/` |
| v7 | `07_BetaTour_v7_POPMUSIC_4plus1` | replace the fifth POPMUSIC candidate | 40 new | `results/v7/` |
| v8 | `08_BetaTour_v8_POPMUSIC_5plus1` | add a sixth candidate to POPMUSIC-5 | 40 new | `results/v8/` |
| v9 | `09_BetaTour_v9_live_LKH_pilot` | live candidate interventions without restart | 8 (reused from v8) | `results/v9/` |
| v10 | `10_BetaTour_v10_external_repair_pilot` | external repair of paused tours | 8 (as v9) | `results/v10/` |
| v11 | `11_BetaTour_v11_POP6_candidate_mechanism` | candidate telemetry inside LKH (2 vCPUs, superseded) | 4 (as v9, 10k) | `results/v11/` |
| v11r2 | `12_BetaTour_v11r2_POP6_candidate_mechanism` | v11 with corrected code on 12 vCPUs | 4 (as v11) | `results/v11r2/` |
| v12 | `13_BetaTour_v12_candidate_self_correction` | how much of the POPMUSIC candidate deficit LKH-3 repairs itself | 43 with reference tours (from v2-v5) | `results/v12/` |
| v21 | `23_BetaTour_v21_alpha_encoding_control` | candidate-file encoding of the restarted arms; comparison with corrected files | 24 new (10,000 nodes) | `results/v21/` |
| v22 | `24_BetaTour_v22_corrected_files_other_sizes` | the comparison with corrected files on the other sizes and on TSPLIB | 43 (32 new, 11 TSPLIB) | `results/v22/` |
| v23 | `25_BetaTour_v23_restart_marking` | other rules for the zero alpha-values of the candidate file (exploratory) | 12 new (10,000 nodes) | `results/v23/` |
| v24 | `26_BetaTour_v24_longer_budget_20k` | the comparison with corrected files at 20,000 nodes with twice the budget | 20 new (20,000 nodes) | not run yet |

Runs v1, v2, v4, v8, v12, v21 and v23 used an NVIDIA L4 GPU and runs v3, v5 to v7, v9, v10, v11r2 and v22 an NVIDIA A100, all with 12 visible vCPUs; run v11 ran on a Tesla T4 with two vCPUs, below its stated requirement (Section 4.13 of the paper). Run numbers follow the laboratory record; numbers 13 to 20 belong to a planned hard-instance run and to exploratory solver pilots that are not part of the paper and are not in this repository.

## Repository layout

```
notebooks/    the Colab notebooks of the runs; notebooks 01 to 13 as they were run (with their outputs), notebooks 23
              to 26 (runs v21 to v24) without outputs (their reports are in the result ZIPs)
src/          tsplab.py (instances, LKH-3 control, tour ensembles, coverage) and tsp_model.py (the BetaTour model)
experiments/  code of the later runs: LKH-3 hooks and their builds (livehook/, v10_repair/, v11_candidate/), live-pilot and
              policy code (v9_live/, policy/), the run-v11r2 pipeline, the run-v12 diagnosis (v12_self_correction/),
              the runs with corrected candidate files (v21_alpha_encoding/, v22_other_sizes/, v23_restart_marking/,
              v24_longer_budget/), their tests and the protocols of runs v12 and v21 to v24 (PROTOCOL_v12_v13.md,
              PROTOCOL_v21.md to PROTOCOL_v24.md)
analysis/     scripts that recompute the statistics of the paper from results/
results/      outputs of runs v1-v4 (reports, tables, traces) and the result ZIPs of runs v5-v12 and v21-v23 (see
              results/README.md)
figures/      the five figures of the paper
checkpoints/  the trained BetaTour checkpoint used in runs v2 to v12 (see checkpoints/README.md)
```

Each notebook from run v9 on embeds the exact experiment code it ran, and its result ZIP stores a copy under `code/`; `experiments/` holds the latest version of each module.

## Naming note

BetaTour was called TourFold during development. Notebook titles, code identifiers (for example `TourFoldNet`), log messages, result file names and the Google Drive folder (`MyDrive/TourFold_TSP`) keep that name, so that the notebooks run unchanged and the files match the logs. In run v1, `TourFold` denotes BetaTour-kNN; from run v2 on, `TourFold-v2` denotes BetaTour. The explanations inside the notebooks, the log messages and the automatic reports (`RAPOR*.md`) are in Turkish; `results/README.md` describes the files in English.

## Reproducing the experiments

All notebooks run on Google Colab with a GPU and 12 virtual CPUs. Open a notebook in Colab, choose such a runtime and run all cells.

**Runs v1 to v4.** Each notebook

1. downloads and compiles LKH-3.0.14 from its author's website and downloads TSPLIB when it is not yet on Drive,
2. stores every intermediate result on Google Drive in parts and resumes from the last completed part after an interrupted session,
3. writes a report, the result tables and the figures, and saves them as a zip archive.

Run the notebooks in order: runs v2 to v4 read the data written by run v1, and runs v3 and v4 read the BetaTour checkpoint trained in run v2.

Estimated run time per stage (minutes, summed over the Colab sessions of the original runs):

| Notebook | Stage | Time (min) |
| --- | --- | --- |
| 01, run v1 (L4) | reference tours and 16-tour ensembles of the evaluation sets | 326 |
| | training data: 30,000 instances with reference tours and ensembles | 778 |
| | training of BetaTour-kNN | 108 |
| | training of the single-instance model | 67 |
| | first LKH tests | 48 |
| | other stages (LKH build, self-tests, classical candidate sets, coverage) | about 3 |
| 02, run v2 (L4) | training of BetaTour | 105 |
| | coverage study | 2 |
| | single-core LKH experiment | 129 |
| | other stages (loading data, self-tests, earlier models) | about 4 |
| 03, run v3 (A100) | coverage study | 1 |
| | single-core LKH experiment | 134 |
| | other stages | about 3 |
| 04, run v4 (L4) | instances, model check and analysis | about 3 |
| | multi-core LKH experiment | 219 |
| **Total** | | **about 32 h** |

**Runs v5 to v12 and v21 to v23.** These notebooks fix this repository at commit `d63f7e17942a6af5cba14ad010edca1069d477b3` and the checkpoint by its SHA-256, regenerate their instances from fixed seeds, store every completed instance or arm under `MyDrive/BetaTour/<experiment>` and skip completed work when they are run again. A manifest freezes the settings and code hashes; a changed setting needs a new experiment name. Arms run one after another with six parallel LKH processes, and each arm uses the wall-clock budget of its instance, so a run takes roughly the number of instances times the number of arms times the budget, plus set-up and archiving. For example, run v11r2 took about 3-5 min for set-up and the LKH builds, 1 min for its warm-up checks, 48 min for 4 instances × 6 arms × 120 s and 2-4 min for archiving; run v12 took 3-5 min for set-up, 1 min for its warm-up, 75 min for the 43 instances (68 min of solver budgets plus first trials, model and analysis) and 1-2 min for archiving. Runs v9 to v11r2 patch LKH-3.0.14; they download it from its author's website or, if that fails, from a pinned copy, and check the archive and the patched source files against recorded SHA-256 values.

The model initialization was not seeded and deterministic GPU operations were not enforced, so a retrained model will not be identical to the checkpoint used in the paper; that checkpoint is `checkpoints/betatour_v2.pt`. All seeds, LKH parameters and model settings are listed in Appendix A of the paper.

## Re-running the analyses

The analysis scripts need neither a GPU nor LKH:

```
pip install -r requirements.txt
python analysis/v3_single_core_paired_tests.py
python analysis/v3_free_ensemble_repeatability_spearman.py
python analysis/v3_additional_pairs_by_size_class.py
python analysis/v2_v3_additional_configurations.py
python analysis/v4_single_runs_and_best_of_six.py
python analysis/v4_size_split_and_spread.py
python analysis/v4_time_profiles_and_resume.py
python analysis/v4_per_instance_table.py
python analysis/v11_mechanism_pilot.py results/v11r2/v11r2_pop6_candidate_mechanism_results.zip
python analysis/v12_self_correction.py results/v12/v12_selfcorrect43_results.zip results/v5/paper51_v1_results.zip
python analysis/v21_file_encoding.py results/v21/v21_alphaenc24_results.zip results/v5/paper51_v1_results.zip
python analysis/v22_other_sizes.py results/v22/v22_sizes43_results.zip
python analysis/v23_restart_marking.py results/v23/v23_marking12_results.zip results/v21/v21_alphaenc24_results.zip
```

The run-v4 and earlier scripts read `results/` and print the statistics they compute. The run-v11 script recomputes Tables 8 and 9 of the paper from the run-v11r2 result ZIP (or the run-v11 figures of Appendix A.12 from `results/v11/`), the run-v12 script recomputes Table 10 and the checks against run v5, the run-v21 script recomputes the comparisons of that run and the corresponding ones of run v5, and the run-v22 and run-v23 scripts recompute the comparisons of those runs (the run-v23 script also prints the comparisons it shares with run v21). For runs v5 to v10, the `summary/` tables inside the result ZIPs are the tables quoted in the paper. The tests of the run-v11r2 and run-v12 code run without LKH (`experiments/v11_candidate/README.md`, `experiments/v12_self_correction/README.md`).

## Data

- **Synthetic instances** are not stored here. They are regenerated deterministically from the seeds in Appendix A.2 of the paper (training instance i: seed 1,000,000 + i; evaluation sets: seeds 101, 202, 303, 404, 505 and 606; runs v6 to v8: seeds 2101-2606, 3103-3406 and 4107-4410; run v21: seeds 5111 and 5212; run v22: seeds 6101, 6202, 6505 and 6606; run v23: seeds 7111 and 7212; run v24: seeds 8505 and 8606).
- **TSPLIB** instances are not redistributed. The notebooks download them from the TSPLIB website: https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
- **LKH-3** is not redistributed. The notebooks download it from its author's website: http://webhotel4.ruc.dk/~keld/research/LKH-3/
- `results/v4/results/instances/` holds, for each of the 51 instances of the multi-core experiment, the improvement traces of all twelve LKH-3 runs.
- The run-v7 result ZIP here omits its candidate and first-phase arrays (254 MB, beyond GitHub's file-size limit). The complete run-v7 archive and the audit ZIPs of runs v8 to v12, with raw tours, candidate snapshots and stored model outputs, are available from the author.

## Trained model

`checkpoints/betatour_v2.pt` holds the trained BetaTour model of runs v2 to v12; `checkpoints/README.md` shows how to load it.

## Citation

Please cite the paper; `CITATION.cff` describes this repository. Until the paper is published:

```
@unpublished{kotan2026betatour,
  author = {Kotan, Kurban},
  title  = {Learning Near-Optimal {TSP} Edges from Tour Ensembles: Prediction Quality versus Solver Performance in {LKH-3}},
  note   = {Manuscript submitted for publication},
  year   = {2026}
}
```

## License

The code (notebooks, `src/`, `experiments/` and `analysis/`) is released under the MIT License (see `LICENSE`). The results and figures are released under CC BY 4.0. LKH-3 and TSPLIB keep their own terms.

## Contact

Kurban KOTAN, Department of Artificial Intelligence and Machine Learning, Manisa Celal Bayar University, kurban.kotan@cbu.edu.tr, ORCID [0000-0002-6660-0565](https://orcid.org/0000-0002-6660-0565)
