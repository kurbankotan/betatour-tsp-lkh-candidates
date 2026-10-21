# BetaTour: learning near-optimal TSP candidate edges from LKH-3 tour ensembles

Code, notebooks and results for the manuscript

> K. Kotan, "Better Candidates, Same Speed: Learning Near-Optimal TSP Edges from Tour Ensembles with BetaTour," manuscript submitted for publication, 2026.

BetaTour is a graph neural network with 1.9 million parameters. It reads an ensemble of cheap single-trial LKH-3 tours, in the way protein structure predictors read a multiple sequence alignment, and ranks the candidate edges of every node for LKH-3. The study asks two questions: does such a model rank the edges of near-optimal tours better than LKH's own candidate generators, and do the better lists make LKH-3 faster when every cost (ensemble, model inference, list construction and LKH preprocessing) is charged to the time budget?

## Main results

Reference-tour edges missing from the first five candidates of each node, per 1,000 nodes (lower is better; BetaTour and counting read the same eight-tour ensemble):

| Candidate list | Validation | Uniform 1k | Clustered 1k | Uniform 10k | Clustered 10k | TSPLIB |
| --- | --- | --- | --- | --- | --- | --- |
| POPMUSIC | 3.95 | 5.23 | 5.62 | 5.60 | 6.18 | 7.25 |
| α-nearness after ascent | 4.56 | 2.27 | 3.89 | 2.79 | 2.46 | 4.69 |
| Counting, M = 8 | 1.32 | 1.59 | 1.86 | 2.08 | 2.69 | 3.93 |
| **BetaTour, M = 8** | **0.48** | **0.31** | **0.61** | **0.42** | **0.52** | **2.55** |

- The better lists did not make LKH-3 faster. With all overhead charged to the budget, BetaTour-guided LKH-3 tied ascent-free LKH-3 with POPMUSIC candidates on one core (24 wins, 4 ties, 23 losses over 51 instances of 1,000 to 20,000 nodes) and in a six-core multi-start setting (18/15/18).
- The ascent-free baseline itself beat LKH-3's default configuration on 41 of the 51 instances at these budgets, so a comparison with the default configuration alone mixes the effect of the candidates with the time saved by skipping the subgradient ascent.

## Repository layout

```
notebooks/    the four Google Colab notebooks (runs v1 to v4), exactly as they were run
src/          tsplab.py (instances, LKH-3 control, tour ensembles, coverage) and tsp_model.py (the BetaTour model)
analysis/     scripts that recompute the statistics of the paper from results/
results/      outputs of the four runs: reports, tables, LKH improvement traces, figures, logs and configurations
figures/      the four figures of the paper
checkpoints/  place for the trained BetaTour weights (see checkpoints/README.md)
```

## Naming note

BetaTour was called TourFold during development. Notebook titles, code identifiers (for example `TourFoldNet`), log messages, result file names and the Google Drive folder (`MyDrive/TourFold_TSP`) keep that name, so that the notebooks run unchanged and the files match the logs. In run v1, `TourFold` denotes BetaTour-kNN; from run v2 on, `TourFold-v2` denotes BetaTour. The explanations inside the notebooks, the log messages and the automatic reports (`RAPOR*.md`) are in Turkish; `results/README.md` describes the files in English.

## Reproducing the experiments

The four notebooks run on Google Colab with 12 virtual CPUs and a GPU (runs v1, v2 and v4 used an NVIDIA L4, run v3 an NVIDIA A100). Open a notebook in Colab, choose a runtime with 12 virtual CPUs and run all cells. Each notebook

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

The model initialization was not seeded and deterministic GPU operations were not enforced, so a retrained model will not be identical to the checkpoint used in the paper. All seeds, LKH parameters and model settings are listed in Appendix A of the paper.

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
```

Each script reads `results/` and prints the statistics it computes.

## Data

- **Synthetic instances** are not stored here. They are regenerated deterministically from the seeds in Appendix A.2 of the paper (training instance i: seed 1,000,000 + i; evaluation sets: seeds 101, 202, 303, 404, 505 and 606).
- **TSPLIB** instances are not redistributed. The notebooks download them from the TSPLIB website: https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
- **LKH-3** is not redistributed. The notebooks download it from its author's website: http://webhotel4.ruc.dk/~keld/research/LKH-3/
- `results/v4/results/instances/` holds, for each of the 51 instances of the multi-core experiment, the improvement traces of all twelve LKH-3 runs.

## Trained model

See `checkpoints/README.md`.

## Citation

Please cite the paper; `CITATION.cff` describes this repository. Until the paper is published:

```
@unpublished{kotan2026betatour,
  author = {Kotan, Kurban},
  title  = {Better Candidates, Same Speed: Learning Near-Optimal {TSP} Edges from Tour Ensembles with {BetaTour}},
  note   = {Manuscript submitted for publication},
  year   = {2026}
}
```

## License

The code (notebooks, `src/` and `analysis/`) is released under the MIT License (see `LICENSE`). The results and figures are released under CC BY 4.0. LKH-3 and TSPLIB keep their own terms.

## Contact

Kurban KOTAN, Department of Artificial Intelligence and Machine Learning, Manisa Celal Bayar University, kurban.kotan@cbu.edu.tr, ORCID [0000-0002-6660-0565](https://orcid.org/0000-0002-6660-0565)
