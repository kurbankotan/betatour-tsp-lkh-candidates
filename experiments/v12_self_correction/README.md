# Run v12: does LKH-3 repair the candidate deficit of POPMUSIC lists by itself?

BetaTour's lists miss far fewer near-optimal edges than LKH-3's POPMUSIC lists, yet they did not make LKH-3 faster.
Within a run, LKH-3 appends the edges of every new best tour to its candidate lists (AdjustCandidateSet, called from
FindTour); with one run per process the added edges stay in the lists for the rest of the run. This run measures how much of the POPMUSIC deficit that
mechanism closes during the search. The protocol is in `../PROTOCOL_v12_v13.md` and was written before the run.

## Design

- 43 instances with a reference tour (16 x 1,000, 16 x 10,000 nodes and 11 TSPLIB instances), budgets of Table 1.
- Per instance, six native, ascent-free POPMUSIC-5 LKH-3.0.14 processes (seeds 101-106) under one wall-clock
  deadline; no source patch. Each process writes its initial lists; every rewrite of its best-tour file is copied.
- Six separate single-trial runs with the same seeds form the M = 6 ensemble of the BetaTour and counting top-5 lists.
- Measures per 1,000 nodes: reference edges missing from C0, from the BetaTour and counting lists, and from the dynamic
  set C0 + all best-tour edges by 25, 50, 75 and 100% of the budget; recovery shares; reference edges that only the
  BetaTour lists contain.
- Primary comparison: dynamic set at the budget versus BetaTour top-5, exact sign test over instances.

## Files

- `selfcorrect.py`: LKH runs, polling, edge accounting and summary tables.
- `build_colab_notebook.py`: writes `notebooks/13_BetaTour_v12_candidate_self_correction.ipynb` with the module
  embedded byte for byte.
- `tests/fake_lkh_v12.py`: protocol stand-in for LKH-3 (candidate file, trials, in-place tour rewrites, trace lines).
- `tests/test_selfcorrect.py`: unit tests and the pipeline with the stand-in solver.
- `tests/test_notebook_fake_lkh.py`: runs the notebook's code cells end to end twice (the second run checks resume).

```
python experiments/v12_self_correction/tests/test_selfcorrect.py --src src --checkpoint checkpoints/betatour_v2.pt
python experiments/v12_self_correction/tests/test_notebook_fake_lkh.py --repo . --checkpoint checkpoints/betatour_v2.pt
```

## Running it

Open the notebook in Colab with an A100 or L4 GPU and high RAM (12 vCPUs) and run all cells. Estimated time: set-up and
LKH build 3-5 min, warm-up 1 min, solver about 68 min (16 x 5 s + 16 x 120 s + the TSPLIB budgets), first trials, model
and analysis 10-15 min, archiving 1-2 min; about 1.5 h in total. Completed instances are kept on Drive
(`MyDrive/BetaTour/v12_selfcorrect43`) and skipped when the notebook is run again.

## Result (run on 12 November 2026)

All 43 instances completed in one Colab session (NVIDIA L4, 12 vCPUs). First-trial costs equalled those of run v5 on
all 43 instances, the initial lists of every full run equalled those of its single-trial run, and the best-of-six
lengths equalled run v5's uninterrupted POPMUSIC arm on 34 instances (at most 0.0045% longer on the other nine).

Per 1,000 nodes, the initial POPMUSIC-5 lists missed 5.51 reference edges, the BetaTour top-5 lists 0.88 and counting
2.66. The search added 66% of the missing edges to LKH-3's candidate set by a quarter of the budget and 78% by the end
(1.36 still missing). The prespecified primary comparison went against H1: at the budget the dynamic set missed more
reference edges than the BetaTour lists on 33 instances, fewer on 9 and as many on 1 (exact sign p = 0.0003). Reference
edges missing from the initial lists were in the final tours in 72% of cases, against 95% for edges present in them, but
they made up only 4.9% of the 48.5 reference edges per 1,000 nodes that the final tours lacked. The BetaTour lists held
1.06 reference edges per 1,000 nodes that the search never reached.

`python analysis/v12_self_correction.py results/v12/v12_selfcorrect43_results.zip results/v5/paper51_v1_results.zip`
recomputes Table 10 of the paper and the checks against run v5.
