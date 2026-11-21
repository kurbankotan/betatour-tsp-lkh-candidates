# Run v28 (pilot): staged merging of the tours at the cut

Run v26 (`../v26_recombination_confirmation/README.md`) found that at 10,000 nodes GA-EAX gives shorter tours from the
102 tours of the six LKH-3 processes (the best tour and the last 16 trial tours of each) than from their six best
tours, but that at 20,000 and 50,000 nodes GA-EAX does not converge on the 102 tours in the time left after the cut.
Run v27 (`../v27_union_exact/README.md`) found that neither exact merging on the union graph of the six best tours nor
edges ranked by BetaTour shortened the GA-EAX merge. This pilot asks whether merging the 102 tours in two stages (six
groups at the same time, then the group results with the six best tours) gives shorter tours at 20,000 nodes than the
GA-EAX arm of run v26 and than LKH-3's own merging. The protocol is `../PROTOCOL_v28.md`; it states the evidence of two
technical checks before the run (exact repair of small windows gained far less than merging; staged merging matched
GA-EAX on the 102 tours on one instance of 20,000 nodes in less than half of the time) and the rule by which the run
decides whether a confirmation run is prepared. The notebook refuses to start unless that file and the notebook
itself are on the main branch on GitHub as built.

## Design

- 14 new instances: 12 of 20,000 nodes (6 uniform, 6 clustered; 240 s) for the readings and 2 of 50,000 nodes (1 + 1;
  600 s, descriptive).
- One recorded run per instance, as in runs v26 and v27: six POPMUSIC-5 LKH-3.0.14 processes (seeds 201-206) for the
  whole budget (`lkh6`). All other arms start from its tours at half of the budget.
- `trials_50`, `best6_50`, `dual_50`: GA-EAX on the 102 tours and on the six best tours, two processes at the same
  time, and the shorter result (run v26). `lkhmerge_50`: LKH-3's own merging of the six best tours (run v26).
- `staged_50`: stage 1 deals the 96 tours other than the six best tours in turn to six groups and runs six GA-EAX
  processes at the same time until 60% of the time left; stage 2 merges the six group results and the six best tours
  with one GA-EAX process until the budget.
- Readings (12 instances of 20,000 nodes, exact sign test, no correction): R1 `staged_50` / `dual_50`, R2 `staged_50`
  / `lkhmerge_50`. A confirmation run follows only if both hold.

## Files

- `cells/`: the sources of the notebook cells (configuration, set-up with the check of the protocol on GitHub,
  instances, the arms, self-check, run loop, analysis, archive). The recorded run, GA-EAX and LKH-3 merging use the
  code of run v26 unchanged (`../v26_recombination_confirmation/byprod.py`, embedded as `byprod_v26.py`); the staged
  arm is in `cells/c5_experiment.py` (`staged_groups`, `staged_merge`, `make_staged`), and the self-check runs it on a
  small instance.
- `build_colab_notebook.py`: writes `notebooks/30_BetaTour_v28_staged_merging_pilot.ipynb`, embeds the module and
  writes the SHA-256 of the protocol, of the module and of the build into the first cell.
- `tests/run_notebook_locally.py`: runs the notebook's code cells outside Colab in test mode (two processes, three
  small instances, two groups) and imitates a lost session between or inside instances.

```
BETATOUR_V28_TEST=1 BETATOUR_V28_REPO=. BETATOUR_V28_LKH=/path/to/LKH-3.0.14/LKH \
    BETATOUR_V28_EAX_SRC=/path/to/GA-EAX BETATOUR_V28_WORK=/tmp/v28_test \
    python experiments/v28_staged_merging/tests/run_notebook_locally.py
```

## Running it

Commit and push the protocol and the notebook first. Then open the notebook in Colab with an L4 GPU and high RAM
(12 vCPUs, 53 GB; the GPU is not used) and run all cells. Estimated time: set-up (two builds of LKH-3 and four of
GA-EAX) 5-7 min; self-check 2-3 min; 12 instances of 20,000 nodes at about 11.5 min each, 2.3 h; 2 of 50,000 nodes at
about 28 min each, 1 h; analysis and archive 1 min; about 3.4 h in total. Completed instances are kept on Drive
(`MyDrive/BetaTour/v28_staged14`) and skipped when the notebook is run again; an instance interrupted by a change of
machine runs again from the start. `MAX_NEW_INSTANCES` in the first cell splits the run into sessions.
