# Run v29 (confirmation run): staged merging of the tours at the cut

Run v28 (`../v28_staged_merging/README.md`), a pilot, found that merging the 102 tours of the six LKH-3 processes in
two stages (six GA-EAX groups at the same time, then the group results with the six best tours) gave shorter tours at
20,000 nodes than the GA-EAX arm of run v26 (10 of 12 instances) and than LKH-3's own merging (12 of 12). This run
repeats the comparison with the same settings on 18 new synthetic instances of 20,000 and 50,000 nodes, with two
hypotheses and the correction fixed before the run (`../PROTOCOL_v29.md`), and adds five TSPLIB instances with known
optimum as a descriptive check. The notebook refuses to start unless the protocol and the notebook itself are on the
main branch on GitHub as built.

## Design

- 23 instances: 12 new instances of 20,000 nodes (6 uniform, 6 clustered; 240 s), 6 new instances of 50,000 nodes
  (3 + 3; 600 s) and the TSPLIB instances rl11849, usa13509, brd14051, d15112 and d18512 (0.02 s per node,
  descriptive).
- One recorded run per instance and the arms of run v28, unchanged: `lkh6`, `trials_50`, `best6_50`, `dual_50`,
  `lkhmerge_50` and `staged_50`.
- Hypotheses (18 synthetic instances, exact sign test, Holm over the two): H1 `staged_50` / `lkhmerge_50`, H2
  `staged_50` / `dual_50`. Staged merging is recorded as confirmed only if both are supported.

## Files

- `cells/`: the sources of the notebook cells; they are those of run v28 with the instances of this run (including
  the TSPLIB instances, which the set-up downloads as run v26 did), the optimum in each instance record and the
  analysis of the two hypotheses with Holm correction. The recorded run, GA-EAX and LKH-3 merging use the code of run
  v26 unchanged (`../v26_recombination_confirmation/byprod.py`, embedded as `byprod_v26.py`).
- `build_colab_notebook.py`: writes `notebooks/31_BetaTour_v29_staged_confirmation.ipynb`, embeds the module and
  writes the SHA-256 of the protocol, of the module and of the build into the first cell.
- `tests/run_notebook_locally.py`: runs the notebook's code cells outside Colab in test mode (two processes, three
  small synthetic instances, one TSPLIB-format file written locally, two groups) and imitates a lost session between
  or inside instances.

```
BETATOUR_V29_TEST=1 BETATOUR_V29_REPO=. BETATOUR_V29_LKH=/path/to/LKH-3.0.14/LKH \
    BETATOUR_V29_EAX_SRC=/path/to/GA-EAX BETATOUR_V29_WORK=/tmp/v29_test \
    python experiments/v29_staged_confirmation/tests/run_notebook_locally.py
```

## Running it

Commit and push the protocol and the notebook first. Then open the notebook in Colab with an L4 GPU and high RAM
(12 vCPUs, 53 GB; the GPU is not used) and run all cells. Estimated time: set-up (two builds of LKH-3, four of GA-EAX
and the TSPLIB files) 5-7 min; self-check 2-3 min; 12 instances of 20,000 nodes at about 11.5 min each, 2.3 h; 6 of
50,000 nodes at about 28 min each, 2.8 h; 5 TSPLIB instances at 11-17 min each, 1.1 h; analysis and archive 1 min;
about 6.3 h in total. Completed instances are kept on Drive (`MyDrive/BetaTour/v29_staged23`) and skipped when the
notebook is run again; an instance interrupted by a change of machine runs again from the start. With
`MAX_NEW_INSTANCES = 12` in the first cell the run takes two sessions.
