# Run v27 (pilot): exact merging on the union graph of the tours

Run v26 (`../v26_recombination_confirmation/README.md`) found that merging the tours of six LKH-3 processes at half of
the wall-clock budget gives shorter tours than letting the processes run, and that LKH-3's own merging captures most
of the gain of GA-EAX. This pilot asks two questions about the merging step. Is the shortest tour of the union graph
of the six tours (every edge used by at least one of them), computed exactly, shorter than the tour GA-EAX finds? And
does adding the BetaTour model's most probable edges outside the union shorten that exact merge? The protocol is
`../PROTOCOL_v27.md`; it states the evidence of a technical check before the run (GA-EAX reached the exact optimum of
the union graph on one instance of 3,000 nodes and went slightly beyond it on one of 10,000 nodes; the model's edges
were not used) and the rule by which the run decides whether a confirmation run is prepared. The notebook refuses to
start unless that file and the notebook itself are on the main branch on GitHub as built.

## Design

- 14 new instances: 12 of 10,000 nodes (6 uniform, 6 clustered; 120 s) for the readings and 2 of 20,000 nodes (1 + 1;
  240 s, descriptive).
- One recorded run per instance, as in run v26: six POPMUSIC-5 LKH-3.0.14 processes (seeds 201-206) for the whole
  budget (`lkh6`). All other arms start from its tours at half of the budget.
- `trials_50`, `best6_50`, `dual_50`: GA-EAX on 102 tours and on the six best tours, two processes at the same time,
  and the shorter result (run v26). `lkhmerge_50`: LKH-3's own merging of the six best tours (run v26).
- `exact6_50`: the shortest tour of the union graph of the six best tours, by branch and cut with SCIP 10.0;
  `exact6beta_50`: the same graph plus the 2% x n most probable BetaTour edges outside the union (the model reads the
  six first-trial tours); `exact6near_50`: plus as many shortest edges of the same model graph (control). One core
  each, the three at the same time, each with the rest of the budget minus its own preparation.
- Readings (12 instances of 10,000 nodes, exact sign test, no correction): R1 `exact6_50` / `best6_50`, R2
  `exact6_50` / `dual_50`, R3 `exact6beta_50` / `exact6_50`, R4 `exact6beta_50` / `exact6near_50`. A confirmation run
  follows only if R1 or R2 holds; the model's edges are taken further only if R3 and R4 hold.

## Files

- `unionx.py`: the union graph, the reduction (a node with two remaining edges uses both; a node with two fixed edges
  uses no other; an edge that would close a fixed path into a short cycle is removed), the branch and cut with SCIP
  (degree equations and subtour cuts from the connected components of LP and integral solutions, on the contracted
  fixed paths), the choice of extra edges and a command line that solves one graph in its own process with a
  wall-clock deadline. NumPy and PySCIPOpt.
- `cells/`: the sources of the notebook cells (configuration, set-up with the check of the protocol on GitHub,
  instances, the arms, self-check, run loop, analysis, archive). The recorded run, GA-EAX and LKH-3 merging use the
  code of run v26 unchanged (`../v26_recombination_confirmation/byprod.py`, embedded as `byprod_v26.py`).
- `build_colab_notebook.py`: writes `notebooks/29_BetaTour_v27_union_exact_pilot.ipynb`, embeds both modules and
  writes the SHA-256 of the protocol, of both modules and of the build into the first cell.
- `tests/test_unionx.py`: tests of `unionx.py` (the reduction and the exact solve against enumeration on small random
  graphs; a union of six 2-opt tours of 400 nodes; the time limit and the command line; the extra edges).
- `tests/run_notebook_locally.py`: runs the notebook's code cells outside Colab in test mode (two processes, three
  small instances, the model on the CPU) and imitates a lost session between or inside instances.

```
python experiments/v27_union_exact/tests/test_unionx.py
BETATOUR_V27_TEST=1 BETATOUR_V27_REPO=. BETATOUR_V27_LKH=/path/to/LKH-3.0.14/LKH \
    BETATOUR_V27_EAX_SRC=/path/to/GA-EAX BETATOUR_V27_WORK=/tmp/v27_test \
    python experiments/v27_union_exact/tests/run_notebook_locally.py
```

## Running it

Commit and push the protocol and the notebook first. Then open the notebook in Colab with an L4 GPU and high RAM
(12 vCPUs, 53 GB) and run all cells. Estimated time: set-up (PySCIPOpt, two builds of LKH-3 and four of GA-EAX, the
model) 6-8 min; self-check 2-3 min; 12 instances of 10,000 nodes at about 5 min each, 1 h; 2 of 20,000 nodes at about
10 min each, 20 min; analysis and archive 1 min; about 1.5 h in total. Completed instances are kept on Drive
(`MyDrive/BetaTour/v27_union14`) and skipped when the notebook is run again; an instance interrupted by a change of
machine runs again from the start. `MAX_NEW_INSTANCES` in the first cell splits the run into sessions.
