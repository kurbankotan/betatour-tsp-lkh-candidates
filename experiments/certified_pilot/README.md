# Certified candidate graphs: decision pilot

This folder holds a pilot for a follow-up project. It is **not a run of the BetaTour study**. The manuscript uses its
proven optima and its coverage counts together with run v30 (Sections 4.17 and 5.12, Appendix A.16); its main
question was answered negatively (see the result below).

## Question

A learned model proposes sparse candidate lists. Can a lower-bound certificate prove that the lists are complete?

An edge is eliminated only when every tour that uses it is provably longer than a known tour. The edges that remain
form a graph that contains every optimal tour. A node's candidate list is *certified complete* when it contains all of
the node's remaining edges. The pilot measures how sparse this graph is and how well BetaTour's ranking proposes it,
compared with the α-, POPMUSIC-, counting- and nearest-neighbour lists.

## Certificates

**C1** is computed by `certpilot.py` in exact integer arithmetic. It is the 1-tree bound W with LKH-3's node
penalties (π), together with the α-values of all edges of the complete graph. A tour that uses edge e costs at least
(W + α(e)) / PRECISION.

**C2** comes from Concorde's root LP with cutting planes (`concorde -B -z k`). It consists of the lower bound LB and
the k lowest reduced costs rc of every node. A tour that uses edge e costs at least LB + rc(e) when rc are the reduced
costs of a dual solution of value at least LB. C2 is used only when the following checks pass against the proven
optimal tour (`c2_checks`):

1. LB ≤ OPT.
2. No edge of the optimal tour is eliminated.
3. Dual consistency holds: OPT ≥ LB + Σ over the tour edges of rc + Σ over all edges of max(0, −rc). Reduced costs
   that omit the cut duals violate this in general (see `tests/test_certificates.py`).

C2 is in floating point and is not an exact proof. An exact check of the dual solution is left for a later step.

## Design

**Instances.** There are 20 new instances from the study's generator: 1,000 and 2,000 nodes, uniform and clustered,
five of each (generator seeds 15101, 15202, 15303 and 15404). There are also four TSPLIB instances with known optima:
pr1002, pcb1173, rl1304 and u1817. For these, Concorde's optimum must equal the known value.

**LKH-3.0.14**, unmodified, supplies:

- six first-trial tours with POPMUSIC-5 (seeds 201–206; the ensemble that BetaTour reads, as in the study);
- the α-5 lists and node penalties (subgradient ascent; used by C1);
- one upper-bound (UB) tour (POPMUSIC-5, 0.01 s per node).

**BetaTour** uses the checkpoint `betatour_v2.pt` (SHA-256 2322f66c…) on the graph of 16 nearest neighbours plus
ensemble edges.

**Concorde (co031219 with QSopt)** is built from the official University of Waterloo pages. Each instance gets two
runs:

- the root LP with `-B` and `-z`;
- a full solve that gives the proven optimum (time limit 45 min).

**Self-check.** A 200-node instance first establishes what the `-z` dump contains. With z = n − 1 the dump must cover
the complete graph and pass the checks; otherwise the run is repeated with the complete edge file (`-E`). The
per-node dump (z = 60) is then compared with the complete dump: its thresholds must bound every unlisted reduced cost.
In the experiment, k = 200 per node.

## Decision rule (written before the pilot)

- **G1.** C2 passes its checks on every instance with a proven optimum.
- **G2.** With C2 and UB = the LKH tour, the mean degree of the remaining graph is at most 8 in every synthetic group.
  At most 1% of nodes may be undecided by the per-node dump.
- **G3.** BetaTour's top-5 lists are certified complete at more nodes than both LKH's α-5 lists and the counting top-5
  lists. The margin must be at least 5 percentage points at both sizes (uniform and clustered pooled).

How the outcomes are read:

| Outcome | Next step |
|---|---|
| G1–G3 pass | Build the method: LKH on certified-complete lists, and an exact independent check of the certificate. |
| G1 and G2 pass, G3 fails | A certified sparse graph exists, but BetaTour is not a better proposer than simple rankings. |
| G2 fails | The root LP certificate alone is too weak; stronger proofs are needed, such as the local elimination of Cook, Helsgaun, Hougardy and Schroeder (arXiv:2307.07054). |
| G1 fails | The meaning of the `-z` dump must be resolved first. |

## Running

1. Build the notebook: `python experiments/certified_pilot/build_colab_notebook.py`. This writes
   `notebooks/32_Certified_candidate_graph_pilot.ipynb`.
2. Open the notebook in Colab with an L4 GPU and High RAM (12 vCPUs).
3. Run all cells. The first cell mounts Google Drive, and results go to `MyDrive/BetaTour/certpilot1`. If the session
   drops, run the notebook again; finished steps are skipped.

Estimated time on 12 vCPUs:

| Step | Time |
|---|---|
| Set-up | 5–8 min |
| Self-check | 1–2 min |
| LKH | 2–4 min |
| BetaTour | about 1 min |
| Concorde | 30–60 min |
| Analysis | 2–4 min |

**Outputs** (on Drive):

- `selfcheck/selfcheck.json`;
- `instances/<name>/` with `lkh.npz`, `model.npz`, `concorde_root.json`, `rc_keep.npz`, `concorde_full.json`,
  `opt_tour.npz` and `analysis.json`;
- `logs/`, including Concorde's output and help text;
- `summary/summary.json` and `summary/instances.csv`.

## Tests

```
python experiments/certified_pilot/tests/test_certificates.py --src src
python experiments/certified_pilot/tests/run_notebook_locally.py --repo . --lkh /path/to/LKH-3.0.14/LKH
```

The first test checks C1 against brute force on 8-node instances. It checks the 1-tree, every α-value and the
elimination rule over all tours, with penalties large enough to make transformed costs negative. It also checks C2
with the exact subtour LP of 9-node instances.

The second test runs the whole notebook in its test mode with small instances. Concorde is replaced by
`tests/fake_concorde.py`, a stand-in that solves the subtour LP and the exact problem with HiGHS. It is not Concorde.

## Result (26 November 2026)

The notebook ran once in Colab on a Tesla T4 runtime with 8 vCPUs; the self-check and all 24 instances completed.

**Self-check.** On the 200-node instance, the `-z` dump with z = n − 1 covered the complete graph (19,900 edges, six
decimals) and passed the checks. The per-node dump (z = 60) listed at least 60 edges per node, and its thresholds
bounded every unlisted reduced cost. The experiment therefore used per-node dumps with k = 200. The root LP solved
this instance (gap 0), so its checks were weak; the checks on the 21 instances below were not.

**Proven optima.** Concorde proved the optimum of 21 of the 24 instances. Two uniform 2,000-node instances
(`cp1_uni2k_00`, `cp1_uni2k_02`) and u1817 were not finished within 45 min. On the three solved TSPLIB instances,
Concorde's optimum equals the published value.

Mean solve times and branch-and-bound nodes:

| Group | Solve time | Branch-and-bound nodes |
|---|---|---|
| uniform, 1,000 nodes | 184 s | 32 |
| clustered, 1,000 nodes | 120 s | 25 |
| uniform, 2,000 nodes (three solved) | 2,077 s | 163 |
| clustered, 2,000 nodes | 1,258 s | 143 |

Gaps of the bounds to the optimum (in % of the optimum, by group):

| Bound | Gap |
|---|---|
| Root LP | 0.032–0.059% (TSPLIB 0.007%) |
| 1-tree | 0.75–3.06% |
| LKH tour (upper bound) | 0.003–0.115% |

**Decision rule:**

| Test | Outcome | Values |
|---|---|---|
| G1: C2 passes its checks | passed | All 21 instances with a proven optimum passed; no optimal edge was eliminated; dual slack ≥ 0. |
| G2: remaining graph sparse | not passed | Mean degree with C2 and UB = LKH tour: 10.6, 7.2, 16.1 and 56.2 by synthetic group (limit 8). With the optimal tour as UB: 7.9, 6.9, 9.5 and 10.0. On one clustered 2,000-node instance, the LKH tour was 0.57% above the optimum, so 19% of that group's nodes stayed undecided. |
| G3: BetaTour the better proposer | not passed | Nodes with a certified-complete top-5 list (C2, UB = LKH tour), at 1,000 / 2,000 nodes: BetaTour 14.3% / 2.5%, α-5 17.1% / 3.3%, counting 13.5% / 2.7%. |

The C1 certificate (1-tree) eliminated far too little: 120–1,100 edges remained per node.

**Decision.** By the rule written before the pilot, certified candidate lists built from root-LP certificates are not
pursued: G2 is not met. A learned proposer would not change that, because even a perfect proposer is limited by the
size of the remaining graph.

**By-product: coverage of proven optimal tours (exploratory).** The 21 proven optimal tours have 29,479 edges in
total. These are the edges that each five-candidate list missed (an edge counts as covered when it is in either end's
list):

| Lists | Missed optimal edges | Instances with the whole optimal tour inside the lists |
|---|---|---|
| BetaTour top five | 23 | 11/21 |
| BetaTour top three + POPMUSIC | 51 | 5/21 |
| counting top five | 63 | 8/21 |
| α-5 (LKH-3) | 105 | 0/21 |
| POPMUSIC-5 | 168 | 0/21 |
| nearest five | 598 | 0/21 |

Per instance, BetaTour's top five missed fewer optimal edges than:

| Lists | Fewer / equal / more | Exact sign test (unadjusted) |
|---|---|---|
| counting top five | 13 / 7 / 1 | p = 0.0018 |
| α-5 | 20 / 0 / 1 | p < 0.001 |
| POPMUSIC-5 | 21 / 0 / 0 | p < 0.001 |

These comparisons were not hypotheses of the pilot. They motivated run v30 (`../PROTOCOL_v30.md`), which tested
whether such lists let LKH-3 reach the proven optimum sooner; they did (`../v30_time_to_optimum/README.md`).

The per-instance table that the notebook wrote (`summary/instances.csv`) is in `results/certified_pilot/`, and
`python analysis/certified_pilot_coverage.py results/certified_pilot/instances.csv` recomputes the two coverage
tables above from it.

**Correction (26 November 2026, after run v30).** The mean solve time of the three solved uniform 2,000-node
instances was first given as 2,326 s, which is the mean over all five instances with the two unfinished ones at
their 2,700 s limit. The mean over the three solved instances is 2,077 s.
