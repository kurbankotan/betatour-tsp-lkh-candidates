# BetaTour v11 and v11r2: residual candidate search pilot

**Status:** completed. Run v11 ran on an under-provisioned runtime (Tesla T4, 2 vCPUs) and is superseded for its endpoint by run v11r2, which used the corrected code in this folder on an A100 with 12 vCPUs (Sections 4.13, 5.8 and A.12 of the paper). Neither run supports a claim that BetaTour beats POPMUSIC, LKH or any other solver.

## Why this experiment

The v9 live-search pilot added one BetaTour candidate to selected POPMUSIC-5 rows. Across eight previously examined instances, `beta_sparse` beat native POPMUSIC-5 in 6/8 (mean relative tour-cost difference −0.00304%), but it did not beat the stronger POPMUSIC-6 control (4/8, mean +0.00266%; positive means worse). It also did not establish an advantage over the tour-frequency control. Those numbers are exploratory development results. Extra candidate capacity may explain some of the POPMUSIC-5 difference.

The v10 pilot tested a different, external tour-repair mechanism. Its repairs shortened 43/48 barrier tours in the BetaTour arm, yet the resumed main LKH final tour beat every repaired tour, so none of the repairs affected the hybrid endpoint. The repair callback also paused the main search for about 22 seconds. That negative result does not directly test whether a learned candidate helps *inside* LKH search.

This pilot returns to the v9 search intervention, starts from a stronger POPMUSIC-6 candidate set, and asks two narrower questions:

1. Are model-proposed residual edges enumerated by LKH's search, and do any enter a newly improved incumbent tour?
2. At the same end-to-end wall budget, is their effect better than simply using POPMUSIC-6/7 or choosing equally many edges by frequency or a deterministic random control?

The frozen TourFold/BetaTour checkpoint predicts reference-tour edge likelihood, not expected marginal LKH search gain. A utility-trained selector is a later, conditional experiment.

## Arms and controls

| Arm | Role |
| --- | --- |
| Native POPMUSIC-6 | Unmodified practical benchmark. |
| Native POPMUSIC-7 | Stronger candidate-capacity benchmark. |
| Patched POPMUSIC-6 sham | Same first-trial barrier and model inference, with no candidate action. |
| Patched POPMUSIC-6 + BetaTour | Add a sparse model-ranked residual edge at selected source nodes. |
| Patched POPMUSIC-6 + tour frequency | Same source-node gate and action count; choose the residual edge by ensemble frequency. |
| Patched POPMUSIC-6 + random | Same source-node gate and action count; choose a deterministic random eligible residual edge. |

One policy call computes the gate and the destinations of all three guided arms, so they share the same source nodes, directed action count and unique undirected edge count (asserted). No native candidate is removed. All selectors draw from the same 16-neighbour model graph and exclude edges already in the seed's POP6 graph or first tour. The frequency control is conditional on that gate and graph; in runs v11 and v11r2, 98.5% of its selected edges had zero ensemble frequency, so it acts mainly as a nearest-novel-neighbour control.

## Protocol

1. Pinned LKH 3.0.14 source (FindTour.c and the 11 patched files are hash-checked), checkpoint, code and binary hashes. Six fixed LKH seeds per instance under one monotonic wall deadline that includes start-up, preprocessing, GPU inference, synchronisation and search.
2. Guided actions are applied at a completed first-trial barrier. Pre-action tours, candidate states, model decision inputs, gates and capacities are hashed per seed; any mismatch excludes the instance from same-state policy comparisons. Native POPMUSIC-6/7 remain descriptive practical benchmarks.
3. Each action is inserted at slot 3 with α = 500. **LKH adjusts candidate rows after every better tour (AdjustCandidateSet, called from FindTour), so in run v11 slot 3 was guaranteed only until the first improvement after the barrier.** From v11r2 the hook enforces slot 3 as a floor at every completed-trial boundary: an action edge that LKH moved behind slot 3 is moved back, while one that LKH moved ahead stays there. The hook counts restorations, and the build records which reordering rule the pinned upstream AdjustCandidateSet.c applies; in run v11r2 it recorded that AdjustCandidateSet moves edges shared by the two most recent best tours to the front of each list and does not sort by α (ResetCandidateSet sorts by α when the run ends). LKH moved an added edge behind slot 3 about 0.9 times per action during a run.
4. Telemetry: candidate enumerations in nine instrumented move loops, passage through the first top-level adjacency/G1 gate (not full gain-filter passage), entries into locally accepted LK tours through StoreTour (merge-introduced edges are not counted), and appearances in globally improved incumbents. Incumbent entries are observational and do not show that an action caused an improvement. The small aggregate file is written at every completed trial and better tour for every patched arm; per-action rows at most every five seconds.
5. GPU inference was not bitwise reproducible across separately launched arms in run v9. From v11r2, every patched arm runs and is charged for its own inference, but arms whose model input (coordinates, k and the six first tours) is bitwise identical decide from the first stored output. The audit records the decision basis, both probability hashes and their maximum difference. (On the T4 of run v11 the outputs happened to match exactly.)
6. Native arms: LKH rewrites OUTPUT_TOUR_FILE in place. From v11r2 the poll only copies completed rewrites inside the deadline; tours are parsed and costs recomputed afterwards. If a termination cuts a rewrite, the last complete tour seen is the deadline result and the case is flagged. Native timestamps are approximate; patched `best.events` are exact. Cross-arm time-to-target curves are exploratory.
7. Four previously examined **10,000-node** development instances (two uniform, two clustered), 120 s per arm. The instance, not the seed, is the unit of analysis. Primary endpoint: the shortest valid EUC_2D tour among six seeds at the deadline, with exact two-sided sign tests reported as descriptive. Mean-of-six and seed-level summaries, per-action mechanism rates and the trial-count ratio against sham were added after run v11 and are descriptive.
8. Freeze one policy, its gate, slot, alpha, candidate budget and statistical plan **before** generating a disjoint holdout. Claiming superior performance requires a robust advantage over native POPMUSIC-6 and the matched controls, with reported uncertainty.

## Run history

**Run v11** (`notebooks/11_BetaTour_v11_POP6_candidate_mechanism.ipynb`, results in `results/v11/`): Tesla T4 GPU with 2 visible vCPUs, below the stated requirement; six LKH processes shared two vCPUs. All 24 arms completed; first tours, candidate states, model outputs and gates matched across the four patched arms on all four instances. The first trial ended 49–76 s into the 120 s budget, leaving 1–35 trials per seed after the action. Best of six seeds, BetaTour additions were worse than native POPMUSIC-6 (0/4, mean +0.033%), sham (0/4, +0.025%) and frequency (0/4, +0.017%), and lost 1/3 to random (+0.010%) and POPMUSIC-7 (+0.007%); exact two-sided sign p = 0.125 for 0/4. Over 24 seed pairs, BetaTour and sham split 12/12 (mean +0.0001%). Native POPMUSIC-7 was worse than POPMUSIC-6 on all four instances. Mechanism: every action was enumerated (≈4,040 times on average); BetaTour edges entered accepted LK tours 1.61 times per action (frequency 1.37, random 0.86) and improved incumbents 5.7 times per 1,000 actions (4.5, 1.4), but the guided arms completed a median of 14–18% fewer trials than sham. Because of the hardware, run v11's endpoint is not comparable with runs v9–v10.

**Run v11r2** (`notebooks/12_BetaTour_v11r2_POP6_candidate_mechanism.ipynb`, results in `results/v11r2/`): same instances, seeds, arms and budget, with the corrections in items 3, 5 and 6, a hard requirement of an A100 or L4 GPU and at least 12 vCPUs, and a warm-up that checks a guided arm and the decision cache before the pilot. It ran on an A100-SXM4-80GB with 12 vCPUs. Best of six, BetaTour additions beat random additions on all four instances and frequency additions on three, tied sham 2/2 and lost to native POPMUSIC-6 on three (mean +0.0007%). BetaTour edges entered the current tour 19.0 times per action (frequency 16.1, random 10.7) and new best tours 11.2 times per 1,000 actions (8.1 and 2.8), but every guided arm completed fewer trials than sham (median 80-89% of the sham count). `python analysis/v11_mechanism_pilot.py results/v11r2/v11r2_pop6_candidate_mechanism_results.zip` recomputes these numbers.

## Decision rule for further work

If proposed edges are seldom enumerated or never enter a better incumbent, investigate the search interface or stop this intervention. If enumeration is frequent but final quality does not improve against strong controls, the next step is a small residual utility head trained on paired short LKH continuations from the same state, labelled by marginal downstream gain per unit of charged compute. A larger holdout and a manuscript performance claim are justified only after the mechanism succeeds on development data and the policy is frozen.

## Tests

* `python test_policy.py`: policy invariants, a loop-based oracle for the vectorised gate, and the decision cache under simulated nondeterministic inference.
* `python hook_logic_test/test_hook_logic.py`: compiles the hook against a mock LKH state with gcc (AddressSanitizer and UBSan) and checks action application, slot-floor restoration after two kinds of row reordering, counters, telemetry and atomic incumbent files. No LKH sources are needed.
* `python fake_lkh/test_pipeline_fake_lkh.py --src REPO/src [--checkpoint REPO/checkpoints/betatour_v2.pt]`: runs the notebook's warm-up, run, analysis and archive cells end to end with a protocol-level fake solver, then checks the truncated-native-file fallback and the patched-event recovery.
* `python test_search_hook.py --source-dir LKH-3.0.14 --binary PATCHED/LKH`: smoke test with the real patched LKH binary.

## Reproduction and archive

The v11r2 entry point is [`notebooks/12_BetaTour_v11r2_POP6_candidate_mechanism.ipynb`](../../notebooks/12_BetaTour_v11r2_POP6_candidate_mechanism.ipynb), the executed notebook. `python experiments/v11_candidate/build_colab_notebook.py` regenerates it without outputs under the name it was generated with, `notebooks/06_BetaTour_v11r2_POP6_candidate_mechanism.ipynb`. The run-v11 notebook, `notebooks/11_BetaTour_v11_POP6_candidate_mechanism.ipynb`, embeds the earlier code of this folder (review snapshot 9442512) exactly as it ran on two vCPUs. Each notebook's frozen manifest, Drive folder, resume checks, source hashes and result/audit ZIPs are part of the run. The full audit ZIP holds raw tours, candidate snapshots and decision caches; the compact results ZIP alone cannot independently verify route validity or source-state equality. Any post-run code change must use a new experiment name rather than continue under a previous manifest. `PRE_RUN_REVIEW_BRIEF.md` is the review brief written before run v11; it records the hypothesis, controls and primary endpoint fixed before that run.
