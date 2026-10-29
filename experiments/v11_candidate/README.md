# BetaTour v11: residual candidate search pilot

**Status:** development experiment, before Colab results. The v11 code and this protocol make no claim that BetaTour beats POPMUSIC, LKH, or any other solver. The manuscript's performance claims must remain unchanged until a frozen policy passes an independent evaluation.

## Why this experiment

The v9 live-search pilot added one BetaTour candidate to selected POPMUSIC-5 rows. Across eight previously examined instances, `beta_sparse` beat native POPMUSIC-5 in 6/8 (mean relative tour-cost difference −0.00304%), but it did not beat the stronger POPMUSIC-6 control (4/8, mean +0.00266%; positive means worse). It also did not establish an advantage over the tour-frequency control. Those numbers are exploratory development results. Extra candidate capacity may explain some of the POPMUSIC-5 difference.

The v10 pilot tested a different, external tour-repair mechanism. Its repairs shortened 43/48 barrier tours in the BetaTour arm, yet the resumed main LKH final tour beat every repaired tour, so none of the repairs affected the hybrid endpoint. The repair callback also paused the main search for about 22 seconds. That negative result does not directly test whether a learned candidate helps *inside* LKH search.

This pilot returns to the v9 search intervention, starts from a stronger POPMUSIC-6 candidate set, and asks two narrower questions:

1. Are model-proposed residual edges enumerated by LKH's search, and do any enter a newly improved incumbent tour?
2. At the same end-to-end wall budget, is their effect better than simply using POPMUSIC-6/7 or choosing equally many edges by frequency or a deterministic random control?

The frozen TourFold/BetaTour checkpoint predicts reference-tour edge likelihood, not expected marginal LKH search gain. A utility-trained selector is a later, conditional experiment if the instrumentation and matched-state pilot reveal a useful mechanism.

## Planned arms and controls

| Arm | Role |
| --- | --- |
| Native POPMUSIC-6 | Unmodified practical benchmark. |
| Native POPMUSIC-7 | Stronger candidate-capacity benchmark. |
| Patched POPMUSIC-6 sham | Same first-trial barrier and model inference, with no candidate action. |
| Patched POPMUSIC-6 + BetaTour | Add a sparse model-ranked residual edge at selected source nodes. |
| Patched POPMUSIC-6 + tour frequency | Same source-node gate and action count; choose the residual edge by ensemble frequency. |
| Patched POPMUSIC-6 + random | Same source-node gate and action count; choose a deterministic random eligible residual edge. |

The three guided arms must report both directed actions and *unique undirected* inserted edges. An action-count match alone does not prove an equal graph perturbation; a result table must make both counts visible. None of these arms may remove a native candidate just to make its intervention look stronger. POPMUSIC-7 remains an external capacity check rather than a same-state counterfactual.

All three selectors draw from the same 16-neighbor model graph and use the same BetaTour-defined source-node gate. They exclude edges already in the seed's POP6 graph or first tour. The frequency control is therefore conditional on that gate and graph; its selected-edge frequency distribution is exported. If most selected frequencies are zero, interpret that arm as a distance-tie-break control rather than a strong tour-frequency baseline.

## Protocol gates

1. Use pinned LKH 3.0.14 source, checkpoint, code and binary hashes. Run six fixed LKH seeds per instance under one monotonic wall deadline that includes preprocessing, GPU inference, synchronization and search.
2. Apply guided actions only at a fixed completed-trial barrier. For each instance and seed, compare the pre-action tour and candidate state across patched arms. Mark mismatches explicitly and exclude them from same-state policy comparisons. The native POPMUSIC-6/7 arms remain descriptive practical benchmarks.
3. Audit every action against the native and post-action candidate sets. The v11 hook counts candidate enumerations, passage through the first top-level adjacency/G1 gate, accepted-tour edge entries, and appearances in globally improved incumbent tours. The gate count is before the breadth cutoff and is not full gain-filter passage. The hook does not count every gain filter or attribute an improvement causally to an action edge. Aggregate and per-edge telemetry is flushed at completed trials and periodically during long trials, after at least five seconds and a 65,536-enumeration check; the result reports snapshot age at termination so an interrupted final interval can be judged.
4. Use the four already seen **10,000-node** development instances (two uniform, two clustered) to check the mechanism. This first pilot cannot support a 20,000-node conclusion. Analyze one instance, not one seed, as an independent unit. The primary endpoint is the best valid EUC_2D tour across six seeds at the declared deadline; record per-seed costs. Patched arms emit exact hook-level `best.events` incumbent records. Native POPMUSIC-6/7 record `native_improvement_events` by polling the modification time of LKH's output-tour file. These native wall timestamps are approximate and may miss rapidly overwritten incumbents, so cross-arm time-to-target curves are exploratory only; a precise speed claim needs comparable tracing and overhead validation.
5. Freeze one policy, its gate, slot, alpha, candidate budget and statistical plan **before** generating a disjoint holdout. Do not pick the best policy on the holdout. Claiming superior performance requires a robust advantage over native POPMUSIC-6 and the matched heuristic controls, with reported uncertainty.

## Decision rule for further work

If proposed edges are seldom enumerated or never enter a better incumbent, investigate the search interface or stop this intervention. Incumbent entry alone does not prove that the edge caused the improvement. If enumeration is frequent but final quality does not improve against strong controls, train a small residual utility head on paired short LKH continuations from the same state; use the exploratory timing traces to prioritize follow-up, not as a definitive speed result. Its label should be marginal downstream gain per unit of charged compute, not whether an edge belongs to a reference tour. A larger holdout and manuscript revision are justified only after the mechanism succeeds on development data and the policy is frozen.

## Reproduction and archive

The experiment entry point is [`notebooks/05_BetaTour_v11_POP6_candidate_mechanism.ipynb`](../../notebooks/05_BetaTour_v11_POP6_candidate_mechanism.ipynb), regenerated by `python experiments/v11_candidate/build_colab_notebook.py`. Its frozen manifest, Drive folder, resume checks, source hashes and result/audit ZIPs are part of the reproducible run. Save the raw tours and candidate snapshots in the full audit archive; a compact results ZIP alone cannot independently verify route validity or source-state equality. Any post-run code change must use a new experiment name rather than silently continuing under a previous manifest.
