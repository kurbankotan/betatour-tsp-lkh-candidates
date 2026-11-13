> **Note.** This brief was written before run v11 and accompanied the run-v11 code when that code was sent for an independent review (see the generative-AI declaration of the paper). It is kept unchanged below as the record of the hypothesis, the controls and the primary endpoint fixed before the run. The results of runs v11 and v11r2 are reported in `README.md` and in Sections 5.9 and A.12 of the paper.

# BetaTour v11: independent review brief

This is a request for a technical critique of a **proposed experiment**, not a request to assume its results are positive. Please inspect the code and protocol before interpreting any run. The manuscript is intentionally unchanged pending a positive, disjoint evaluation.

## What the earlier pilots actually showed

- v9 `beta_sparse`: 6/8 wins over native POPMUSIC-5, mean relative cost −0.00304%; 4/8 wins over native POPMUSIC-6, mean +0.00266% (worse). The eight instances were development cases, and the POPMUSIC-5 advantage could be affected by extra candidate capacity. There was no reliable separation from the tour-frequency control.
- v10 external repair: 43/48 BetaTour barrier tours improved, but all 48 resumed LKH final tours beat their repaired counterparts. The on-the-side repair was never selected as the final result and paused the main search for about 22 seconds. It also used a reference-edge likelihood checkpoint as a repair-window risk score without training for repair utility.

## v11 hypothesis

Residual model edges may help *inside* a strong POPMUSIC-6/LKH search when the model identifies a missing useful edge. Candidate quantity and overhead must be controlled. A positive final tour alone is insufficient: the experiment should show that the action is enumerated by the search and beats relevant native and heuristic controls under equal wall time. The hook also observes whether an action enters a globally better incumbent. This is observational, not proof that the action caused an accepted move or improvement. The current checkpoint is frozen; a utility predictor is only a possible next stage.

Planned controls are native POPMUSIC-6/7, patched POPMUSIC-6 sham, and patched POPMUSIC-6 with equal-gate BetaTour, frequency and deterministic random residual candidates. The first pilot has only four previously examined 10,000-node instances (two uniform, two clustered), so it is a mechanism check rather than a 20,000-node or independent performance test. Six seeds are run per instance; inference and synchronization are charged. Same-state claims require matching pre-action tour and candidate hashes. Use the instance-level best-of-six endpoint; do not treat six seeds as six independent test instances. Patched arms log exact hook-level `best.events`. Native POP6/7 `native_improvement_events` come from polling output-tour modification times, which are approximate and can miss rapid improvements. Cross-arm time-to-target curves are exploratory, not a precise speed claim.

Guided candidates must be absent from the seed's POP6 candidate graph and first tour. The three selectors share the model graph and BetaTour source-node gate. The analysis exports the selected-edge ensemble frequencies; if those values are mostly zero, the frequency arm effectively ranks by its distance tie break. Candidate and incumbent telemetry is periodically snapshotted during long trials, but the last interrupted interval may be missing; inspect `telemetry_age_at_end_s` before interpreting small counter differences.

## Questions worth trying to falsify

1. Do the instrumented candidate-enumeration counters cover every relevant LKH move class? Identify their exact source locations and omissions. The hook does **not** measure gain-filter passage or every accepted local k-opt move; does its incumbent-entry count clearly avoid being described as either? Do not infer causal use from post-action membership or final-tour overlap alone.
2. Do guided arms choose exactly the same source-node gate and directed action count? Do they match in *unique undirected* added edges? If not, how should the comparison be qualified?
3. Is a first-completed-trial barrier enough to reproduce the same tour, native candidate set and stochastic search state across separately launched arms? Are all mismatches detected and excluded from a paired policy claim?
4. Are unmodified POPMUSIC-6/7, sham, model, frequency and random compared under the same wall-clock budget, including CPU/GPU use, file transfer, inference and the barrier? Could the model win simply by receiving more candidate capacity? Does native output-tour polling add noticeable overhead or miss improvements that distort exploratory time-to-target curves?
5. Are the feature and training targets aligned with actual marginal search gain? The existing edge probability may rank near-reference edges while being irrelevant to accepted LKH moves. What paired counterfactual rollout would provide a defensible utility label?
6. Would a claimed gain survive a disjoint holdout, multiple instance families, both endpoint quality and time-to-target, and a stronger baseline? Is the contribution distinct from published NeuroLKH-style candidate/penalty guidance?

Please report concrete correctness defects separately from methodological limits and from speculative architecture suggestions. A clean run with no crashes is not evidence of search benefit. If the pilot fails its mechanism gate, the result should be recorded as negative and the current manuscript left unchanged.
