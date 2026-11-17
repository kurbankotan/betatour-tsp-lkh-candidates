# BetaTour v15: live candidate replacement pilot

This exploratory experiment tests whether BetaTour can improve LKH-3.0.14 by replacing low-priority POPMUSIC-6 candidates **in place** after the first completed trial. Unlike the v11r2 candidate-addition pilot, the intervention does not lengthen directed candidate rows at the barrier. LKH may later append or reorder rows through its ordinary search logic. A victim can remain in the reverse row, so the intervention is not guaranteed to keep the number of unique undirected edges fixed. The code records this condition.

The four 10,000-node instances are reused from earlier pilots. Results are for policy development, not an independent performance claim. The manuscript is unchanged pending a positive disjoint holdout.

## Arms and fairness

| Arm | Intervention after first trial |
| --- | --- |
| `native_pop6` | Unmodified, uninterrupted POPMUSIC-6 LKH; practical baseline. |
| `sham` | Patched LKH pauses for the barrier and pays for model inference; no swap. |
| `random_swap` | Deterministically random eligible new neighbor. |
| `nearest_swap` | Nearest eligible new neighbor. |
| `beta_swap` | Highest model probability among eligible new neighbors. |

The three swap arms use identical source nodes, removed neighbors, and action counts per seed. The policy protects the first two positions in each candidate row and every edge in the union of the six first tours. The C hook independently checks the `protected.edges` file before accepting swaps. A proposed new neighbor is absent from the live *undirected* candidate graph and from all six first tours. The C hook replaces the victim in the same row position, leaves the row length unchanged, and writes `post.candidates`; Python verifies the exact row before accepting an endpoint.

Each arm starts six fresh LKH processes with the same seeds. The four patched arms must match on six first tours, native/live candidate snapshots, model decision array, source gates, victims, and action capacities for a policy comparison. A single 120-second wall deadline includes LKH startup, first trial, barrier, model inference, and subsequent search. The primary exploratory endpoint is the shortest valid EUC_2D tour of six seeds at the deadline; each instance is one unit. The six seeds are not six independent replicates.

## Colab

Open [`notebooks/15_BetaTour_v15_live_swap_pilot.ipynb`](../../notebooks/15_BetaTour_v15_live_swap_pilot.ipynb) in Colab Pro and run all cells. CPU, GPU, or High RAM runtimes are accepted. CUDA is used when available; CPU is the fallback. The notebook records GPU, visible vCPU count and library versions. Fewer than six vCPUs make six parallel LKH searches slower and may exhaust the 120-second budget before the swap barrier; such an arm is incomplete and excluded from endpoint comparisons. Compare only arms run under the same hardware profile.

If a two-vCPU CPU runtime cannot reach the barrier in 120 seconds, set all four `budget_s` entries in notebook cell 1 to `300.0` and use a fresh `EXPERIMENT_NAME`, such as `v15_live_swap_pilot_cpu300`. The longer run is a separate, internally matched experiment; do not pool its endpoints with the 120-second run.

The notebook stores each completed arm atomically under `MyDrive/BetaTour/v15_live_swap_pilot`. On restart it verifies completed arms and skips them; an interrupted arm starts in a new attempt directory. A changed code/settings/hardware profile requires a new experiment name. The compact results ZIP downloads through the browser after completion, while the full audit ZIP stays in Drive. The audit includes tours, snapshots, action files, protected edges, telemetry, and embedded code.

Regenerate the notebook with:

```bash
python experiments/v15_live_swap/build_colab_notebook.py
```

## Interpretation

A swap arm beating the sham but losing to native POPMUSIC-6 would mean the intervention has a measurable search effect but does not repay its barrier and inference cost. A win over native POPMUSIC-6 on four reused instances is a lead for a frozen, disjoint test; it is not evidence of general solver superiority. Candidate edge visits, accepted-tour entries, and appearances in improved incumbents are mechanism observations, not causal credit. The v12 self-correction result suggests that target-tour edge recall is a weak proxy for search gain, so the next model should be trained on marginal continuation gain per unit time if this pilot warrants further work.
