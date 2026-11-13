# BetaTour v10: external repair pilot

**Status:** completed development pilot (run v10 of the paper, Sections 4.13, 5.8 and A.11). It made no performance claim and found no benefit. `README_tr.md` is the original Turkish protocol that was bundled with the notebook.

## Question and method

Run v9 added candidate edges once after the first trial and showed no reliable gain over POPMUSIC-6. Run v10 tested a mechanism that changes the current tour directly. Six POPMUSIC-6 LKH processes (seeds 301-306) paused at the end of the first trial that ended at least 35% of the budget after their search began (hook `BetaTourRepairHook.h`, built by `build_repair_hook.py`). While they waited, every seed's barrier tour received eight rounds of compressed repair (`compressed_repair.py`, `repair_engine.py`):

1. choose three cut edges that are close in the plane but at least 256 tour positions apart;
2. free a window of 256 tour positions around each cut and contract every untouched tour path into a mandatory zero-cost edge;
3. solve the reduced instance with a separate LKH-3 run (ALPHA candidates without ascent, five per node, 100 trials, 2 s limit) and expand it back to a full tour;
4. keep the result only if it is a valid tour that is strictly shorter.

The repair phase was capped at 30% of the budget. The LKH processes then resumed unchanged; the repaired tours were never injected into them. The endpoint of a seed was the shorter of its final LKH tour and its repaired tour, and the arm endpoint was the best of the six seeds. Model inference, repairs and the pause counted toward the arm's wall-clock deadline (120 s at 10,000 nodes, 300 s at 20,000 nodes).

## Arms

| Arm | Role |
| --- | --- |
| `native_pop6` | Unmodified, uninterrupted POPMUSIC-6. |
| `sham` | Same hook and pause, no repair. |
| `repair_random` | Repair with uniformly random cut anchors. |
| `repair_longest` | Cuts biased towards long tour edges. |
| `repair_disagreement` | Cuts biased towards tour edges used by few of the six barrier tours. |
| `repair_betatour` | Cuts biased towards edges with a low BetaTour probability (risk = 1 - model probability, from the six barrier tours). |

All repair arms used the same repair engine, number of rounds and caps; only the cut score differed. Because the pause was triggered by time, separately run arms could pause after different trials and in different states; the analysis cell reports how often the barrier tours matched.

## Files

- `repair_pilot.py`: runs one arm (pause, inference, repairs, resume, endpoint).
- `compressed_repair.py`: exact path compression, reduced-instance solve and expansion.
- `repair_engine.py`: cut selection and tour-cost helpers.
- `BetaTourRepairHook.h`, `build_repair_hook.py`: delayed-barrier hook for LKH-3.0.14 and its build.
- `test_compressed_repair.py`: checks that compression and expansion preserve tours and costs.

The notebook `notebooks/10_BetaTour_v10_external_repair_pilot.ipynb` embeds the exact files it ran, and `results/v10/` holds its result archive.
