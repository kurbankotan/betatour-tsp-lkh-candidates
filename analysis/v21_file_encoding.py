"""Recompute the comparisons of run v21 (candidate-file encoding control) from the result ZIP.

Usage:
    python analysis/v21_file_encoding.py results/v21/v21_alphaenc24_results.zip [results/v5/paper51_v1_results.zip]

It reads only the archived instance records and shares no code with the notebook. Percentage difference =
100 x (first arm - second arm) / second arm; negative values favour the first arm. Counts are instances on which the
first arm was shorter / equal / longer. best6: best of the six seeds. mean6: mean over the six seeds of the paired
differences. Instances are the units of the exact two-sided sign test and of the bootstrap (20,000 resamples). With
the run-v5 ZIP the script also prints the corresponding comparisons of run v5 on its 10,000-node instances.
"""

from __future__ import annotations

import json
import sys
import zipfile
from math import comb

import numpy as np

ARMS = ["pop_full", "pop_native", "pop_rank", "beta_tree", "beta_rank", "freq_tree"]
PRIMARY = [("P1", "pop_rank", "pop_native", "mean6"), ("P2", "beta_tree", "pop_full", "best6")]
PAIRS = [("pop_native", "pop_full"), ("pop_rank", "pop_full"), ("beta_tree", "pop_native"), ("beta_rank", "pop_rank"),
         ("beta_rank", "beta_tree"), ("beta_tree", "freq_tree"), ("freq_tree", "pop_full"), ("beta_rank", "pop_full")]


def sign_p(wins: int, losses: int) -> float:
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def seed_bests(r: dict, arm: str, t: float) -> list[int]:
    return [min(c for c, tt in run["events"] if tt <= t + 1e-9) for run in r["arms"][arm]["runs"]]


def delta(r: dict, a: str, b: str, endpoint: str, t: float) -> float:
    x, y = seed_bests(r, a, t), seed_bests(r, b, t)
    if endpoint == "best6":
        return pct(min(x), min(y))
    return float(np.mean([pct(u, v) for u, v in zip(x, y)]))


def describe(values: list[float]) -> tuple[str, float]:
    v = np.asarray(values)
    wins, losses = int((v < 0).sum()), int((v > 0).sum())
    boot = v[np.random.default_rng(2026).integers(0, len(v), (20000, len(v)))].mean(1)
    low, high = np.quantile(boot, [0.025, 0.975])
    p = sign_p(wins, losses)
    return (f"{wins:>2}/{len(v) - wins - losses}/{losses:<2}  mean {v.mean():+.4f}  median {np.median(v):+.4f}  "
            f"95% CI {low:+.4f} to {high:+.4f}  sign p {p:.4f}"), p


def main(path: str, v5_path: str | None = None) -> None:
    z = zipfile.ZipFile(path)
    assert z.testzip() is None, "CRC error"
    manifest = json.loads(z.read("manifest.json"))
    recs = sorted((json.loads(z.read(n)) for n in z.namelist() if n.startswith("instances/") and n.endswith(".json")),
                  key=lambda r: r["ordinal"])
    budget = manifest["design"]["budget_s"]
    assert len(recs) == 24 and all(r["complete"] and r["budget_s"] == budget for r in recs)
    assert all(r["code_sha256"] == manifest["code_sha256"] for r in recs)
    for r in recs:                                   # stored summaries agree with the stored traces
        for arm in ARMS:
            assert seed_bests(r, arm, budget) == [run["best_at_budget"] for run in r["arms"][arm]["runs"]]
    runs = [(arm, run) for r in recs for arm in ARMS for run in r["arms"][arm]["runs"]]
    stopped = sum(run["timed_out"] for arm, run in runs if arm != "pop_full")
    print(f"{manifest['experiment_name']}: {len(recs)} instances; GPU {manifest['hardware']['gpu']}, "
          f"{manifest['hardware']['vcpus_visible']} vCPUs; sessions {len({r['session_id'] for r in recs})}; "
          f"LKH {manifest['lkh_source']}")
    print(f"first trial of the uninterrupted arm equals the shared first phase on "
          f"{sum(all(r['first_trial_matches_uninterrupted']) for r in recs)} instances; tree-rule mismatches "
          f"{sum(sum(r['lists']['tree_rule_mismatch']) for r in recs)}; restarted processes stopped from outside "
          f"{stopped} of {sum(arm != 'pop_full' for arm, _ in runs)}")

    print("\nPrimary comparisons at the budget")
    ps = []
    for key, a, b, endpoint in PRIMARY:
        text, p = describe([delta(r, a, b, endpoint, budget) for r in recs])
        ps.append(p)
        print(f"  {key} {a} / {b} ({endpoint}): {text}")
    order = np.argsort(ps)
    holm, running = [0.0, 0.0], 0.0
    for rank, j in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[j])
        holm[j] = min(1.0, running)
    print("  Holm-adjusted p: " + ", ".join(f"{k[0]} {h:.4f}" for k, h in zip(PRIMARY, holm)))

    print("\nSecondary comparisons at the budget (descriptive)")
    for a, b in [(p[1], p[2]) for p in PRIMARY] + PAIRS:
        for endpoint in ("best6", "mean6"):
            print(f"  {a + ' / ' + b:<25} {endpoint}: {describe([delta(r, a, b, endpoint, budget) for r in recs])[0]}")

    print("\nBy family (primary endpoints)")
    for group in ("uni10k", "clu10k"):
        sub = [r for r in recs if r["group"] == group]
        for key, a, b, endpoint in PRIMARY:
            print(f"  {group} {key}: {describe([delta(r, a, b, endpoint, budget) for r in sub])[0]}")

    print("\nTime course (share of the budget)")
    for a, b, endpoint in [("pop_rank", "pop_native", "mean6"), ("beta_tree", "pop_full", "best6"),
                           ("pop_native", "pop_full", "best6"), ("beta_tree", "freq_tree", "best6")]:
        for f in (0.25, 0.5, 0.75, 1.0):
            print(f"  {a + ' / ' + b:<25} {endpoint} {int(100 * f):>3}%: "
                  f"{describe([delta(r, a, b, endpoint, f * budget) for r in recs])[0]}")

    print("\nArms: trials completed within the budget (mean of seeds and instances), share of the first-trial tour's")
    print("edges with a zero alpha-value in the arm's file, time charged before LKH's own time limit (s)")
    for arm in ARMS:
        trials = np.mean([run["trials_at_fraction"][-1] for r in recs for run in r["arms"][arm]["runs"]])
        if arm == "pop_full":
            print(f"  {arm:<10} {trials:6.1f}")
        else:
            kept = np.mean([r["lists"]["kept_share"][arm] for r in recs])
            prelude = np.mean([r["arms"][arm]["prelude_s"] for r in recs])
            print(f"  {arm:<10} {trials:6.1f}   kept share {kept:.3f}   charged {prelude:5.2f}")

    if v5_path:
        z5 = zipfile.ZipFile(v5_path)
        v5 = [json.loads(z5.read(n)) for n in sorted(z5.namelist())
              if n.startswith("instances/") and ("uni10k" in n or "clu10k" in n)]
        best = lambda r, arm: min(run["best_at_budget"] for run in r["arms"][arm]["runs"])
        print(f"\nRun v5, its {len(v5)} instances of 10,000 nodes (files without LKH's alpha-values and fathers), best6")
        for a, b in [("restart_pop_raw", "pop_full"), ("betatour", "pop_full"), ("betatour", "restart_pop_raw")]:
            print(f"  {a + ' / ' + b:<28} {describe([pct(best(r, a), best(r, b)) for r in v5])[0]}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
