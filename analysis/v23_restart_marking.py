"""Recompute the comparisons of run v23 (exploratory run on restart markings) from the result ZIP.

Usage:
    python analysis/v23_restart_marking.py results/v23/v23_marking12_results.zip [results/v21/v21_alphaenc24_results.zip]

It reads only the archived instance records and shares no code with the notebook. Percentage difference =
100 x (first arm - second arm) / second arm; negative values favour the first arm. Counts are instances on which the
first arm was shorter / equal / longer. best6: best of the six seeds. mean6: mean over the six seeds of the paired
differences. Instances are the units of the exact two-sided sign test (ties excluded) and of the bootstrap (20,000
resamples). No comparison of this run is a confirmatory test. With the run-v21 ZIP the script also prints the
comparisons that the two runs share, side by side.
"""

from __future__ import annotations

import json
import sys
import zipfile
from math import comb

import numpy as np

ARMS = ["pop_full", "pop_native", "pop_model", "beta_tree", "beta_model", "beta_count"]
MAIN = [("M1", "beta_model", "beta_tree"), ("M2", "beta_count", "beta_tree"), ("M3", "pop_model", "pop_native")]
PAIRS = [("beta_model", "beta_count"), ("beta_model", "pop_full"), ("beta_count", "pop_full"), ("beta_tree", "pop_full"),
         ("pop_model", "pop_full"), ("pop_native", "pop_full"), ("beta_tree", "pop_native"), ("beta_model", "pop_model")]
SHARED = [("beta_tree", "pop_full"), ("pop_native", "pop_full"), ("beta_tree", "pop_native")]


def sign_p(wins: int, losses: int) -> float:
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def seed_bests(r: dict, arm: str, fraction: float = 1.0) -> list:
    t = fraction * r["budget_s"] + 1e-9
    return [min(c for c, tt in run["events"] if tt <= t) for run in r["arms"][arm]["runs"]]


def delta(r: dict, a: str, b: str, endpoint: str, fraction: float = 1.0) -> float:
    x, y = seed_bests(r, a, fraction), seed_bests(r, b, fraction)
    if endpoint == "best6":
        return pct(min(x), min(y))
    return float(np.mean([pct(u, v) for u, v in zip(x, y)]))


def describe(values: list) -> str:
    v = np.asarray(values)
    wins, losses = int((v < 0).sum()), int((v > 0).sum())
    boot = v[np.random.default_rng(2026).integers(0, len(v), (20000, len(v)))].mean(1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return (f"{wins:>2}/{len(v) - wins - losses}/{losses:<2}  mean {v.mean():+.4f}  median {np.median(v):+.4f}  "
            f"95% CI {low:+.4f} to {high:+.4f}  sign p {sign_p(wins, losses):.4f}")


def load(path: str) -> tuple:
    z = zipfile.ZipFile(path)
    assert z.testzip() is None, "CRC error"
    manifest = json.loads(z.read("manifest.json"))
    recs = sorted((json.loads(z.read(n)) for n in z.namelist() if n.startswith("instances/") and n.endswith(".json")),
                  key=lambda r: r["ordinal"])
    assert all(r["complete"] and r["code_sha256"] == manifest["code_sha256"] for r in recs)
    return manifest, recs


def main(path: str, v21_path: str | None = None) -> None:
    manifest, recs = load(path)
    assert len(recs) == 12 and all(r["marking_sha256"] == manifest["marking_sha256"] for r in recs)
    for r in recs:                                   # stored summaries agree with the stored traces
        for arm in ARMS:
            assert seed_bests(r, arm) == [run["best_at_budget"] for run in r["arms"][arm]["runs"]]
    restarted = [run for r in recs for arm in ARMS[1:] for run in r["arms"][arm]["runs"]]
    lists = lambda key: np.array([r["lists"][key] for r in recs])
    print(f"{manifest['experiment_name']}: {len(recs)} instances; GPU {manifest['hardware']['gpu']}, "
          f"{manifest['hardware']['vcpus_visible']} vCPUs; sessions {len({r['session_id'] for r in recs})}; "
          f"first instance finished {min(r['finished_utc'] for r in recs)}, last {max(r['finished_utc'] for r in recs)}")
    print(f"first trial of the uninterrupted arm equals the shared first phase on "
          f"{sum(all(r['first_trial_matches_uninterrupted']) for r in recs)} instances; tree-rule mismatches "
          f"{int(lists('tree_rule_mismatch').sum())}; marking-writer mismatches {int(lists('mark_writer_mismatch').sum())}; "
          f"restarted processes stopped from outside {sum(run['timed_out'] for run in restarted)} of {len(restarted)}")
    count, both = lists("first_tour_edges_trusted_by_count"), lists("first_tour_edges_trusted_by_both")
    print(f"model threshold {lists('model_threshold').min():.4f} to {lists('model_threshold').max():.4f}; edges of the "
          f"first tour protected by the count marking {count.min():.3f} to {count.max():.3f} (mean {count.mean():.3f}); "
          f"share of them also protected by the model marking {(both / count).mean():.3f}")

    print("\nMain readings at the budget (mean6)")
    for key, a, b in MAIN:
        print(f"  {key} {a + ' / ' + b:<24} {describe([delta(r, a, b, 'mean6') for r in recs])}")

    print("\nAll comparisons at the budget (descriptive)")
    for a, b in [(m[1], m[2]) for m in MAIN] + PAIRS:
        for endpoint in ("best6", "mean6"):
            print(f"  {a + ' / ' + b:<24} {endpoint}: {describe([delta(r, a, b, endpoint) for r in recs])}")

    print("\nBy family (main readings, mean6)")
    for group in ("uni10k", "clu10k"):
        sub = [r for r in recs if r["group"] == group]
        for key, a, b in MAIN:
            print(f"  {group} {key}: {describe([delta(r, a, b, 'mean6') for r in sub])}")

    print("\nTime course (share of the budget; mean6)")
    for a, b in [(m[1], m[2]) for m in MAIN] + [("beta_tree", "pop_full")]:
        for f in (0.25, 0.5, 0.75, 1.0):
            print(f"  {a + ' / ' + b:<24} {int(100 * f):>3}%: {describe([delta(r, a, b, 'mean6', f) for r in recs])}")

    print("\nArms: instances with the shortest tour of the six arms (best6, ties counted for every arm); trials completed")
    print("within the budget (mean of seeds and instances); share of the first-trial tour's edges with a zero alpha-value")
    for arm in ARMS:
        shortest = sum(r["arms"][arm]["best_at_budget"] == min(r["arms"][a]["best_at_budget"] for a in ARMS) for r in recs)
        trials = np.mean([run["trials_at_fraction"][-1] for r in recs for run in r["arms"][arm]["runs"]])
        kept = "" if arm == "pop_full" else f"   kept share {np.mean([r['lists']['kept_share'][arm] for r in recs]):.3f}"
        print(f"  {arm:<10} shortest on {shortest:>2}   trials {trials:6.1f}{kept}")

    if v21_path:
        _, v21 = load(v21_path)
        print(f"\nComparisons shared with run v21 ({len(v21)} instances there, {len(recs)} here; not pooled)")
        for a, b in SHARED:
            for endpoint in ("best6", "mean6"):
                print(f"  {a + ' / ' + b:<24} {endpoint}  run v21: {describe([delta(r, a, b, endpoint) for r in v21])}")
                print(f"  {'':<24} {'':<5}  run v23: {describe([delta(r, a, b, endpoint) for r in recs])}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
