"""Recompute the comparisons of run v22 (corrected candidate files on the other sizes and on TSPLIB) from the result ZIP.

Usage:
    python analysis/v22_other_sizes.py results/v22/v22_sizes43_results.zip

It reads only the archived instance records and shares no code with the notebook. Percentage difference =
100 x (first arm - second arm) / second arm; negative values favour the first arm. Counts are instances on which the
first arm was shorter / equal / longer. best6: best of the six seeds. mean6: mean over the six seeds of the paired
differences. Instances are the units of the exact two-sided sign test (ties excluded) and of the bootstrap (20,000
resamples).
"""

from __future__ import annotations

import json
import sys
import zipfile
from math import comb

import numpy as np

ARMS = ["pop_full", "pop_native", "beta_tree", "freq_tree"]
PRIMARY = [("P1", "beta_tree", "pop_full", "mean6"), ("P2", "beta_tree", "freq_tree", "mean6")]
PAIRS = [("beta_tree", "pop_full"), ("beta_tree", "freq_tree"), ("pop_native", "pop_full"), ("beta_tree", "pop_native"),
         ("freq_tree", "pop_full"), ("freq_tree", "pop_native")]
GROUPS = ["uni1k", "clu1k", "uni20k", "clu20k", "tsplib"]


def sign_p(wins: int, losses: int) -> float:
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def seed_bests(r: dict, arm: str, fraction: float = 1.0) -> list:
    """Best tour length of each seed within the given share of the instance's budget (None: no tour yet)."""
    t = fraction * r["budget_s"] + 1e-9
    out = []
    for run in r["arms"][arm]["runs"]:
        costs = [c for c, tt in run["events"] if tt <= t]
        out.append(min(costs) if costs else None)
    return out


def delta(r: dict, a: str, b: str, endpoint: str, fraction: float = 1.0):
    x, y = seed_bests(r, a, fraction), seed_bests(r, b, fraction)
    if any(v is None for v in x + y):
        return None
    if endpoint == "best6":
        return pct(min(x), min(y))
    return float(np.mean([pct(u, v) for u, v in zip(x, y)]))


def describe(values: list) -> tuple:
    v = np.asarray([x for x in values if x is not None])
    wins, losses = int((v < 0).sum()), int((v > 0).sum())
    boot = v[np.random.default_rng(2026).integers(0, len(v), (20000, len(v)))].mean(1)
    low, high = np.quantile(boot, [0.025, 0.975])
    p = sign_p(wins, losses)
    return (f"{wins:>2}/{len(v) - wins - losses:>2}/{losses:<2}  mean {v.mean():+.4f}  median {np.median(v):+.4f}  "
            f"95% CI {low:+.4f} to {high:+.4f}  sign p {p:.4f}"), p


def main(path: str) -> None:
    z = zipfile.ZipFile(path)
    assert z.testzip() is None, "CRC error"
    manifest = json.loads(z.read("manifest.json"))
    recs = sorted((json.loads(z.read(n)) for n in z.namelist() if n.startswith("instances/") and n.endswith(".json")),
                  key=lambda r: r["ordinal"])
    assert len(recs) == 43 and all(r["complete"] and r["code_sha256"] == manifest["code_sha256"] for r in recs)
    for r in recs:                                   # stored summaries agree with the stored traces
        for arm in ARMS:
            assert seed_bests(r, arm) == [run["best_at_budget"] for run in r["arms"][arm]["runs"]]
    sets = [("all", recs), ("large", [r for r in recs if r["n"] >= 7000]), ("small", [r for r in recs if r["n"] < 7000])]
    sets += [(g, [r for r in recs if r["group"] == g]) for g in GROUPS]
    print(f"{manifest['experiment_name']}: {len(recs)} instances; GPU {manifest['hardware']['gpu']}, "
          f"{manifest['hardware']['vcpus_visible']} vCPUs; sessions {len({r['session_id'] for r in recs})}; "
          f"first instance finished {min(r['finished_utc'] for r in recs)}, last {max(r['finished_utc'] for r in recs)}; "
          f"LKH {manifest['lkh_source']}")
    print(f"first trial of the uninterrupted arm equals the shared first phase on "
          f"{sum(all(r['first_trial_matches_uninterrupted']) for r in recs)} instances; tree-rule mismatches "
          f"{sum(sum(r['lists']['tree_rule_mismatch']) for r in recs)}")
    print("restarted processes stopped by the outside deadline: " + ", ".join(
        f"{g} {sum(run['timed_out'] for r in sub for arm in ARMS[1:] for run in r['arms'][arm]['runs'])}/{18 * len(sub)}"
        for g, sub in sets[3:]))

    print("\nPrimary comparisons at the budget (all 43 instances)")
    ps = []
    for key, a, b, endpoint in PRIMARY:
        text, p = describe([delta(r, a, b, endpoint) for r in recs])
        ps.append(p)
        print(f"  {key} {a} / {b} ({endpoint}): {text}")
    order = np.argsort(ps)
    holm, running = [0.0, 0.0], 0.0
    for rank, j in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[j])
        holm[j] = min(1.0, running)
    print("  Holm-adjusted p: " + ", ".join(f"{k[0]} {h:.4f}" for k, h in zip(PRIMARY, holm)))

    print("\nSecondary comparisons at the budget (descriptive)")
    for name, sub in sets:
        print(f"  {name} ({len(sub)} instances)")
        for a, b in PAIRS:
            for endpoint in ("best6", "mean6"):
                print(f"    {a + ' / ' + b:<25} {endpoint}: {describe([delta(r, a, b, endpoint) for r in sub])[0]}")

    print("\nTime course (share of the budget; mean6)")
    for name, sub in [sets[0], ("20k", [r for r in recs if r["group"] in ("uni20k", "clu20k")]), sets[-1]]:
        for a, b in PAIRS[:4]:
            for f in (0.25, 0.5, 0.75, 1.0):
                print(f"  {name:<6} {a + ' / ' + b:<25} {int(100 * f):>3}%: "
                      f"{describe([delta(r, a, b, 'mean6', f) for r in sub])[0]}")

    print("\nShortest tour of the four arms (best6; instances, ties counted for every arm; in brackets: alone)")
    for name, sub in [sets[0]] + sets[3:]:
        cells = []
        for arm in ARMS:
            low = [min(r["arms"][a]["best_at_budget"] for a in ARMS) for r in sub]
            hit = [r["arms"][arm]["best_at_budget"] == lo for r, lo in zip(sub, low)]
            alone = [h and sum(r["arms"][a]["best_at_budget"] == lo for a in ARMS) == 1 for r, lo, h in zip(sub, low, hit)]
            cells.append(f"{arm} {sum(hit)} ({sum(alone)})")
        print(f"  {name:<7} {len(sub):>2}: " + ", ".join(cells))

    print("\nTSPLIB: gap to the optimal length (%), best6 / mean6 / seeds at the optimum")
    tsplib = sets[-1][1]
    for r in tsplib:
        print(f"  {r['name']:<9} n {r['n']:>6} budget {r['budget_s']:>6.1f} s  " + "  ".join(
            f"{arm} {pct(r['arms'][arm]['best_at_budget'], r['optimum']):.4f} / "
            f"{pct(r['arms'][arm]['mean_at_budget'], r['optimum']):.4f} / "
            f"{sum(run['best_at_budget'] == r['optimum'] for run in r['arms'][arm]['runs'])}" for arm in ARMS))
    for arm in ARMS:
        print(f"  {arm:<10} mean gap best6 {np.mean([pct(r['arms'][arm]['best_at_budget'], r['optimum']) for r in tsplib]):.4f}"
              f"  mean6 {np.mean([pct(r['arms'][arm]['mean_at_budget'], r['optimum']) for r in tsplib]):.4f}"
              f"  instances solved to the optimum {sum(r['arms'][arm]['best_at_budget'] == r['optimum'] for r in tsplib)}")

    print("\n1,000 nodes: seeds (of 48 per family) at the shortest tour that any arm found on the instance")
    for g in ("uni1k", "clu1k"):
        sub = [r for r in recs if r["group"] == g]
        print(f"  {g}: " + ", ".join(
            f"{arm} {sum(run['best_at_budget'] == min(r['arms'][a]['best_at_budget'] for a in ARMS) for r in sub for run in r['arms'][arm]['runs'])}"
            for arm in ARMS))

    print("\nBy group: trials completed within the budget (mean of seeds and instances); time by which the restarted search")
    print("starts after the first trial of the uninterrupted process has ended (% of the budget, pop_native / beta_tree)")
    for g, sub in sets[3:]:
        trials = {arm: np.mean([run["trials_at_fraction"][-1] for r in sub for run in r["arms"][arm]["runs"]]) for arm in ARMS}
        late = {arm: np.mean([100 * (x["search_start_s"] - y["events"][0][1]) / r["budget_s"] for r in sub
                              for x, y in zip(r["arms"][arm]["runs"], r["arms"]["pop_full"]["runs"])])
                for arm in ("pop_native", "beta_tree")}
        print(f"  {g:<7} " + "  ".join(f"{arm} {v:6.1f}" for arm, v in trials.items())
              + f"   restart starts later by {late['pop_native']:.2f} / {late['beta_tree']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
