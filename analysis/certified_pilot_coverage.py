"""Recompute the coverage of proven optimal tours from the per-instance table of the certified-graph pilot.

Usage:
    python analysis/certified_pilot_coverage.py results/certified_pilot/instances.csv

The table is `summary/instances.csv` as the pilot's notebook wrote it (experiments/certified_pilot/). For each of the
instances whose optimum Concorde proved, `missing_opt_pct.<list>` is the share of the optimal tour's edges that
neither endpoint lists among its first five candidates; the script turns it back into edge counts and prints the
totals, the instances whose whole optimal tour lies inside the lists and the per-instance comparisons of BetaTour's
top five with the other lists (exact two-sided sign test, ties excluded, uncorrected). These comparisons were
exploratory (manuscript, Section 5.12). It also prints Concorde's solve times by group.
"""

from __future__ import annotations

import csv
import sys
from math import comb

LISTS = [("betatour", "BetaTour top five"), ("betatour_mix", "BetaTour top three + POPMUSIC"),
         ("count", "counting top five"), ("alpha_lkh", "alpha-5 (LKH-3, after ascent)"),
         ("popmusic", "POPMUSIC-5 (first process)"), ("nearest", "nearest five")]


def sign_p(wins: int, losses: int) -> float:
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def main(path: str) -> None:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    solved = [r for r in rows if r["opt"] not in ("", "nan")]
    print(f"{len(rows)} instances, {len(solved)} with an optimum proven by Concorde; "
          f"{sum(int(r['n']) for r in solved):,} optimal edges")
    known = [r for r in solved if r["known_opt"] not in ("", "nan")]
    print("TSPLIB optima equal to the published value: "
          + ", ".join(f"{r['name']} {r['matches_known']}" for r in known))
    miss = {key: [round(float(r[f"missing_opt_pct.{key}"]) * int(r["n"]) / 100) for r in solved] for key, _ in LISTS}
    for key, label in LISTS:
        print(f"  {label:32s} missed {sum(miss[key]):4d} optimal edges; whole optimal tour inside the lists on "
              f"{sum(m == 0 for m in miss[key]):2d}/{len(solved)} instances")
    print("BetaTour top five against (fewer / equal / more missed edges, exact sign test):")
    for key, label in LISTS[1:]:
        w = sum(a < b for a, b in zip(miss["betatour"], miss[key]))
        t = sum(a == b for a, b in zip(miss["betatour"], miss[key]))
        loss = sum(a > b for a, b in zip(miss["betatour"], miss[key]))
        print(f"  {label:32s} {w:2d} / {t:2d} / {loss:2d}  p = {sign_p(w, loss):.2g}")
    print("Concorde's full solve, solved instances (mean seconds, mean branch-and-bound nodes):")
    for group in sorted({r["group"] for r in solved}):
        g = [r for r in solved if r["group"] == group]
        print(f"  {group:7s} {len(g)} solved: {sum(float(r['concorde_full_s']) for r in g) / len(g):7.0f} s, "
              f"{sum(float(r['bbnodes']) for r in g) / len(g):5.0f} nodes")
    unsolved = [r["name"] for r in rows if r["opt"] in ("", "nan")]
    print("not finished within the limit: " + ", ".join(unsolved))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
