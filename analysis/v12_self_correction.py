"""Recompute Table 10 and the run-v12 statistics of the paper from the result ZIP.

Usage:
    python analysis/v12_self_correction.py results/v12/v12_selfcorrect43_results.zip [results/v5/paper51_v1_results.zip]

It reads only the archived instance records. With the run-v5 ZIP it also repeats the descriptive comparison with the
uninterrupted POPMUSIC arm of run v5 (first-trial costs and best-of-six lengths). Instances are the units of the sign
test; group values are means over instances of seed means.
"""

from __future__ import annotations

import json
import sys
import zipfile
from math import comb

import numpy as np

GROUPS = ["uni1k", "clu1k", "uni10k", "clu10k", "tsplib"]


def sign_p(wins: int, losses: int) -> float:
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def instance_values(r: dict) -> dict:
    n, rows = r["n"], r["seed_rows"]
    per = lambda key: float(np.mean([1000.0 * s[key] / n for s in rows]))
    absent = sum(n - s["r0_in_final"] - s["rc_in_final"] for s in rows)
    absent_from_c0 = sum(s["missing_c0"] - s["r0_in_final"] for s in rows)
    r0 = sum(s["missing_c0"] for s in rows)
    rc = sum(s["present_c0"] for s in rows)
    out = {"missing_c0": per("missing_c0"), "missing_beta": 1000.0 * r["missing_beta"] / n,
           "missing_count": 1000.0 * r["missing_count"] / n}
    for f in (25, 50, 75, 100):
        out[f"dynamic_{f}"] = per(f"missing_dynamic_{f}")
        out[f"reached_{f}_pct"] = 100.0 * sum(s[f"recovered_{f}"] for s in rows) / r0
    out["in_final_if_missing_pct"] = 100.0 * sum(s["r0_in_final"] for s in rows) / r0
    out["in_final_if_present_pct"] = 100.0 * sum(s["rc_in_final"] for s in rows) / rc
    out["absent_from_final"] = 1000.0 * absent / len(rows) / n
    # undefined when every final tour equals the reference tour edge for edge
    out["absent_missing_from_c0_pct"] = 100.0 * absent_from_c0 / absent if absent else float("nan")
    out["beta_not_reached"] = per("r0_beta_not_reached")
    out["reached_not_beta"] = per("r0_reached_not_beta")
    return out


def main(path: str, v5_path: str | None = None) -> None:
    z = zipfile.ZipFile(path)
    assert z.testzip() is None, "CRC error"
    manifest = json.loads(z.read("manifest.json"))
    records = [json.loads(z.read(n)) for n in sorted(z.namelist())
               if n.startswith("instances/") and n.endswith(".json")]
    assert len(records) == 43 and all(r["complete"] for r in records)
    assert all(r["code_sha256"] == manifest["code_sha256"] for r in records)
    print(f"{manifest['experiment_name']}: {len(records)} instances; GPU {manifest['hardware']['gpu']}, "
          f"{manifest['hardware']['vcpus_visible']} vCPUs; LKH {manifest['lkh_source']}")
    vals = {r["name"]: (r["group"], instance_values(r)) for r in records}
    keys = list(next(iter(vals.values()))[1])
    print("\nTable 10 (means over instances; edges per 1,000 nodes unless marked %)")
    print("measure".ljust(30) + "".join(g.rjust(9) for g in GROUPS + ["all"]))
    for k in keys:
        cells = [np.nanmean([v[k] for g, v in vals.values() if g == grp]) for grp in GROUPS]
        cells.append(np.nanmean([v[k] for _, v in vals.values()]))
        print(k.ljust(30) + "".join(f"{c:9.2f}" if "pct" not in k else f"{c:9.1f}" for c in cells))
    d = np.array([v["dynamic_100"] - v["missing_beta"] for _, v in vals.values()])
    wins, ties, losses = int((d < -1e-12).sum()), int((np.abs(d) <= 1e-12).sum()), int((d > 1e-12).sum())
    same = [n for n, (_, v) in vals.items() if np.isnan(v["absent_missing_from_c0_pct"])]
    print("Instances whose six final tours all equal the reference tour (share undefined):", same)
    print(f"\nPrimary: dynamic set at the budget fewer/equal/more missing than BetaTour top-5: "
          f"{wins}/{ties}/{losses}; exact sign p = {sign_p(wins, losses):.5f}")
    flags = {(r["name"], s): f for r in records for s, f in r["flags"].items()}
    print("Seed runs with flags:", len(flags), sorted({x for f in flags.values() for x in f}))
    print(f"Instance wall time: {sum(r['wall_s'] for r in records) / 60:.1f} min in total")
    if v5_path:
        z5 = zipfile.ZipFile(v5_path)
        same_first = equal_best = worse = better = 0
        rel = []
        for r in records:
            r5 = json.loads(z5.read(f"instances/{r['name']}.json"))
            same_first += [int(x) for x in r5["phase1"]["first_costs"]] == r["first_trial_costs"]
            b5, b12 = r5["arms"]["pop_full"]["best_at_budget"], r["best_of_six_cost"]
            equal_best += b12 == b5
            worse += b12 > b5
            better += b12 < b5
            rel.append(100.0 * (b12 - b5) / b5)
        print(f"\nRun v5 check: first-trial costs identical on {same_first}/43 instances; best of six equal on "
              f"{equal_best}, longer on {worse}, shorter on {better} (max {max(rel):+.4f}%)")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
