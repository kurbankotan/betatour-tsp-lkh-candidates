"""Recompute the comparisons of run v25 (pilot on recombining LKH-3's trial tours with GA-EAX) from the result ZIP.

Usage:
    python analysis/v25_trial_tour_recombination.py results/v25/v25_trialtours21_results.zip [--tours SRC [TSPLIB_DIR]]

It reads only the archived records and shares no code with the notebook. The result of every arm is recomputed
from the recorded time courses (wall-clock seconds from the start of the arm) and compared with the archived value:
`lkh6` is the shortest best tour that the six LKH-3 processes had recorded at the budget; a merge arm is the shorter
of the best tour at its cut and the best tour of GA-EAX within its share of the budget; `cascade` is the shorter of
the best first-trial tour and the best tour of GA-EAX within its share; `eax_scratch` (a reference, not an arm) is
the best tour of GA-EAX from scratch after six times the budget. Percentage difference = 100 x (first arm - second
arm) / second arm; negative values favour the first arm. Counts are instances on which the first arm was shorter /
equal / longer. Instances are the units of the exact two-sided sign test (ties excluded) and of the bootstrap
(20,000 resamples). No comparison of this run is a confirmatory test.

With --tours the script also recomputes the length of every archived tour: SRC is the `src` folder of this
repository (the synthetic instances are regenerated from their seeds) and TSPLIB_DIR, if given, a folder with the
TSPLIB files (`rl11849.tsp`, ...); TSPLIB instances without a file are skipped.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from math import comb

import numpy as np

SPLITS = ["33", "50", "67"]
MERGE = [f"{kind}_{s}" for s in SPLITS for kind in ("trials", "best6")]
ARMS = ["lkh6"] + MERGE + ["cascade"]
PRIMARY, CONTROL = "trials_50", "best6_50"
CLASSES = ["10k", "TSPLIB", "50k"]
PAIRS = [("P1", PRIMARY, "lkh6"), ("C1", CONTROL, "lkh6"), ("C2", PRIMARY, CONTROL), ("C3", "cascade", "lkh6"),
         ("C4", PRIMARY, "cascade"), ("S1", "trials_33", "lkh6"), ("S2", "best6_33", "lkh6"),
         ("S3", "trials_67", "lkh6"), ("S4", "best6_67", "lkh6"), ("R1", "eax_scratch", "lkh6"),
         ("R2", PRIMARY, "eax_scratch")]
SYNTHETIC = [("uni10k", 10000, 6, "uniform", 9111), ("clu10k", 10000, 6, "clustered", 9212),
             ("uni50k", 50000, 2, "uniform", 9515), ("clu50k", 50000, 2, "clustered", 9616)]


def sign_p(wins: int, losses: int) -> float:
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def best_until(curve: list, limit: float):
    costs = [c for t, c in curve if t <= limit + 1e-9]
    return min(costs) if costs else None


def size_class(r: dict) -> str:
    return "TSPLIB" if r["group"] == "tsplib" else f"{round(r['n'] / 1000)}k"


def results_of(r: dict, reference: dict | None) -> dict:
    """Arm results recomputed from the time courses; they must equal the archived values."""
    runs = r["lkh6"]["runs"]
    out = {"lkh6": min(best_until(run["improvements"], r["budget_s"]) for run in runs)}
    for arm in MERGE:
        m = r["merges"][arm]
        at_cut = min(x for x in (best_until(run["improvements"], m["cut_s"]) for run in runs) if x is not None)
        eax = best_until(m["eax"]["curve"], m["eax_budget_s"])
        out[arm] = min(x for x in (at_cut, eax) if x is not None)
    c = r["cascade"]
    eax = best_until(c["eax"]["curve"], c["eax_budget_s"])
    out["cascade"] = min(x for x in (c["first_trial_best"], eax) if x is not None)
    assert out == {a: r["results"][a] for a in ARMS}, r["name"]
    out["eax_scratch"] = None
    if reference is not None:
        out["eax_scratch"] = best_until(reference["eax"]["curve"], reference["limit_s"])
        assert out["eax_scratch"] == reference["at_limit"], r["name"]
    return out


def reach_share(r: dict, arm: str, target: int):
    """Share of the budget at which a merge arm first held a tour no longer than `target` (None: never)."""
    m = r["merges"][arm]
    before = [t for run in r["lkh6"]["runs"] for t, c in run["improvements"]
              if c <= target and t <= m["cut_s"] + 1e-9]
    if before:
        return min(before) / r["budget_s"]
    after = [t for t, c in m["eax"]["curve"] if c <= target and t <= m["eax_budget_s"] + 1e-9]
    return (m["cut_s"] + m["prep_s"] + min(after)) / r["budget_s"] if after else None


def describe(values: list, seed: int) -> str:
    x = np.asarray(values, dtype=float)
    wins, losses = int((x < -1e-12).sum()), int((x > 1e-12).sum())
    rng = np.random.default_rng(seed)
    low, high = np.quantile(x[rng.integers(0, len(x), size=(20000, len(x)))].mean(axis=1), [0.025, 0.975])
    return (f"{wins:2d}/{len(x) - wins - losses}/{losses:<2d} mean {x.mean():+.4f}% (95% CI {low:+.4f} to "
            f"{high:+.4f}), median {np.median(x):+.4f}%, sign p = {sign_p(wins, losses):.4f}")


def load(path: str) -> tuple:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        records = [json.loads(z.read(n)) for n in names if n.startswith("instances/") and n.endswith(".json")]
        refs = {}
        for n in names:
            if n.startswith("reference/") and n.endswith(".json"):
                d = json.loads(z.read(n))
                refs[d["name"]] = d
        manifest = json.loads(z.read("manifest.json"))
    return sorted(records, key=lambda r: r["ordinal"]), refs, manifest


def check_tours(path: str, records: list, refs: dict, src: str, tsplib_dir: str | None) -> None:
    sys.path.insert(0, src)
    import tsplab as T
    coords = {}
    for group, n, count, family, seed in SYNTHETIC:
        rng = np.random.default_rng(seed)
        for i in range(count):
            coords[f"p25_{group}_{i:03d}"] = (T.to_int_coords(T.gen_instance(n, family, rng)), "EUC_2D")
    checked = skipped = 0
    with zipfile.ZipFile(path) as z:
        for r in records:
            name = r["name"]
            if name not in coords:
                file = os.path.join(tsplib_dir or "", name + ".tsp")
                if not (tsplib_dir and os.path.exists(file)):
                    skipped += 1
                    continue
                d = T.read_tsplib(file)
                coords[name] = (d["xy"], d["edge_weight_type"])
            xy, ew = coords[name]
            assert hashlib.sha256(np.ascontiguousarray(xy).tobytes()).hexdigest() == r["xy_sha256"], name
            with np.load(io.BytesIO(z.read(f"tours/{name}.npz"))) as arrays:
                for arm in ARMS:
                    tour = arrays[arm].astype(np.int64)
                    assert len(tour) == len(xy) == len(np.unique(tour)), (name, arm)
                    length = int(T.tour_length(xy, tour, ew))
                    assert length == r["final_tour_lengths"][arm] and length <= r["results"][arm], (name, arm)
                    checked += 1
            if name in refs:
                with np.load(io.BytesIO(z.read(f"tours/{name}__eax_scratch.npz"))) as arrays:
                    tour = arrays["eax_scratch"].astype(np.int64)
                    assert len(tour) == len(xy) == len(np.unique(tour)), name
                    assert int(T.tour_length(xy, tour, ew)) == refs[name]["eax"]["final_length"], name
                    checked += 1
    print(f"\nTours: {checked} lengths recomputed, all equal to the archived values; "
          f"{skipped} TSPLIB instances skipped (no file given).")


def main(path: str, tours: list | None = None) -> None:
    records, refs, manifest = load(path)
    values = {r["name"]: results_of(r, refs.get(r["name"])) for r in records}
    hw = manifest["hardware"]
    first_start = datetime.fromtimestamp(min(r["lkh6"]["start_ms"] for r in records) / 1000, timezone.utc)
    stopped = sum(run["stopped_from_outside"] for r in records for run in r["lkh6"]["runs"])
    print(f"Run v25: {len(records)} instances, {len(refs)} reference runs; {hw['gpu']}, {hw['vcpus_visible']} vCPUs, "
          f"{hw['cpu_model']}, {hw['ram_gb']:.0f} GB; code {manifest['code_version']} {manifest['code_sha256'][:12]}, "
          f"notebook build {manifest['notebook_build'][:12]}; sessions: {len({r['session_id'] for r in records})}")
    print(f"First LKH-3 process of `lkh6` started {first_start:%Y-%m-%dT%H:%M:%SZ}; instances finished "
          f"{min(r['finished_utc'] for r in records)} to {max(r['finished_utc'] for r in records)}"
          + (f"; reference runs {min(d['finished_utc'] for d in refs.values())} to "
             f"{max(d['finished_utc'] for d in refs.values())}" if refs else "")
          + f"; LKH-3 processes of `lkh6` stopped from outside: {stopped} of "
          + str(sum(len(r["lkh6"]["runs"]) for r in records)))
    print("All arm results follow from the recorded time courses.")
    groups = [("all", records)] + [(c, [r for r in records if size_class(r) == c]) for c in CLASSES]

    def deltas(rows, a, b):
        return [pct(values[r["name"]][a], values[r["name"]][b]) for r in rows
                if values[r["name"]][a] is not None and values[r["name"]][b] is not None]

    for label, rows in groups:
        print(f"\n{label} ({len(rows)} instances)")
        for i, (key, a, b) in enumerate(PAIRS):
            x = deltas(rows, a, b)
            if x:
                print(f"  {key} {a + ' / ' + b:26s} n={len(x):2d}  {describe(x, 20261125 + i)}")

    small, large = groups[1][1], groups[3][1]
    if small and large:
        d1, d2, d3 = (np.array(deltas(small, PRIMARY, "lkh6")), np.array(deltas(large, PRIMARY, "lkh6")),
                      np.array(deltas(small, PRIMARY, CONTROL)))
        rule = lambda x: (x < 0).sum() > (x > 0).sum() and sign_p(int((x < 0).sum()), int((x > 0).sum())) < 0.05
        print("\nDecision rule of the protocol")
        print(f"  D1 (10,000 nodes: {PRIMARY} shorter than lkh6 on more instances, sign p < 0.05): "
              f"{'holds' if rule(d1) else 'does not hold'}")
        print(f"  D2 (50,000 nodes: shorter on all instances and a larger mean difference than at 10,000 nodes): "
              f"{'holds' if (d2 < 0).all() and abs(d2.mean()) > abs(d1.mean()) else 'does not hold'}")
        print(f"  D3 (10,000 nodes: {PRIMARY} shorter than {CONTROL} on more instances, sign p < 0.05; informative): "
              f"{'holds' if rule(d3) else 'does not hold'}")

    shown = ["lkh6", CONTROL, PRIMARY, "cascade", "eax_scratch"]
    with_opt = [r for r in records if r["optimum"]]
    if with_opt:
        print("\nDistance to the optimum (%)\n  " + f"{'instance':10s}" + "".join(f"{a:>13s}" for a in shown))
        table = [[None if values[r["name"]][a] is None else pct(values[r["name"]][a], r["optimum"]) for a in shown]
                 for r in with_opt]
        for r, row in zip(with_opt, table):
            print(f"  {r['name']:10s}" + "".join(f"{'-':>13s}" if v is None else f"{v:13.4f}" for v in row))
        print(f"  {'mean':10s}" + "".join(f"{np.mean(col):13.4f}" if None not in col else f"{'-':>13s}"
                                         for col in zip(*table)))

    print("\nPer instance: difference from lkh6 (%)\n  " + f"{'instance':16s}{'budget s':>9s}{'lkh6':>12s}"
          + "".join(f"{a:>13s}" for a in shown[1:]))
    for r in records:
        v = values[r["name"]]
        print(f"  {r['name']:16s}{r['budget_s']:9.0f}{v['lkh6']:12d}"
              + "".join(f"{'-':>13s}" if v[a] is None else f"{pct(v[a], v['lkh6']):+13.4f}" for a in shown[1:]))

    avg = lambda items, f: float(np.mean([f(x) for x in items]))
    print("\nBy class (means): LKH-3 between half and all of the budget, GA-EAX in the arms, and two readings chosen "
          "after the results (exploratory)")
    for label, rows in groups[1:]:
        if not rows:
            continue
        lk = [r["lkh6"] for r in rows]
        print(f"  {label}: lkh6 at half of the budget "
              f"{avg(lk, lambda x: pct(x['best_at_fraction'][1], x['best_at_budget'])):+.4f}% longer than at the budget; "
              f"trials per process at the budget {avg(lk, lambda x: x['mean_trials_at_fraction'][-1]):.0f}")
        for arm in MERGE:
            m = [r["merges"][arm] for r in rows]
            union = np.mean([x["population"]["union_edges"] / r["n"] for x, r in zip(m, rows)])
            trials = avg(m, lambda x: np.mean([p["trials_before_cut"] for p in x["per_process"]]))
            print(f"    {arm}: {avg(m, lambda x: x['pop_size']):3.0f} tours, {union:.3f} union edges per node, "
                  f"{trials:3.0f} trials per process before the cut; GA-EAX share "
                  f"{avg(m, lambda x: x['eax_budget_s']):3.0f} s, ended after {avg(m, lambda x: x['eax']['wall_s']):3.0f} s "
                  f"(start-up {avg(m, lambda x: x['eax']['first_line_s']):.1f} s), "
                  f"converged {sum(x['eax']['converged'] for x in m)}/{len(m)}")
        reached = [s for s in (reach_share(r, PRIMARY, values[r["name"]]["lkh6"]) for r in rows) if s is not None]
        cost = avg(rows, lambda r: (6 * r["merges"][PRIMARY]["cut_s"] + r["merges"][PRIMARY]["eax"]["wall_s"])
                   / (6 * r["budget_s"]))
        print(f"    {PRIMARY} held a tour no longer than the final tour of lkh6 on {len(reached)}/{len(rows)} instances, "
              + (f"on average at {np.mean(reached):.2f} of the budget; " if reached else "")
              + f"its process-seconds were {cost:.2f} of those of lkh6")
        c = [r["cascade"] for r in rows]
        print(f"    cascade: {avg(c, lambda x: x['tours']):3.0f} first-trial tours in "
              f"{avg(c, lambda x: x['generation_s']):.0f} s, GA-EAX converged "
              f"{sum(x['eax']['converged'] for x in c)}/{len(c)}")
        ref = [refs[r["name"]] for r in rows if r["name"] in refs]
        if ref:
            print(f"    eax_scratch: limit {avg(ref, lambda x: x['limit_s']):.0f} s, first population after "
                  f"{avg(ref, lambda x: x['eax']['first_line_s']):.0f} s, converged "
                  f"{sum(x['eax']['converged'] for x in ref)}/{len(ref)}, runs in parallel "
                  f"{min(x['parallel_runs'] for x in ref)} to {max(x['parallel_runs'] for x in ref)}")
        base = [values[r["name"]] for r in rows]
        two = [pct(min(v[PRIMARY], v[CONTROL]), v["lkh6"]) for v in base]
        six = [pct(min(v[a] for a in MERGE), v["lkh6"]) for v in base]
        print(f"    exploratory: shorter of {PRIMARY} and {CONTROL} / lkh6 {np.mean(two):+.4f}% "
              f"({sum(x < 0 for x in two)}/{sum(x == 0 for x in two)}/{sum(x > 0 for x in two)}); "
              f"shortest of the six merge arms / lkh6 {np.mean(six):+.4f}%")
    if tours:
        check_tours(path, records, refs, tours[0], tours[1] if len(tours) > 1 else None)


if __name__ == "__main__":
    if len(sys.argv) < 2 or (len(sys.argv) > 2 and (sys.argv[2] != "--tours" or len(sys.argv) not in (4, 5))):
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[3:] if len(sys.argv) > 2 else None)
