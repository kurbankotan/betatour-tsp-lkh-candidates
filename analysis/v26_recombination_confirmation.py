"""Recompute the four tests and the comparisons of run v26 (confirmation run on recombining LKH-3's tours) from the ZIP.

Usage:
    python analysis/v26_recombination_confirmation.py results/v26/v26_recomb38_results.zip [--tours SRC [TSPLIB_DIR]]

It reads only the archived records and shares no code with the notebook. The result of every arm is recomputed
from the recorded time courses (wall-clock seconds from the start of the arm) and compared with the archived value:
`lkh6` is the shortest best tour that the six LKH-3 processes had recorded at the budget; `trials_50` and `best6_50`
are the shorter of the best tour at the cut and the best tour of their GA-EAX run within its time (the tour at the cut
when GA-EAX was not run); `dual_50` is the shorter of the two; `lkhmerge_50` is the shorter of the best tour at the
cut and the best tour that the six merging processes recorded within their time; `cascade` is the shorter of the best
first-trial tour and the best tour of its GA-EAX run within its time; `eax_scratch` (a reference, not an arm) is the
best tour of GA-EAX from scratch within six times the budget. Percentage difference = 100 x (first arm - second arm)
/ second arm; negative values favour the first arm. Counts are instances on which the first arm was shorter / equal /
longer. Tests H1 to H4 as in experiments/PROTOCOL_v26.md: exact two-sided sign test (ties excluded), Holm correction
over the four tests. Intervals: 20,000 bootstrap resamples of instances (the notebook drew 5,000, so the last digit
can differ). Everything but H1 to H4 is exploratory.

With --tours the script also recomputes the length of every archived tour: SRC is the `src` folder of this
repository (the synthetic instances are regenerated from their seeds) and TSPLIB_DIR, if given, a folder with the
TSPLIB files (`pcb3038.tsp`, ...); TSPLIB instances without a file are skipped.
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

TRIALS, BEST6, DUAL, MERGE = "trials_50", "best6_50", "dual_50", "lkhmerge_50"
ARMS = ["lkh6", TRIALS, BEST6, DUAL, MERGE, "cascade"]
TOUR_ARMS = ["lkh6", TRIALS, BEST6, MERGE, "cascade"]
TESTS = [("H1", DUAL, "lkh6", "synthetic"), ("H2", DUAL, MERGE, "synthetic"), ("H3", DUAL, "cascade", "synthetic"),
         ("H4", TRIALS, BEST6, "10k")]
PAIRS = [(k, a, b) for k, a, b, _ in TESTS] + [
    ("S1", TRIALS, "lkh6"), ("S2", BEST6, "lkh6"), ("S3", MERGE, "lkh6"), ("S4", "cascade", "lkh6"),
    ("S5", BEST6, MERGE), ("R1", "eax_scratch", "lkh6"), ("R2", DUAL, "eax_scratch")]
GROUPS = ["synthetic", "10k", "20k", "50k", "tsplib_new", "tsplib_pilot"]
SYNTHETIC = [("uni10k", 10000, 8, "uniform", 10111), ("clu10k", 10000, 8, "clustered", 10212),
             ("uni20k", 20000, 3, "uniform", 10313), ("clu20k", 20000, 3, "clustered", 10414),
             ("uni50k", 50000, 3, "uniform", 10515), ("clu50k", 50000, 3, "clustered", 10616)]


def sign_p(wins: int, losses: int) -> float:
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def holm(pvalues: list) -> list:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def best_until(curve: list, limit: float):
    costs = [c for t, c in curve if t <= limit + 1e-9]
    return min(costs) if costs else None


def in_group(r: dict, group: str) -> bool:
    if group == "synthetic":
        return r["family"] != "tsplib"
    if group.endswith("k") and group[:-1].isdigit():
        return r["family"] != "tsplib" and round(r["n"] / 1000) == int(group[:-1])
    return r["group"] == group


def results_of(r: dict, reference: dict | None) -> dict:
    """Arm results recomputed from the time courses; they must equal the archived values."""
    runs, dual, merge, cascade = r["lkh6"]["runs"], r["dual"], r["lkhmerge"], r["cascade"]
    out = {"lkh6": min(best_until(run["improvements"], r["budget_s"]) for run in runs)}
    cut = dual["arms"][TRIALS]["cut_s"]
    at_cut = min(x for x in (best_until(run["improvements"], cut) for run in runs) if x is not None)
    assert at_cut == dual["best_at_cut"] == merge["best_at_cut"] and merge["cut_s"] == cut, r["name"]
    for arm in (TRIALS, BEST6):
        a = dual["arms"][arm]
        eax = None if a["eax"] is None else best_until(a["eax"]["curve"], a["eax_budget_s"])
        out[arm] = min(x for x in (at_cut, eax) if x is not None)
    out[DUAL] = min(out[TRIALS], out[BEST6])
    found = [best_until(run["improvements"], merge["lkh_budget_s"]) for run in merge["runs"]]
    out[MERGE] = min([at_cut] + [x for x in found if x is not None])
    eax = None if cascade["eax"] is None else best_until(cascade["eax"]["curve"], cascade["eax_budget_s"])
    out["cascade"] = min(x for x in (cascade["first_trial_best"], eax) if x is not None)
    assert out == {a: r["results"][a] for a in ARMS}, r["name"]
    out["eax_scratch"] = None
    if reference is not None:
        out["eax_scratch"] = best_until(reference["eax"]["curve"], reference["limit_s"])
        assert out["eax_scratch"] == reference["at_limit"], r["name"]
    return out


def summary(values: list, seed: int) -> dict:
    x = np.asarray(values, dtype=float)
    wins, losses = int((x < -1e-12).sum()), int((x > 1e-12).sum())
    rng = np.random.default_rng(seed)
    low, high = np.quantile(x[rng.integers(0, len(x), size=(20000, len(x)))].mean(axis=1), [0.025, 0.975])
    return dict(n=len(x), win=wins, tie=len(x) - wins - losses, loss=losses, mean=float(x.mean()),
                median=float(np.median(x)), low=float(low), high=float(high), p=sign_p(wins, losses))


def text(s: dict) -> str:
    return (f"{s['win']:2d}/{s['tie']}/{s['loss']:<2d} mean {s['mean']:+.4f}% (95% CI {s['low']:+.4f} to {s['high']:+.4f}), "
            f"median {s['median']:+.4f}%, sign p = {s['p']:.2g}")


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
            coords[f"c26_{group}_{i:03d}"] = (T.to_int_coords(T.gen_instance(n, family, rng)), "EUC_2D")
    checked, skipped = 0, []
    with zipfile.ZipFile(path) as z:
        for r in records:
            name = r["name"]
            if name not in coords:
                file = os.path.join(tsplib_dir or "", name + ".tsp")
                if not (tsplib_dir and os.path.exists(file)):
                    skipped.append(name)
                    continue
                d = T.read_tsplib(file)
                coords[name] = (d["xy"], d["edge_weight_type"])
            xy, ew = coords[name]
            assert hashlib.sha256(np.ascontiguousarray(xy).tobytes()).hexdigest() == r["xy_sha256"], name
            with np.load(io.BytesIO(z.read(f"tours/{name}.npz"))) as arrays:
                assert sorted(arrays.files) == sorted(TOUR_ARMS), name
                for arm in TOUR_ARMS:
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
    print(f"\nTours: {checked} lengths recomputed, all equal to the archived values"
          + (f"; skipped (no file given): {', '.join(skipped)}." if skipped else "."))


def main(path: str, tours: list | None = None) -> None:
    records, refs, manifest = load(path)
    values = {r["name"]: results_of(r, refs.get(r["name"])) for r in records}
    hw, proto = manifest["hardware"], manifest["protocol"]
    start = datetime.fromtimestamp(min(r["lkh6"]["start_ms"] for r in records) / 1000, timezone.utc)
    stopped = sum(run["stopped_from_outside"] for r in records for run in r["lkh6"]["runs"])
    print(f"Run v26: {len(records)} instances, {len(refs)} reference runs; {hw['gpu']}, {hw['vcpus_visible']} vCPUs, "
          f"{hw['cpu_model']}, {hw['ram_gb']:.0f} GB; code {manifest['code_version']} {manifest['code_sha256'][:12]}, "
          f"notebook build {manifest['notebook_build'][:12]}; sessions: "
          f"{len({r['session_id'] for r in records} | {d['session_id'] for d in refs.values()})}")
    print(f"Protocol {proto['path']} (SHA-256 {proto['sha256'][:12]}) found on GitHub in commit "
          f"{(proto['protocol_commit'] or '-')[:10]} of {proto['protocol_commit_time']}, checked {proto['checked_utc']}; "
          f"first LKH-3 process started {start:%Y-%m-%dT%H:%M:%SZ}; instances finished "
          f"{min(r['finished_utc'] for r in records)} to {max(r['finished_utc'] for r in records)}; reference runs to "
          f"{max(d['finished_utc'] for d in refs.values())}; LKH-3 processes of lkh6 stopped from outside: {stopped} of "
          f"{sum(len(r['lkh6']['runs']) for r in records)}")
    print("All arm results follow from the recorded time courses.")

    def deltas(group, a, b):
        return [pct(values[r["name"]][a], values[r["name"]][b]) for r in records if in_group(r, group)
                and values[r["name"]][a] is not None and values[r["name"]][b] is not None]

    found = [summary(deltas(group, a, b), 20261121 + i) for i, (key, a, b, group) in enumerate(TESTS)]
    adjusted = holm([s["p"] for s in found])
    print("\nTests of the protocol (Holm over H1 to H4)")
    for (key, a, b, group), s, p_adj in zip(TESTS, found, adjusted):
        verdict = "supported" if s["win"] > s["loss"] and p_adj < 0.05 else "not supported"
        print(f"  {key} {a + ' / ' + b:26s} {group:9s} n={s['n']:2d}  {text(s)}, Holm p = {p_adj:.2g}: {verdict}")

    for group in GROUPS:
        rows = [r for r in records if in_group(r, group)]
        print(f"\n{group} ({len(rows)} instances)")
        for i, (key, a, b) in enumerate(PAIRS):
            x = deltas(group, a, b)
            if x:
                print(f"  {key} {a + ' / ' + b:26s} n={len(x):2d}  {text(summary(x, 20261200 + i))}")

    shown = ["lkh6", BEST6, TRIALS, DUAL, MERGE, "cascade", "eax_scratch"]
    for group in ("tsplib_new", "tsplib_pilot"):
        rows = [r for r in records if r["group"] == group]
        print(f"\nDistance to the optimum (%), {group}\n  " + f"{'instance':10s}" + "".join(f"{a:>13s}" for a in shown))
        table = [[None if values[r["name"]][a] is None else pct(values[r["name"]][a], r["optimum"]) for a in shown]
                 for r in rows]
        for r, row in zip(rows, table):
            print(f"  {r['name']:10s}" + "".join(f"{'-':>13s}" if v is None else f"{v:13.4f}" for v in row))
        print(f"  {'mean':10s}" + "".join(f"{np.mean(col):13.4f}" if None not in col else f"{'-':>13s}"
                                         for col in zip(*table)))

    print("\nPer instance: difference from lkh6 (%)\n  " + f"{'instance':16s}{'lkh6':>12s}"
          + "".join(f"{a:>13s}" for a in shown[1:]) + "   source of dual_50")
    for r in records:
        v = values[r["name"]]
        print(f"  {r['name']:16s}{v['lkh6']:12d}"
              + "".join(f"{'-':>13s}" if v[a] is None else f"{pct(v[a], v['lkh6']):+13.4f}" for a in shown[1:])
              + "   " + "+".join(sorted(r["dual"]["source"])))

    avg = lambda items, f: float(np.mean([f(x) for x in items]))
    print("\nBy class (means; exploratory)")
    for group in GROUPS[1:]:
        rows = [r for r in records if in_group(r, group)]
        if not rows:
            continue
        arms = [(r["dual"]["arms"][TRIALS], r["dual"]["arms"][BEST6]) for r in rows]
        merges = [r["lkhmerge"] for r in rows]
        conv = lambda k: (sum(bool(a[k]["eax"] and a[k]["eax"]["converged"]) for a in arms),
                          sum(a[k]["eax"] is not None for a in arms))
        source = {}
        for r in rows:
            key = "+".join(sorted(r["dual"]["source"]))
            source[key] = source.get(key, 0) + 1
        gains = [pct(m["at_deadline"], m["best_at_cut"]) for m in merges if m["at_deadline"] is not None]
        print(f"  {group}: lkh6 at half of the budget "
              f"{avg(rows, lambda r: pct(r['lkh6']['best_at_fraction'][1], r['lkh6']['best_at_budget'])):+.4f}% longer "
              f"than at the budget; trials per process before the cut "
              f"{avg(arms, lambda a: np.mean([p['trials_before_cut'] for p in a[0]['per_process']])):.0f}; union edges "
              f"per node {avg(rows, lambda r: r['dual']['arms'][TRIALS]['population']['union_edges'] / r['n']):.3f} "
              f"(102 tours) and {avg(rows, lambda r: r['dual']['arms'][BEST6]['population']['union_edges'] / r['n']):.3f} "
              f"(6 tours)")
        print(f"    GA-EAX converged on the 102 tours {conv(0)[0]}/{conv(0)[1]}, on the six tours "
              f"{conv(1)[0]}/{conv(1)[1]}; dual_50 came from {source}")
        ran = [m for m in merges if not m["skipped"]]
        trials = [x["trials"] for m in ran for x in m["runs"]]
        print(f"    lkhmerge_50: merging ran on {len(ran)}/{len(merges)} instances, processes with a tour in time "
              f"{avg(ran, lambda m: m['processes_with_tour']) if ran else 0:.1f}, trials per process "
              f"{np.mean(trials) if trials else 0:.0f}, gain over the tour at the cut {np.mean(gains) if gains else 0:+.4f}%, "
              f"common edges kept on {sum(bool(m['keeps_common_edges']) for m in ran)}/{len(ran)}")
    if tours:
        check_tours(path, records, refs, tours[0], tours[1] if len(tours) > 1 else None)


if __name__ == "__main__":
    if len(sys.argv) < 2 or (len(sys.argv) > 2 and (sys.argv[2] != "--tours" or len(sys.argv) not in (4, 5))):
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[3:] if len(sys.argv) > 2 else None)
