"""Recompute the tests and the comparisons of run v29 (confirmation run: staged merging of the tours) from the ZIP.

Usage:
    python analysis/v29_staged_confirmation.py results/v29/v29_staged23_results.zip [--tours SRC [TSPLIB_DIR]]

It reads only the archived records and shares no code with the notebook. The result of every arm is recomputed from
the records and compared with the archived value: `lkh6` is the shortest best tour that the six LKH-3 processes had
recorded at the budget; `trials_50` and `best6_50` are the shorter of the best tour at the cut and the best tour of
their GA-EAX run within its time; `dual_50` is the shorter of the two; `lkhmerge_50` is the shorter of the best tour
at the cut and the best tour that the six merging processes recorded within their time; `staged_50` is the shortest
of the best tour at the cut, the best tour of each stage-1 group within the stage-1 deadline and the best tour of
stage 2 within the budget. The archive stores the time course of every GA-EAX process from its own start, so the
deadline of a stage is known only up to the delay with which its processes started; the script checks that each
archived stage value lies between the best values at the deadline and two seconds before it, and counts how often it
equals the value at the deadline. Percentage difference = 100 x (first arm - second arm) / second arm; negative
values favour the first arm. Counts are instances on which the first arm was shorter / equal / longer. Hypotheses H1
and H2 as in experiments/PROTOCOL_v29.md: exact two-sided sign test (ties excluded) on the 18 synthetic instances,
Holm correction over the two tests; a hypothesis is supported if the first arm is shorter on more instances than it
is longer and the Holm-adjusted p is below 0.05. Intervals: 20,000 bootstrap resamples of instances (the notebook drew
5,000, so the last digit can differ). Everything but H1 and H2, including the TSPLIB instances, is descriptive.

With --tours the script also recomputes the length of every archived tour: SRC is the `src` folder of this repository
(the synthetic instances are regenerated from their seeds) and TSPLIB_DIR, if given, a folder with the TSPLIB files
(`rl11849.tsp`, ...); TSPLIB instances without a file are skipped.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from math import comb

import numpy as np

TRIALS, BEST6, DUAL, MERGE, STAGED = "trials_50", "best6_50", "dual_50", "lkhmerge_50", "staged_50"
ARMS = ["lkh6", TRIALS, BEST6, DUAL, MERGE, STAGED]
TOUR_ARMS = ["lkh6", TRIALS, BEST6, MERGE, STAGED]
HYPOTHESES = [("H1", STAGED, MERGE), ("H2", STAGED, DUAL)]
DESCRIPTIVE = [("D1", STAGED, "lkh6"), ("D2", STAGED, TRIALS), ("D3", DUAL, MERGE), ("D4", TRIALS, BEST6),
               ("D5", DUAL, "lkh6"), ("D6", MERGE, "lkh6")]
GROUPS = ["synthetic", "20k", "uniform 20k", "clustered 20k", "50k", "uniform 50k", "clustered 50k", "tsplib", "all"]
SYNTHETIC = [("uni20k", 20000, 6, "uniform", 13111), ("clu20k", 20000, 6, "clustered", 13212),
             ("uni50k", 50000, 3, "uniform", 13515), ("clu50k", 50000, 3, "clustered", 13616)]
SLACK_S = 2.0


def holm(pvalues: list) -> list:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def sign_p(wins: int, losses: int) -> float:
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def best_until(curve: list, limit: float):
    costs = [c for t, c in curve if t <= limit + 1e-9]
    return min(costs) if costs else None


def in_group(r: dict, group: str) -> bool:
    if group == "all":
        return True
    if group == "tsplib":
        return r["family"] == "tsplib"
    if r["family"] == "tsplib":
        return False
    if group == "synthetic":
        return True
    family, size = group.split() if " " in group else (None, group)
    return round(r["n"] / 1000) == int(size[:-1]) and family in (None, r["family"])


def stage_value(curve: list, limit: float, stored, counts: dict):
    """The archived value of a stage process must lie between its best values at the deadline and SLACK_S before."""
    late, early = best_until(curve, limit), best_until(curve, limit - SLACK_S)
    if stored is None:
        assert early is None, "a stage value is missing although the process had a tour before its deadline"
        return None
    assert late is not None and late <= stored and (early is None or stored <= early), (stored, late, early)
    counts["equal"] += int(stored == late)
    counts["all"] += 1
    return stored


def results_of(r: dict, counts: dict) -> dict:
    """Arm results recomputed from the records; they must equal the archived values."""
    runs, dual, merge, staged = r["lkh6"]["runs"], r["dual"], r["lkhmerge"], r["staged"]
    out = {"lkh6": min(best_until(run["improvements"], r["budget_s"]) for run in runs)}
    cut = dual["arms"][TRIALS]["cut_s"]
    at_cut = min(x for x in (best_until(run["improvements"], cut) for run in runs) if x is not None)
    assert at_cut == dual["best_at_cut"] == merge["best_at_cut"] == staged["best_at_cut"], r["name"]
    assert merge["cut_s"] == cut == staged["cut_s"], r["name"]
    for arm in (TRIALS, BEST6):
        a = dual["arms"][arm]
        eax = None if a["eax"] is None else best_until(a["eax"]["curve"], a["eax_budget_s"])
        out[arm] = min(x for x in (at_cut, eax) if x is not None)
    out[DUAL] = min(out[TRIALS], out[BEST6])
    found = [best_until(run["improvements"], merge["lkh_budget_s"]) for run in merge["runs"]]
    out[MERGE] = min([at_cut] + [x for x in found if x is not None])
    groups = staged["stage1"]
    assert len(groups) == staged["groups"] and sum(g["size"] for g in groups) == staged["pop_size"] - staged["best_tours"]
    assert staged["stage2"]["pop_size"] == len(groups) + staged["best_tours"], r["name"]
    values = [at_cut]
    limit1 = staged["stage1_share"] * staged["budget_s"]
    for g in groups:
        if g["eax"] is None:
            assert g["distinct"] < 2 and g["at_deadline"] == g["best_input"], r["name"]
            values.append(g["at_deadline"])
        else:
            v = stage_value(g["eax"]["curve"], limit1, g["at_deadline"], counts)
            if v is not None:
                values.append(v)
    s2 = staged["stage2"]
    if not s2["skipped"]:
        v = stage_value(s2["eax"]["curve"], s2["budget_s"], s2["at_deadline"], counts)
        if v is not None:
            values.append(v)
    out[STAGED] = min(values)
    assert out == {a: r["results"][a] for a in ARMS}, r["name"]
    out["at_cut"] = at_cut
    return out


def process_seconds(r: dict) -> dict:
    """Process-seconds after the cut, from the wall times of the processes."""
    dual = sum(a["eax"]["wall_s"] for a in r["dual"]["arms"].values() if a["eax"] is not None)
    merge = sum(run["wall_s"] for run in r["lkhmerge"]["runs"])
    s = r["staged"]
    staged = sum(g["eax"]["wall_s"] for g in s["stage1"] if g["eax"] is not None) + \
        (0.0 if s["stage2"]["eax"] is None else s["stage2"]["eax"]["wall_s"])
    return {DUAL: dual, MERGE: merge, STAGED: staged}


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
        manifest = json.loads(z.read("manifest.json"))
    return sorted(records, key=lambda r: r["ordinal"]), manifest


def check_tours(path: str, records: list, src: str, tsplib_dir: str | None) -> None:
    import os
    sys.path.insert(0, src)
    import tsplab as T
    coords = {}
    for group, n, count, family, seed in SYNTHETIC:
        rng = np.random.default_rng(seed)
        for i in range(count):
            coords[f"s29_{group}_{i:03d}"] = T.to_int_coords(T.gen_instance(n, family, rng))
    checked, skipped = 0, []
    with zipfile.ZipFile(path) as z:
        for r in records:
            name, n = r["name"], r["n"]
            if name not in coords:
                file = os.path.join(tsplib_dir or "", name + ".tsp")
                if not (tsplib_dir and os.path.exists(file)):
                    skipped.append(name)
                    continue
                coords[name] = T.read_tsplib(file)["xy"]
            xy = coords[name]
            assert hashlib.sha256(np.ascontiguousarray(xy).tobytes()).hexdigest() == r["xy_sha256"], name
            with np.load(io.BytesIO(z.read(f"tours/{name}.npz"))) as arrays:
                assert sorted(arrays.files) == sorted(TOUR_ARMS), name
                for arm in TOUR_ARMS:
                    tour = arrays[arm].astype(np.int64)
                    assert len(tour) == n == len(np.unique(tour)) and tour.min() == 0 and tour.max() == n - 1, (name, arm)
                    length = int(T.tour_length(xy, tour, r["ew"]))
                    assert length == r["final_tour_lengths"][arm] and length <= r["results"][arm], (name, arm)
                    checked += 1
    print(f"\nTours: {checked} lengths recomputed, all equal to the archived values"
          + (f"; skipped (no file given): {', '.join(skipped)}." if skipped else "."))


def main(path: str, src: str | None = None, tsplib_dir: str | None = None) -> None:
    records, manifest = load(path)
    counts = {"equal": 0, "all": 0}
    values = {r["name"]: results_of(r, counts) for r in records}
    hw, proto = manifest["hardware"], manifest["protocol"]
    start = datetime.fromtimestamp(min(r["lkh6"]["start_ms"] for r in records) / 1000, timezone.utc)
    stopped = sum(run["stopped_from_outside"] for r in records for run in r["lkh6"]["runs"])
    print(f"Run v29: {len(records)} instances; {hw['gpu']} (not used), {hw['vcpus_visible']} vCPUs, {hw['cpu_model']}, "
          f"{hw['ram_gb']:.0f} GB; code {manifest['code_version']} {manifest['code_sha256'][:12]}, notebook build "
          f"{manifest['notebook_build'][:12]}; test mode {manifest['test_mode']}; sessions: "
          f"{len({r['session_id'] for r in records})}")
    print(f"Protocol {proto['path']} (SHA-256 {proto['sha256'][:12]}) found on GitHub in commit "
          f"{(proto['protocol_commit'] or '-')[:10]} of {proto['protocol_commit_time']}, checked {proto['checked_utc']}; "
          f"first LKH-3 process started {start:%Y-%m-%dT%H:%M:%SZ}; instances finished "
          f"{min(r['finished_utc'] for r in records)} to {max(r['finished_utc'] for r in records)}; LKH-3 processes of "
          f"lkh6 stopped from outside: {stopped} of {sum(len(r['lkh6']['runs']) for r in records)}")
    print(f"All arm results follow from the records; archived stage values equal to the value at the deadline: "
          f"{counts['equal']} of {counts['all']} (the others within {SLACK_S:.0f} s before it).")

    def deltas(group, a, b):
        return [pct(values[r["name"]][a], values[r["name"]][b]) for r in records if in_group(r, group)]

    print("\nHypotheses of the protocol (18 synthetic instances; Holm over H1 and H2)")
    found = [summary(deltas("synthetic", a, b), 20261122 + i) for i, (key, a, b) in enumerate(HYPOTHESES)]
    verdicts = {}
    for (key, a, b), s, p_adj in zip(HYPOTHESES, found, holm([s["p"] for s in found])):
        verdicts[key] = s["win"] > s["loss"] and p_adj < 0.05
        print(f"  {key} {a + ' / ' + b:26s} n={s['n']:2d}  {text(s)}, Holm p = {p_adj:.2g}: "
              f"{'supported' if verdicts[key] else 'not supported'}")
    print(f"  Staged merging confirmed (H1 and H2): {verdicts['H1'] and verdicts['H2']}")

    for group in GROUPS:
        rows = [r for r in records if in_group(r, group)]
        print(f"\n{group} ({len(rows)} instances)")
        for i, (key, a, b) in enumerate(HYPOTHESES + DESCRIPTIVE):
            print(f"  {key} {a + ' / ' + b:26s} n={len(rows):2d}  {text(summary(deltas(group, a, b), 20261200 + i))}")

    print("\nPer instance: difference from lkh6 (%)\n  " + f"{'instance':16s}{'lkh6':>12s}"
          + "".join(f"{a:>13s}" for a in ARMS[1:]) + "   source of dual_50")
    for r in records:
        v = values[r["name"]]
        print(f"  {r['name']:16s}{v['lkh6']:12d}" + "".join(f"{pct(v[a], v['lkh6']):+13.4f}" for a in ARMS[1:])
              + "   " + "+".join(sorted(r["dual"]["source"])))

    avg = lambda items, f: float(np.mean([f(x) for x in items]))
    print("\nBy size (means; descriptive)")
    for size in ("20k", "50k", "tsplib"):
        rows = [r for r in records if in_group(r, size)]
        st = [r["staged"] for r in rows]
        conv = lambda arm: sum(bool(r["dual"]["arms"][arm]["eax"] and r["dual"]["arms"][arm]["eax"]["converged"]) for r in rows)
        g_conv = sum(bool(g["eax"] and g["eax"]["converged"]) for s in st for g in s["stage1"])
        g_ran = sum(g["eax"] is not None for s in st for g in s["stage1"])
        s2_conv = sum(bool(s["stage2"]["eax"] and s["stage2"]["eax"]["converged"]) for s in st)
        source = {}
        for r in rows:
            key = "+".join(sorted(r["dual"]["source"]))
            source[key] = source.get(key, 0) + 1
        ps = [process_seconds(r) for r in rows]
        print(f"  {size}: time left after the cut {avg(st, lambda s: s['budget_s']):.0f} s; staged_50 used "
              f"{avg(st, lambda s: s['used_s']):.0f} s ({min(s['used_s'] / s['budget_s'] for s in st):.2f} to "
              f"{max(s['used_s'] / s['budget_s'] for s in st):.2f} of it), stage 1 {avg(st, lambda s: s['stage1_wall_s']):.0f} s, "
              f"stage 2 {avg(st, lambda s: s['stage2']['eax']['wall_s'] if s['stage2']['eax'] else 0.0):.0f} s")
        print(f"    GA-EAX converged: stage-1 groups {g_conv}/{g_ran}, stage 2 {s2_conv}/{len(st)}, 102 tours {conv(TRIALS)}/{len(rows)}, "
              f"six best tours {conv(BEST6)}/{len(rows)}; dual_50 came from {source}; staged_50 came from "
              f"{ {k: sum(s['source'] == k for s in st) for k in sorted({s['source'] for s in st})} }")
        print(f"    process-seconds after the cut: dual_50 {avg(ps, lambda p: p[DUAL]):.0f}, lkhmerge_50 {avg(ps, lambda p: p[MERGE]):.0f}, "
              f"staged_50 {avg(ps, lambda p: p[STAGED]):.0f}; best tour at the cut {avg(rows, lambda r: pct(r['staged']['best_at_cut'], values[r['name']]['lkh6'])):+.4f}% "
              f"from lkh6")
    tsp = [r for r in records if r["family"] == "tsplib" and r["optimum"]]
    if tsp:
        print("\nDistance to the optimum (%), TSPLIB\n  " + f"{'instance':10s}" + "".join(f"{a:>13s}" for a in ARMS))
        table = [[pct(values[r["name"]][a], r["optimum"]) for a in ARMS] for r in tsp]
        for r, row in zip(tsp, table):
            print(f"  {r['name']:10s}" + "".join(f"{v:13.4f}" for v in row))
        print(f"  {'mean':10s}" + "".join(f"{np.mean(col):13.4f}" for col in zip(*table)))
    if src:
        check_tours(path, records, src, tsplib_dir)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 4, 5) or (len(sys.argv) > 2 and sys.argv[2] != "--tours"):
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else None, sys.argv[4] if len(sys.argv) > 4 else None)
