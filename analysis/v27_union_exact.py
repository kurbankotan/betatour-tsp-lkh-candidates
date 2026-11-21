"""Recompute the readings and the comparisons of run v27 (pilot: exact merging on the union graph of the tours) from the ZIP.

Usage:
    python analysis/v27_union_exact.py results/v27/v27_union14_results.zip [--tours SRC]

It reads only the archived records and shares no code with the notebook. The result of every arm is recomputed from
the records and compared with the archived value: `lkh6` is the shortest best tour that the six LKH-3 processes had
recorded at the budget; `trials_50` and `best6_50` are the shorter of the best tour at the cut and the best tour of
their GA-EAX run within its time; `dual_50` is the shorter of the two; `lkhmerge_50` is the shorter of the best tour
at the cut and the best tour that the six merging processes recorded within their time; each exact arm is the tour
that SCIP returned, unless SCIP found it more than one second after the arm's deadline (then the best tour at the
cut counts). Percentage difference = 100 x (first arm - second arm) / second arm; negative values favour the first
arm. Counts are instances on which the first arm was shorter / equal / longer. Readings R1 to R4 as in
experiments/PROTOCOL_v27.md: exact two-sided sign test (ties excluded) on the 12 instances of 10,000 nodes, no
correction; a reading holds if the first arm is shorter on more instances than it is longer and p < 0.05. Intervals:
20,000 bootstrap resamples of instances (the notebook drew 5,000, so the last digit can differ). Everything but R1
to R4 is descriptive.

The shortest tour of the union graph of the six tours at the cut is known on an instance if SCIP proved `exact6_50`
optimal, or if it proved one of the two larger graphs optimal with a tour that uses no edge outside the union.

With --tours the script also recomputes, on the instances regenerated from their seeds (SRC is the `src` folder of
this repository), the length of every archived tour and of the six tours at the cut, the union graph of these six
tours, the edges of every tour outside it, and checks that no archived tour inside the union graph is shorter than a
tour that SCIP proved optimal on a graph that contains the union.
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

TRIALS, BEST6, DUAL, MERGE = "trials_50", "best6_50", "dual_50", "lkhmerge_50"
EXACT, BETA, NEAR = "exact6_50", "exact6beta_50", "exact6near_50"
EXACT_ARMS = [EXACT, BETA, NEAR]
ARMS = ["lkh6", TRIALS, BEST6, DUAL, MERGE] + EXACT_ARMS
TOUR_ARMS = ["lkh6", TRIALS, BEST6, MERGE] + EXACT_ARMS
READINGS = [("R1", EXACT, BEST6), ("R2", EXACT, DUAL), ("R3", BETA, EXACT), ("R4", BETA, NEAR)]
DESCRIPTIVE = [("D1", EXACT, MERGE), ("D2", EXACT, "lkh6"), ("D3", NEAR, EXACT), ("D4", DUAL, MERGE),
               ("D5", BEST6, MERGE), ("D6", DUAL, "lkh6")]
GROUPS = ["10k", "uniform 10k", "clustered 10k", "20k", "all"]
SYNTHETIC = [("uni10k", 10000, 6, "uniform", 11111), ("clu10k", 10000, 6, "clustered", 11212),
             ("uni20k", 20000, 1, "uniform", 11313), ("clu20k", 20000, 1, "clustered", 11414)]


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
    family, size = group.split() if " " in group else (None, group)
    return round(r["n"] / 1000) == int(size[:-1]) and family in (None, r["family"])


def results_of(r: dict) -> dict:
    """Arm results recomputed from the records; they must equal the archived values."""
    runs, dual, merge, exact = r["lkh6"]["runs"], r["dual"], r["lkhmerge"], r["exact"]
    out = {"lkh6": min(best_until(run["improvements"], r["budget_s"]) for run in runs)}
    cut = dual["arms"][TRIALS]["cut_s"]
    at_cut = min(x for x in (best_until(run["improvements"], cut) for run in runs) if x is not None)
    assert at_cut == dual["best_at_cut"] == merge["best_at_cut"] == exact["best_at_cut"], r["name"]
    assert merge["cut_s"] == cut and all(exact["arms"][a]["cut_s"] == cut for a in EXACT_ARMS), r["name"]
    for arm in (TRIALS, BEST6):
        a = dual["arms"][arm]
        eax = None if a["eax"] is None else best_until(a["eax"]["curve"], a["eax_budget_s"])
        out[arm] = min(x for x in (at_cut, eax) if x is not None)
    out[DUAL] = min(out[TRIALS], out[BEST6])
    found = [best_until(run["improvements"], merge["lkh_budget_s"]) for run in merge["runs"]]
    out[MERGE] = min([at_cut] + [x for x in found if x is not None])
    for arm in EXACT_ARMS:
        a, s = exact["arms"][arm], exact["arms"][arm]["solver"]
        assert s["start_length"] == at_cut and s["length"] <= at_cut and s["optimal"] == (s["status"] == "optimal"), \
            (r["name"], arm)
        deadline = s["finished_epoch"] - a["finished_after_deadline_s"]
        late = s["solve_start_epoch"] + s["time_to_best_s"] > deadline + 1.0 and s["length"] < at_cut
        assert late == a["best_found_late"], (r["name"], arm)
        assert a["outside_union"] == a["extra_used"] and (arm != EXACT or a["extra_edges"] == 0), (r["name"], arm)
        out[arm] = at_cut if late else s["length"]
    assert out == {a: r["results"][a] for a in ARMS}, r["name"]
    out["at_cut"] = at_cut
    return out


def union_optimum(r: dict):
    """The length of the shortest tour of the six-tour union graph if a record proves it, else None."""
    arms = r["exact"]["arms"]
    if arms[EXACT]["solver"]["optimal"]:
        return arms[EXACT]["solver"]["length"], EXACT
    for arm in (BETA, NEAR):
        if arms[arm]["solver"]["optimal"] and arms[arm]["outside_union"] == 0:
            return arms[arm]["solver"]["length"], arm
    return None, None


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


def edge_keys(tour: np.ndarray, n: int) -> np.ndarray:
    a, b = tour, np.roll(tour, -1)
    return np.unique(np.minimum(a, b) * n + np.maximum(a, b))


def check_tours(path: str, records: list, src: str) -> None:
    sys.path.insert(0, src)
    import tsplab as T
    coords = {}
    for group, n, count, family, seed in SYNTHETIC:
        rng = np.random.default_rng(seed)
        for i in range(count):
            coords[f"u27_{group}_{i:03d}"] = T.to_int_coords(T.gen_instance(n, family, rng))
    checked = 0
    with zipfile.ZipFile(path) as z:
        for r in records:
            name, n, xy = r["name"], r["n"], coords[r["name"]]
            assert hashlib.sha256(np.ascontiguousarray(xy).tobytes()).hexdigest() == r["xy_sha256"], name
            with np.load(io.BytesIO(z.read(f"tours/{name}.npz"))) as arrays:
                assert sorted(arrays.files) == sorted(TOUR_ARMS + ["cut_tours"]), name
                tours = {arm: arrays[arm].astype(np.int64) for arm in TOUR_ARMS}
                cut_tours = arrays["cut_tours"].astype(np.int64)
            lengths = {}
            for arm, tour in tours.items():
                assert len(tour) == n == len(np.unique(tour)) and tour.min() == 0 and tour.max() == n - 1, (name, arm)
                lengths[arm] = int(T.tour_length(xy, tour, r["ew"]))
                assert lengths[arm] == r["final_tour_lengths"][arm] and lengths[arm] <= r["results"][arm], (name, arm)
                checked += 1
            assert cut_tours.shape == (6, n), name
            cut_lengths = []
            for tour in cut_tours:
                assert len(np.unique(tour)) == n and tour.min() == 0 and tour.max() == n - 1, name
                cut_lengths.append(int(T.tour_length(xy, tour, r["ew"])))
                checked += 1
            assert min(cut_lengths) == r["exact"]["best_at_cut"], name
            union = np.unique(np.concatenate([edge_keys(t, n) for t in cut_tours]))
            assert len(union) == r["exact"]["union_edges"] == r["exact"]["pairwise"]["union_edges"], name
            inside = {}
            for arm, tour in tours.items():
                outside = int((~np.isin(edge_keys(tour, n), union)).sum())
                assert outside == r["edges_outside_union6"][arm], (name, arm)
                inside[arm] = outside == 0
            assert inside[EXACT], name
            # A tour that SCIP proved optimal on a graph that contains the union is not longer than any tour inside it.
            union_tours = [lengths[arm] for arm in TOUR_ARMS if inside[arm]] + cut_lengths
            for arm in EXACT_ARMS:
                if r["exact"]["arms"][arm]["solver"]["optimal"]:
                    assert lengths[arm] <= min(union_tours), (name, arm)
    print(f"\nTours: {checked} lengths recomputed (the seven arms and the six tours at the cut), all equal to the "
          f"archived values; the union graphs, the edges outside them and the proven optima are consistent.")


def main(path: str, src: str | None = None) -> None:
    records, manifest = load(path)
    values = {r["name"]: results_of(r) for r in records}
    hw, proto = manifest["hardware"], manifest["protocol"]
    start = datetime.fromtimestamp(min(r["lkh6"]["start_ms"] for r in records) / 1000, timezone.utc)
    stopped = sum(run["stopped_from_outside"] for r in records for run in r["lkh6"]["runs"])
    print(f"Run v27: {len(records)} instances; {hw['gpu']}, {hw['vcpus_visible']} vCPUs, {hw['cpu_model']}, "
          f"{hw['ram_gb']:.0f} GB; PySCIPOpt {hw['pyscipopt']}, SCIP {hw['scip']}; code {manifest['code_version']} "
          f"{manifest['code_sha256'][:12]}, {manifest['union_version']} {manifest['union_sha256'][:12]}, notebook build "
          f"{manifest['notebook_build'][:12]}; test mode {manifest['test_mode']}; sessions: "
          f"{len({r['session_id'] for r in records})}")
    print(f"Protocol {proto['path']} (SHA-256 {proto['sha256'][:12]}) found on GitHub in commit "
          f"{(proto['protocol_commit'] or '-')[:10]} of {proto['protocol_commit_time']}, checked {proto['checked_utc']}; "
          f"first LKH-3 process started {start:%Y-%m-%dT%H:%M:%SZ}; instances finished "
          f"{min(r['finished_utc'] for r in records)} to {max(r['finished_utc'] for r in records)}; LKH-3 processes of "
          f"lkh6 stopped from outside: {stopped} of {sum(len(r['lkh6']['runs']) for r in records)}")
    print("All arm results follow from the records.")

    def deltas(group, a, b):
        return [pct(values[r["name"]][a], values[r["name"]][b]) for r in records if in_group(r, group)]

    print("\nReadings of the protocol (12 instances of 10,000 nodes; no correction)")
    verdicts = {}
    for i, (key, a, b) in enumerate(READINGS):
        s = summary(deltas("10k", a, b), 20261121 + i)
        verdicts[key] = s["win"] > s["loss"] and s["p"] < 0.05
        print(f"  {key} {a + ' / ' + b:30s} n={s['n']:2d}  {text(s)}: {'holds' if verdicts[key] else 'does not hold'}")
    print(f"  Exact merging shortens the GA-EAX merge (R1 or R2): {verdicts['R1'] or verdicts['R2']}; the model's "
          f"edges shorten the exact merge (R3 and R4): {verdicts['R3'] and verdicts['R4']}")

    for group in GROUPS:
        rows = [r for r in records if in_group(r, group)]
        print(f"\n{group} ({len(rows)} instances)")
        for i, (key, a, b) in enumerate(READINGS + DESCRIPTIVE):
            print(f"  {key} {a + ' / ' + b:30s} n={len(rows):2d}  {text(summary(deltas(group, a, b), 20261200 + i))}")

    print("\nExact arms (SCIP, one core each, the time left after the cut and the arm's own preparation)")
    for arm in EXACT_ARMS:
        arms = [r["exact"]["arms"][arm] for r in records]
        solver = [a["solver"] for a in arms]
        for size in ("10k", "20k"):
            part = [s for r, s in zip(records, solver) if in_group(r, size)]
            print(f"  {arm:14s} {size}: proven optimal {sum(s['optimal'] for s in part)}/{len(part)}")
        unchanged = sum(r["results"][arm] == values[r["name"]]["at_cut"] for r in records)
        open_gaps = [s["gap"] for s in solver if not s["optimal"]]
        proven = ", ".join("%.1f" % s["solve_s"] for s in solver if s["optimal"]) or "-"
        print(f"    time limit {min(a['exact_budget_s'] for a in arms):.1f} to {max(a['exact_budget_s'] for a in arms):.1f} s; "
              f"solve time median {np.median([s['solve_s'] for s in solver]):.1f} s, max {max(s['solve_s'] for s in solver):.1f} s "
              f"(proven optima: {proven} s); finished "
              f"at most {max(a['finished_after_deadline_s'] for a in arms):.2f} s after the deadline; late results: "
              f"{sum(a['best_found_late'] for a in arms)}")
        print(f"    graph: edges per node {np.mean([s['edges'] / s['nodes'] for s in solver]):.3f}, fixed "
              f"{np.mean([s['fixed'] / s['nodes'] for s in solver]):.3f} and removed "
              f"{np.mean([s['removed'] / s['nodes'] for s in solver]):.4f} per node, free edges median "
              f"{np.median([s['free'] for s in solver]):.0f}, core median {np.median([s['core'] for s in solver]):.0f} nodes "
              f"(10k {np.median([s['core'] for r, s in zip(records, solver) if r['n'] == 10000]):.0f}, 20k "
              f"{np.median([s['core'] for r, s in zip(records, solver) if r['n'] == 20000]):.0f})")
        print(f"    branch-and-bound nodes median {np.median([s['bnb_nodes'] for s in solver]):.0f}, cuts median "
              f"{np.median([s['cuts'] for s in solver]):.0f}; gap left when not proven: "
              + (f"median {100 * np.median(open_gaps):.3f}%, max {100 * max(open_gaps):.3f}%" if open_gaps else "-")
              + f"; no shorter tour than the start: {unchanged}/{len(records)}; extra edges in the tour: "
              f"{sum(a['extra_used'] for a in arms)} on {sum(a['extra_used'] > 0 for a in arms)} instances")

    print("\nShortest tour of the six-tour union graph, where a record proves it (descriptive)")
    known = []
    for r in records:
        opt, source = union_optimum(r)
        if opt is None:
            continue
        known.append(r)
        v = values[r["name"]]
        print(f"  {r['name']:15s} {opt:>11d} (from {source}); {BEST6} {pct(v[BEST6], opt):+.4f}%, {DUAL} "
              f"{pct(v[DUAL], opt):+.4f}%, {MERGE} {pct(v[MERGE], opt):+.4f}%, {EXACT} {pct(v[EXACT], opt):+.4f}%")
    for arm in (BEST6, DUAL, MERGE, EXACT):
        x = [pct(values[r["name"]][arm], union_optimum(r)[0]) for r in known]
        s = summary(x, 20261300)
        print(f"  {arm:14s} against the union optimum on {len(known)} instances: shorter / equal / longer "
              f"{s['win']}/{s['tie']}/{s['loss']}, mean {s['mean']:+.4f}%")

    print("\nPer instance: difference from lkh6 (%); exact arms: proven optimal (*)\n  " + f"{'instance':16s}{'lkh6':>12s}"
          + "".join(f"{a:>15s}" for a in ARMS[1:]))
    for r in records:
        v = values[r["name"]]
        star = {a: "*" if r["exact"]["arms"][a]["solver"]["optimal"] else " " for a in EXACT_ARMS}
        print(f"  {r['name']:16s}{v['lkh6']:12d}"
              + "".join(f"{pct(v[a], v['lkh6']):+14.4f}{star.get(a, ' ')}" for a in ARMS[1:]))

    avg = lambda items, f: float(np.mean([f(x) for x in items]))
    print("\nBy size (means; descriptive)")
    for size in ("10k", "20k"):
        rows = [r for r in records if in_group(r, size)]
        conv = lambda arm: (sum(bool(r["dual"]["arms"][arm]["eax"] and r["dual"]["arms"][arm]["eax"]["converged"])
                                for r in rows), sum(r["dual"]["arms"][arm]["eax"] is not None for r in rows))
        source = {}
        for r in rows:
            key = "+".join(sorted(r["dual"]["source"]))
            source[key] = source.get(key, 0) + 1
        print(f"  {size}: union edges per node {avg(rows, lambda r: r['exact']['union_edges'] / r['n']):.3f} (six tours), "
              f"{avg(rows, lambda r: r['dual']['arms'][TRIALS]['population']['union_edges'] / r['n']):.3f} (102 tours); "
              f"edges of the six tours shared pairwise {avg(rows, lambda r: r['exact']['pairwise']['pairwise_shared']):.4f}; "
              f"GA-EAX converged on the 102 tours {conv(TRIALS)[0]}/{conv(TRIALS)[1]}, on the six tours "
              f"{conv(BEST6)[0]}/{conv(BEST6)[1]}; dual_50 came from {source}")
        print("    edges outside the six-tour union, mean per tour: " + ", ".join(
            f"{arm} {avg(rows, lambda r: r['edges_outside_union6'][arm]):.2f}" for arm in TOUR_ARMS))
        model = [r["exact"]["arms"][BETA]["model"] for r in rows]
        print(f"    BetaTour (exact6beta_50): {rows[0]['exact']['k_extra']} edges added per instance, inference "
              f"{avg(model, lambda m: m['t_model_s']):.2f} s and preparation {avg(model, lambda m: m['t_prep_s']):.2f} s; "
              f"smallest predicted probability of an added edge {min(m['prob_min'] for m in model):.3f} to "
              f"{max(m['prob_min'] for m in model):.3f}")
    if src:
        check_tours(path, records, src)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 4) or (len(sys.argv) == 4 and sys.argv[2] != "--tours"):
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[3] if len(sys.argv) == 4 else None)
