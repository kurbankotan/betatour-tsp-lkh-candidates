"""Recompute the tests and the comparisons of run v30 (time to the proven optimum) from the ZIP.

Usage:
    python analysis/v30_time_to_optimum.py results/v30/v30_tto37_results.zip

It reads only the archived records and shares no code with the notebook. For every run the time to the optimum is
recomputed from the record and compared with the archived value: an uninterrupted arm (`pop`, `alpha`) takes LKH-3's
`Time.total`; a restarted arm (`pop_r`, `beta`, `count`, `mix`, `union`) takes the pause plus the `Time.total` of the
continuation, or the CPU time of the first trial when that trial already reached the optimum; a run that did not reach
the optimum within the cap takes infinity. PAR10 is the mean over the six seeds with infinity replaced by 10 x the cap,
and it is compared with `summary/par10_wide.csv`. Hypotheses H1 (beta < pop) and H2 (beta < alpha) and the secondary
test (beta < count) as in experiments/PROTOCOL_v30.md: exact two-sided sign test over the 37 instances on PAR10, where
two values within 1% of the instance's cap count as a tie and ties are excluded; Holm correction over H1 and H2; a
hypothesis is supported if beta is lower on more instances than it is higher and its Holm-adjusted p is below 0.05.
Counts are instances on which the first arm was lower / tied / higher. Ratio = geometric mean over instances of the
first arm's PAR10 divided by the second's, with a 95% percentile interval from 20,000 bootstrap resamples of instances
(the notebook drew 5,000, so the last digit can differ). Everything else is descriptive. The last block prints two
checks that the protocol did not specify (manuscript, Section 5.12): the tests with a fixed number of seconds
added to every beta run, and the time until the first of the six seeds of an arm reaches the optimum.
"""

from __future__ import annotations

import json
import math
import sys
import zipfile
from math import comb

import numpy as np

ARMS = ["pop", "alpha", "pop_r", "beta", "count", "mix", "union"]
RESTARTED = {"pop_r", "beta", "count", "mix", "union"}
GROUPS = ["uni1k", "clu1k", "uni2k", "clu2k", "tsplib"]
SUBSETS = {"all": lambda g: True, "synthetic": lambda g: g != "tsplib", "TSPLIB": lambda g: g == "tsplib"}
HYPOTHESES = [("H1", "beta", "pop"), ("H2", "beta", "alpha")]
PAIRS = [("beta", "count"), ("beta", "pop_r"), ("pop_r", "pop"), ("mix", "beta"), ("union", "beta"), ("count", "pop"),
         ("count", "alpha"), ("mix", "pop"), ("union", "pop"), ("union", "alpha"), ("alpha", "pop")]
PAR_FACTOR, TIE_FRACTION, REPLICATES = 10.0, 0.01, 20000


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


def average_ranks(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def time_to_optimum(r: dict) -> float:
    if r["arm"] in RESTARTED and r.get("first_found"):
        t = float(r["first_cpu"])
    elif r["found"]:
        t = float(r["time_total"]) + (float(r["pause"]) if r["arm"] in RESTARTED else 0.0)
    else:
        return math.inf
    return t if t <= r["cap"] + 1e-9 else math.inf


def load(path: str) -> tuple:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        runs = [json.loads(z.read(n)) for n in names if n.startswith("runs/") and n.endswith(".json")]
        lists = {n.split("/")[1]: json.loads(z.read(n)) for n in names
                 if n.startswith("instances/") and n.endswith("/lists.json")}
        manifest = json.loads(z.read("manifest.json"))
        wide = z.read("summary/par10_wide.csv").decode().splitlines()
    archived = {}
    header = wide[0].split(",")[1:]
    for line in wide[1:]:
        cells = line.split(",")
        archived[cells[0]] = dict(zip(header, map(float, cells[1:])))
    return runs, lists, manifest, archived


def compare(par: dict, caps: dict, names: list, first: str, second: str, tie_fraction: float, seed: int) -> dict:
    a = np.array([par[n][first] for n in names])
    b = np.array([par[n][second] for n in names])
    tol = tie_fraction * np.array([caps[n] for n in names])
    tie = np.isclose(a, b, rtol=1e-9, atol=1e-9) | (np.abs(a - b) <= tol)
    wins, losses = int(((a < b) & ~tie).sum()), int(((a > b) & ~tie).sum())
    r = np.log(a / b)
    boot = r[np.random.default_rng(seed).integers(0, len(r), size=(REPLICATES, len(r)))].mean(axis=1)
    return dict(n=len(names), win=wins, tie=int(tie.sum()), loss=losses, p=sign_p(wins, losses),
                ratio=float(np.exp(r.mean())), low=float(np.exp(np.quantile(boot, 0.025))),
                high=float(np.exp(np.quantile(boot, 0.975))))


def text(s: dict) -> str:
    return (f"n={s['n']:2d}  {s['win']:2d}/{s['tie']:2d}/{s['loss']:<2d}  sign p = {s['p']:.2g}  PAR10 ratio "
            f"{s['ratio']:.3f} (95% CI {s['low']:.3f} to {s['high']:.3f})")


def main(path: str) -> None:
    runs, lists, manifest, archived = load(path)
    hw, proto = manifest["hardware"], manifest["protocol"]
    print(f"Run v30: {len(runs)} runs; {hw['gpu']} (BetaTour inference), {hw['vcpus_visible']} vCPUs, "
          f"{hw['parallel_lkh']} LKH-3 processes at a time, {hw['cpu_model']}; code {manifest['code_version']} "
          f"{manifest['tto_sha256'][:12]}, notebook build {manifest['notebook_build'][:12]}; test mode "
          f"{manifest['test_mode']}; sessions: {len({r['session'] for r in runs})}")
    print(f"Protocol {proto['path']} (SHA-256 {proto['sha256'][:12]}) found on GitHub in commit "
          f"{proto['protocol_commit'][:10]} of {proto['protocol_commit_time']}, checked {proto['checked_utc']}; runs "
          f"finished {min(r['utc'] for r in runs)} to {max(r['utc'] for r in runs)}")

    # integrity and the time to the optimum of every run
    keys = {(r["instance"], r["arm"], r["seed"]) for r in runs}
    names = sorted({r["instance"] for r in runs})
    assert len(keys) == len(runs) == len(names) * len(ARMS) * 6, "missing or duplicate runs"
    ran = [r for r in runs if not r.get("first_found")]          # the others reused a first trial at the optimum
    assert all(r["returncode"] == 0 and not r["timed_out"] for r in ran), "failed LKH-3 process"
    assert not any(r["shorter"] or r["new_optimum"] for r in ran), "a tour shorter than the optimum"
    assert all(r["cost"] == r["optimum"] for r in ran if r["found"]), "a found run without the optimal length"
    tto = {}
    for r in runs:
        t = time_to_optimum(r)
        stored = math.inf if r["tto"] is None else float(r["tto"])
        assert (math.isinf(t) and math.isinf(stored)) or abs(t - stored) < 1e-9, (r["instance"], r["arm"], r["seed"])
        tto[(r["instance"], r["arm"], r["seed"])] = t
    group = {r["instance"]: r["group"] for r in runs}
    caps = {r["instance"]: float(r["cap"]) for r in runs}
    par = {n: {a: float(np.mean([PAR_FACTOR * caps[n] if math.isinf(tto[(n, a, s)]) else tto[(n, a, s)]
                                 for s in range(201, 207)])) for a in ARMS} for n in names}
    same = all(abs(par[n][a] - archived[n][a]) < 1e-6 for n in names for a in ARMS)
    reused = sorted({(r["instance"], r["seed"]) for r in runs if r.get("first_found")})
    print(f"All {len(runs)} times to the optimum follow from the records; PAR10 equals summary/par10_wide.csv: {same}; "
          f"every LKH-3 process ended normally, and every run that reached the optimum has its length; first trials "
          f"that already reached it (their time counts in every restarted arm): {reused}")

    print("\nHypotheses of the protocol (37 instances; ties within 1% of the cap; Holm over H1 and H2)")
    found = [compare(par, caps, names, a, b, TIE_FRACTION, 20261126 + i) for i, (_, a, b) in enumerate(HYPOTHESES)]
    verdicts = {}
    for (key, a, b), s, p_adj in zip(HYPOTHESES, found, holm([s["p"] for s in found])):
        verdicts[key] = s["win"] > s["loss"] and p_adj < 0.05
        print(f"  {key} {a + ' < ' + b:15s} {text(s)}, Holm p = {p_adj:.2g}: "
              f"{'supported' if verdicts[key] else 'not supported'}")
    sec = compare(par, caps, names, "beta", "count", TIE_FRACTION, 20261130)
    credited = verdicts["H1"] and verdicts["H2"] and sec["win"] > sec["loss"] and sec["p"] < 0.05
    print(f"  secondary beta < count {text(sec)}")
    print(f"  H1 and H2 supported: {verdicts['H1'] and verdicts['H2']}; learned ranking credited: {credited}")
    print("\nSensitivity (not in the protocol): the same tests without the tie rule")
    for i, (a, b) in enumerate([("beta", "pop"), ("beta", "alpha"), ("beta", "count")]):
        print(f"  {a + ' < ' + b:15s} {text(compare(par, caps, names, a, b, 0.0, 20261140 + i))}")

    print("\nOther pairs by subset (descriptive; ties within 1% of the cap)")
    for i, (a, b) in enumerate([("beta", "pop"), ("beta", "alpha")] + PAIRS):
        for subset, keep in SUBSETS.items():
            sub = [n for n in names if keep(group[n])]
            print(f"  {a + ' / ' + b:15s} {subset:9s} {text(compare(par, caps, sub, a, b, TIE_FRACTION, 20261200 + i))}")

    print("\nBy group: runs that reached the optimum (%), median time to the optimum over all runs (s; inf = more "
          "than half missed it) and geometric mean of PAR10 (s)")
    print(f"  {'group':8s}{'':12s}" + "".join(f"{a:>9s}" for a in ARMS))
    for g in GROUPS + ["all"]:
        sub = [n for n in names if g == "all" or group[n] == g]
        cells = {a: [tto[(n, a, s)] for n in sub for s in range(201, 207)] for a in ARMS}
        print(f"  {g:8s}{'reached %':12s}" + "".join(f"{100 * np.mean(np.isfinite(cells[a])):9.1f}" for a in ARMS))
        print(f"  {'':8s}{'median s':12s}" + "".join(f"{np.median(cells[a]):9.2f}" for a in ARMS))
        print(f"  {'':8s}{'PAR10 gm s':12s}"
              + "".join(f"{np.exp(np.mean([np.log(par[n][a]) for n in sub])):9.2f}" for a in ARMS))
    print("  instances on which all six seeds reached the optimum: "
          + ", ".join(f"{a} {sum(all(np.isfinite(tto[(n, a, s)]) for s in range(201, 207)) for n in names)}"
                      for a in ARMS))

    print("\nPause of the restarted arms (s; mean by group) and BetaTour's inference (s)")
    for g in GROUPS:
        sub = [n for n in names if group[n] == g]
        print(f"  {g:8s}" + "".join(f"{a} {np.mean([lists[n]['pause'][a] for n in sub]):.2f}  " for a in sorted(RESTARTED))
              + f"inference {np.mean([lists[n]['model_s'] for n in sub]):.3f}, first trials (longest of six) "
              f"{np.mean([max(lists[n]['first_cpu']) for n in sub]):.2f}")

    cov = [(n, a, s, lists[n]["missed_optimal_edges"][a][s - 201], np.isfinite(tto[(n, a, s)]))
           for n in names if lists[n].get("missed_optimal_edges") for a in ARMS for s in range(201, 207)]
    print(f"\nOptimal edges missed by the lists and success ({len({c[0] for c in cov})} instances whose optimal tours the "
          f"pilot saved; descriptive)")
    for a in ARMS:
        rows = [c for c in cov if c[1] == a]
        print(f"  {a:6s} missed per run: mean {np.mean([c[3] for c in rows]):.2f}, max {max(c[3] for c in rows)}; "
              f"runs that reached the optimum {100 * np.mean([c[4] for c in rows]):.1f}%")
    for low, high, label in [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 10 ** 9, ">10")]:
        rows = [c for c in cov if low <= c[3] <= high]
        print(f"  lists that missed {label:>4s} optimal edges: {len(rows):3d} runs, "
              f"{100 * np.mean([c[4] for c in rows]):.1f}% reached the optimum")
    by_instance = {}
    for n, a, s, missed, ok in cov:
        by_instance.setdefault(n, []).append((missed, ok))
    means = np.array([[np.mean([m for m, _ in v]), np.mean([o for _, o in v])] for v in by_instance.values()])
    print(f"  Spearman correlation over instances between mean missed edges and share of successful runs: "
          f"{spearman(means[:, 0], means[:, 1]):.2f}")
    within = [spearman([m for m, _ in v], [o for _, o in v]) for v in by_instance.values()
              if len({m for m, _ in v}) > 1 and len({o for _, o in v}) > 1]
    print(f"  within instances (where both vary): negative on {sum(r < 0 for r in within)} of {len(within)}, "
          f"median {np.median(within):.2f}")

    print("\nExploratory checks after the run (not in the protocol)")
    print("  1. Overhead: seconds added to every beta run (for example, inference on a CPU instead of the GPU); a run "
          "that then exceeds the cap counts as missed")
    for extra in (0.25, 0.5, 1.0, 2.0):
        shifted = {n: dict(par[n]) for n in names}
        for n in names:
            ts = [tto[(n, "beta", s)] + extra for s in range(201, 207)]
            shifted[n]["beta"] = float(np.mean([t if t <= caps[n] + 1e-9 else PAR_FACTOR * caps[n] for t in ts]))
        line = "; ".join(f"{a + ' < ' + b}: {text(compare(shifted, caps, names, a, b, TIE_FRACTION, 20261300 + k))}"
                         for k, (a, b) in enumerate([("beta", "pop"), ("beta", "alpha"), ("beta", "count")]))
        print(f"    +{extra:.2f} s  {line}")
    print("  2. Six processes in parallel: time until the first of the six seeds of an arm reaches the optimum "
          "(10 x the cap if none does)")
    first = {n: {a: min(tto[(n, a, s)] for s in range(201, 207)) for a in ARMS} for n in names}
    six = {n: {a: (t if np.isfinite(t) else PAR_FACTOR * caps[n]) for a, t in first[n].items()} for n in names}
    print("    instances solved by the six processes: "
          + ", ".join(f"{a} {sum(np.isfinite(first[n][a]) for n in names)}" for a in ARMS))
    print("    geometric mean (s): "
          + ", ".join(f"{a} {np.exp(np.mean([np.log(six[n][a]) for n in names])):.2f}" for a in ARMS))
    for k, (a, b) in enumerate([("beta", "pop"), ("beta", "alpha"), ("beta", "count"), ("union", "beta"),
                                ("union", "pop"), ("alpha", "pop")]):
        print(f"    {a + ' / ' + b:15s} {text(compare(six, caps, names, a, b, TIE_FRACTION, 20261400 + k))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
