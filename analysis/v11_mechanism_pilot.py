"""Recompute the run-v11 and run-v11r2 statistics of the paper from a result ZIP.

Usage:
    python analysis/v11_mechanism_pilot.py results/v11r2/v11r2_pop6_candidate_mechanism_results.zip
    python analysis/v11_mechanism_pilot.py results/v11/v11_pop6_candidate_mechanism_results.zip

It reads only the archived records (result.json, telemetry.txt, applied.stats and
best.events of every arm and seed) and prints Table 8, Table 9 and the timing and
state-matching figures quoted in Sections 5.9 and A.12. Instances, not seeds, are the
units of the sign tests; the seed-level counts are descriptive.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from math import comb
from pathlib import PurePosixPath

import numpy as np

GUIDED = ("beta_add", "frequency_add", "random_add")
NAMES = {"beta_add": "BetaTour", "frequency_add": "frequency", "random_add": "random",
         "sham": "sham", "native_pop6": "native POP6", "native_pop7": "native POP7"}


def key_values(text: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        key, value = line.split()
        out[key] = float(value)
    return out


def sign_p(wins: int, losses: int) -> float:
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def main(path: str) -> None:
    archive = zipfile.ZipFile(path)
    assert archive.testzip() is None, "CRC error"
    read = lambda name: archive.read(name).decode("utf-8")
    manifest = json.loads(read("manifest.json"))
    arms, seeds = manifest["arms"], manifest["seeds"]
    instances = sorted({PurePosixPath(n).parts[1] for n in archive.namelist()
                        if n.startswith("instances/") and n.endswith("/result.json")})
    records = {(i, a): json.loads(read(f"instances/{i}/{a}/result.json")) for i in instances for a in arms}
    print(f"{manifest['experiment_name']}: {len(instances)} instances x {len(arms)} arms; "
          f"GPU {manifest['hardware']['gpu']}, {manifest['hardware']['vcpus']} vCPUs")
    for record in records.values():
        assert record["complete"] and record["code_sha256"] == manifest["code_sha256"]
        assert min(record["seed_costs"]) == record["best_cost"]

    patched = ("sham",) + GUIDED
    for inst in instances:
        audits = [records[inst, a]["policy_audit"] for a in patched]
        same = all(len({json.dumps(records[inst, a][key]) for a in patched}) == 1
                   for key in ("first_tour_hashes", "live_candidate_hashes", "native_candidate_hashes"))
        same &= len({a["model_probability_sha256"] for a in audits}) == 1
        same &= len({json.dumps(a["gate_sha256_per_seed"]) for a in audits}) == 1
        print(f"  {inst}: patched states, model outputs and gates identical = {same}")

    best = {k: r["best_cost"] for k, r in records.items()}
    seed = {k: np.asarray(r["seed_costs"], float) for k, r in records.items()}
    rel = lambda a, b: 100.0 * (a - b) / b
    print("\nBest-of-six comparisons (negative favours the first arm)")
    print("first / second | W/T/L | mean % | median % | sign p | seed W/T/L")
    pairs = [("beta_add", c) for c in ("native_pop6", "native_pop7", "sham", "random_add", "frequency_add")]
    pairs += [("sham", "native_pop6"), ("native_pop7", "native_pop6"), ("frequency_add", "sham"), ("random_add", "sham")]
    for t, c in pairs:
        r = np.array([rel(best[i, t], best[i, c]) for i in instances])
        s = np.concatenate([rel(seed[i, t], seed[i, c]) for i in instances])
        w, l = int((r < 0).sum()), int((r > 0).sum())
        print(f"{NAMES[t]} / {NAMES[c]} | {w}/{len(r) - w - l}/{l} | {r.mean():+.5f} | {np.median(r):+.5f} | "
              f"{sign_p(w, l):.3f} | {(s < 0).sum()}/{(s == 0).sum()}/{(s > 0).sum()}")

    telemetry, timing = {}, []
    for (inst, arm), record in records.items():
        if arm not in patched:
            continue
        for k, s in enumerate(seeds):
            base = f"{record['audit_relpath']}/seed_{s}"
            tel = key_values(read(f"{base}/telemetry.txt"))
            tel["selected"] = record["policy_audit"]["selected_actions_per_seed"][k]
            telemetry[inst, arm, s] = tel
            first = float(read(f"{base}/best.events").splitlines()[0].split()[0]) - record["start_epoch_s"]
            wait = key_values(read(f"{base}/applied.stats"))["wait_wall_s"]
            timing.append((first, first + wait, record["policy_audit"]["model_inference_s"]))
    timing = np.asarray(timing)
    print(f"\nFirst trial ended {timing[:, 0].min():.1f}-{timing[:, 0].max():.1f} s; search resumed "
          f"{timing[:, 1].min():.1f}-{timing[:, 1].max():.1f} s; mean inference {timing[:, 2].mean():.2f} s")
    sham_trials = {(i, s): telemetry[i, "sham", s]["trials_completed_after_action"] for i in instances for s in seeds}
    print("Trials after the pause (sham): min", min(sham_trials.values()), "max", max(sham_trials.values()))

    print("\nTelemetry per action | BetaTour | frequency | random")
    rows = {}
    for arm in GUIDED:
        t = {key: sum(telemetry[i, arm, s][key] for i in instances for s in seeds)
             for key in telemetry[instances[0], arm, seeds[0]]}
        a = t["selected"]
        ratio = np.median([telemetry[i, arm, s]["trials_completed_after_action"] / sham_trials[i, s]
                           for i in instances for s in seeds])
        rows[arm] = [t["action_candidate_enumerations"] / a, t["top_level_action_first_gate_passes"] / a,
                     t["action_accepted_edge_entries"] / a, 100 * t["distinct_actions_entering_accepted_tours"] / a,
                     1000 * t["action_incumbent_entries"] / a, 100 * t["current_better_tour_actions"] / a,
                     t.get("slot_restorations", float("nan")) / a, ratio]
    labels = ["enumerations", "first gain-test passes", "entries into the current tour",
              "actions entering the current tour (%)", "entries into new best tours per 1,000",
              "actions in the final best tours (%)", "slot restorations", "completed trials relative to sham"]
    for k, label in enumerate(labels):
        print(f"{label} | " + " | ".join(f"{rows[a][k]:.4g}" for a in GUIDED))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
