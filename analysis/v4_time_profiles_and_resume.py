"""Run v4: mean gap over the budget and the improvements right after the resume (Section 5.3)."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
import json, glob, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(ROOT, "src"))
import tsplab as T
inst = {json.load(open(f))["inst"]: json.load(open(f)) for f in sorted(glob.glob(f"{RES}/v4/results/instances/*.json"))}
best = json.load(open(f"{RES}/v4/results/lkh_v4.json"))["best"]
cls = lambda s: "TSPLIB" if s == "tsplib" else s[3:]
fr = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
for c in ["1k", "10k", "TSPLIB", "20k"]:
    xs = [x for x in inst.values() if cls(x["set"]) == c]
    out = {}
    for k in ("pop", "tf"):
        arr = np.array([T.curve_abs(r["events"], best[x["inst"]], [f * x["budget"] for f in fr]) for x in xs for r in x[k]["runs"]], dtype=float)
        out[k] = np.nanmean(arr, 0)
        nev = np.mean([len(r["events"]) for x in xs for r in x[k]["runs"]])
        out[k + "_nev"] = nev
    print(f"{c:6s} per-run mean gap at " + " ".join(f"{int(f*100)}%" for f in fr))
    print("   pop ", np.round(out["pop"], 4), f"events/run {out['pop_nev']:.1f}")
    print("   tf  ", np.round(out["tf"], 4), f"events/run {out['tf_nev']:.1f}")
# what happens right after the resume at 20k and 10k: first resumed event vs trial-1 cost, and time of it
print()
for c in ["10k", "20k", "TSPLIB"]:
    xs = [x for x in inst.values() if cls(x["set"]) == c]
    gains, dts, n_after = [], [], []
    for x in xs:
        off = x["tf"]["meta"]["resume_at"]
        for r in x["tf"]["runs"]:
            ev = sorted(r["events"], key=lambda e: e[1])
            first = ev[0][0]
            after = [e for e in ev if e[1] > off]
            if after:
                gains.append(100 * (first - after[0][0]) / first); dts.append(after[0][1] - off)
            n_after.append(len(after))
    print(f"{c}: first event after resume: improvement over trial 1 {np.mean(gains):.3f}% after {np.mean(dts):.1f}s; events after resume per run {np.mean(n_after):.1f}")
    # baseline: second event relative to trial 1
    g2, t2 = [], []
    for x in xs:
        for r in x["pop"]["runs"]:
            ev = sorted(r["events"], key=lambda e: e[1])
            if len(ev) > 1:
                g2.append(100 * (ev[0][0] - ev[1][0]) / ev[0][0]); t2.append(ev[1][1] - ev[0][1])
    print(f"      baseline: second event improves trial 1 by {np.mean(g2):.3f}% after {np.mean(t2):.1f}s")
