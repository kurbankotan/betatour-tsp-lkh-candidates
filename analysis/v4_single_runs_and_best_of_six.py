"""Run v4: single-run and best-of-six gaps, per-run paired tests and spread of the six runs (Table 4, Figure 4)."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
import json, glob, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest
sys.path.insert(0, os.path.join(ROOT, "src"))
import tsplab as T
inst = [json.load(open(f)) for f in sorted(glob.glob(f"{RES}/v4/results/instances/*.json"))]
best = json.load(open(f"{RES}/v4/results/lkh_v4.json"))["best"]
cls = lambda s: "TSPLIB" if s == "tsplib" else s[3:]
rows = []
for x in inst:
    b, B = best[x["inst"]], x["budget"]
    for k in ("pop", "tf"):
        for r in x[k]["runs"]:
            g = T.curve_abs(r["events"], b, [0.25 * B, 0.5 * B, B])
            fin = min([c for c, t in r["events"] if t <= B] or [np.inf])
            rows.append(dict(cls=cls(x["set"]), set=x["set"], inst=x["inst"], conf=k, seed=r["seed"], g25=g[0], g50=g[1], g100=g[2], fin=fin))
d = pd.DataFrame(rows)
print("== per-run mean gap (%) by class, and best-of-6 for reference")
pr = d.groupby(["cls", "conf"])[["g25", "g50", "g100"]].mean().unstack(1).round(4)
print(pr)
print("\n== per-run paired W/T/L (TF vs POP, same instance+seed) by class")
for col in ("g50", "g100"):
    for c in ["1k", "10k", "TSPLIB", "20k"]:
        s = d[d.cls == c].pivot_table(index=["inst", "seed"], columns="conf", values=col)
        dd = (s["tf"] - s["pop"]).values
        w, l = int((dd < -1e-12).sum()), int((dd > 1e-12).sum()); t = len(dd) - w - l
        nz = dd[np.abs(dd) > 1e-12]
        print(f"{col} {c:6s}: {w}/{t}/{l}  sign p={binomtest(w, w+l).pvalue if w+l else float('nan'):.3g}  wilcoxon p={wilcoxon(nz).pvalue if len(nz)>5 else float('nan'):.3g}  mean diff {np.mean(dd):+.4f}")
print("\n== diversity: spread of the 6 runs' final costs (relative std %, and # distinct finals)")
div = d.groupby(["cls", "inst", "conf"]).agg(std=("fin", lambda v: 100 * np.std(v) / np.mean(v)), distinct=("fin", lambda v: len(set(v)))).reset_index()
print(div.groupby(["cls", "conf"])[["std", "distinct"]].mean().unstack(1).round(4))
