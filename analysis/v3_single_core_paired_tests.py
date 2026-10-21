"""Run v3: pooled paired W/T/L, sign test and Wilcoxon test of every configuration against the baseline and the control (Table 3, Figure 3)."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
import json, glob, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest, spearmanr
sys.path.insert(0, os.path.join(ROOT, "src"))
import tsplab as T
runs = [x for f in sorted(glob.glob(f"{RES}/v3/results/lkh_eval_parts/part_*.json")) for x in json.load(open(f))]
runs2 = [x for f in sorted(glob.glob(f"{RES}/v2/results/lkh_eval_parts/part_*.json")) for x in json.load(open(f))]
L = json.load(open(f"{RES}/v3/results/lkh_eval_v3.json"))
best = L["best"]
key2name = {x["key"]: x["config"] for x in runs}
cls = lambda s: "TSPLIB" if s == "tsplib" else s[3:]
def gaps(rs, fr):
    out = {}
    for x in rs:
        b = best[x["inst"]]
        out[(x["inst"], x["config"])] = T.curve_abs(x["events"], b, [fr * x["budget"]])[0]
    return out
G = {fr: gaps(runs, fr) for fr in (0.25, 0.5, 1.0)}
inst = sorted({x["inst"] for x in runs}); iset = {x["inst"]: x["set"] for x in runs}
def cmp(a, b, fr=1.0, sub=None):
    d = []
    for i in inst:
        if sub and cls(iset[i]) not in sub: continue
        ga, gb = G[fr][(i, key2name[a])], G[fr][(i, key2name[b])]
        if ga is None or gb is None or np.isnan(ga) or np.isnan(gb): continue
        d.append(ga - gb)
    d = np.array(d)
    w, t, l = int((d < -1e-12).sum()), int((np.abs(d) <= 1e-12).sum()), int((d > 1e-12).sum())
    nz = d[np.abs(d) > 1e-12]
    pw = wilcoxon(nz).pvalue if len(nz) > 5 else np.nan
    return w, t, l, binomtest(w, w + l).pvalue if w + l else np.nan, pw, float(np.mean(d)), float(np.median(d))
print("== pooled, 100%B: W/T/L, sign p, Wilcoxon p, mean diff (gap points %), median diff")
for a, b in [("tf5","pop"),("tfpop","pop"),("tf4pop","pop"),("cnt","pop"),("ctrl","pop"),
             ("tf5","ctrl"),("tfpop","ctrl"),("tf4pop","ctrl"),("cnt","ctrl"),("tfnear","ctrl"),("tfuni","ctrl"),
             ("tf5","cnt"),("tfpop","cnt"),("tf4pop","tfpop"),("tfpop","tf5")]:
    for fr in (0.5, 1.0):
        r = cmp(a, b, fr)
        print(f"{a:7s} vs {b:5s} @{int(fr*100):3d}%: {r[0]:2d}/{r[1]:2d}/{r[2]:2d}  sign p={r[3]:.3g}  wilcoxon p={r[4]:.3g}  mean {r[5]:+.4f}  median {r[6]:+.4f}")
