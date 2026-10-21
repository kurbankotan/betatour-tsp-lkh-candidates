"""Run v3: W/T/L by size class for additional pairs of configurations."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
import json, glob, sys, numpy as np
from scipy.stats import wilcoxon, binomtest
sys.path.insert(0, os.path.join(ROOT, "src"))
import tsplab as T
runs = [x for f in sorted(glob.glob(f"{RES}/v3/results/lkh_eval_parts/part_*.json")) for x in json.load(open(f))]
best = json.load(open(f"{RES}/v3/results/lkh_eval_v3.json"))["best"]
R = {(x["inst"], x["key"]): x for x in runs}
inst = sorted({x["inst"] for x in runs}); iset = {x["inst"]: x["set"] for x in runs}
cls = lambda s: "TSPLIB" if s == "tsplib" else s[3:]
g = lambda x, t: T.curve_abs(x["events"], best[x["inst"]], [t])[0]
def wtl(a, b, fr=1.0):
    out, allw, allt, alll, d = [], 0, 0, 0, []
    for c in ["1k", "10k", "TSPLIB", "20k"]:
        w = t = l = 0
        for i in inst:
            if cls(iset[i]) != c: continue
            xa, xb = R[(i, a)], R[(i, b)]
            ga, gb = g(xa, fr * xa["budget"]), g(xb, fr * xb["budget"])
            dd = ga - gb; d.append(dd)
            w += dd < -1e-12; l += dd > 1e-12; t += abs(dd) <= 1e-12
        out.append(f"{w}/{t}/{l}"); allw += w; allt += t; alll += l
    d = np.array(d); nz = d[np.abs(d) > 1e-12]
    return out, f"{allw}/{allt}/{alll}", binomtest(allw, allw + alll).pvalue, wilcoxon(nz).pvalue
for a, b in [("tf5", "cnt"), ("tf4pop", "pop"), ("tf4pop", "ctrl"), ("tfpop", "ctrl")]:
    print(a, b, wtl(a, b))
print({k for (_, k) in R})
