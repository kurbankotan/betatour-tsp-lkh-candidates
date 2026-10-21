"""Run v3: free-ensemble estimate (Section 5.2), repeatability of runs v2 and v3 (Section 6.4) and the coverage-gap Spearman correlation (Section 6.2)."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
import json, glob, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest, spearmanr
sys.path.insert(0, os.path.join(ROOT, "src"))
import tsplab as T
runs = [x for f in sorted(glob.glob(f"{RES}/v3/results/lkh_eval_parts/part_*.json")) for x in json.load(open(f))]
runs2 = [x for f in sorted(glob.glob(f"{RES}/v2/results/lkh_eval_parts/part_*.json")) for x in json.load(open(f))]
best = json.load(open(f"{RES}/v3/results/lkh_eval_v3.json"))["best"]
R = {(x["inst"], x["key"]): x for x in runs}
inst = sorted({x["inst"] for x in runs}); iset = {x["inst"]: x["set"] for x in runs}
cls = lambda s: "TSPLIB" if s == "tsplib" else s[3:]
g = lambda x, t: T.curve_abs(x["events"], best[x["inst"]], [t])[0]

print("== (b) 'free ensemble': TF gap at B vs POPMUSIC gap at B - offset (same LKH-phase time)")
for k in ("ctrl", "cnt", "tf5", "tfpop", "tf4pop"):
    rows = []
    for i in inst:
        x, p = R[(i, k)], R[(i, "pop")]
        B, o = x["budget"], x["offset"]
        a, b = g(x, B), g(p, B - o)
        rows.append((cls(iset[i]), a, b))
    d = pd.DataFrame(rows, columns=["c", "tf", "pop"])
    d["w"] = (d.tf < d["pop"] - 1e-12) | (d.tf.notna() & d["pop"].isna())
    d["l"] = (d.tf > d["pop"] + 1e-12) | (d.tf.isna() & d["pop"].notna())
    out = []
    for c in ["1k", "10k", "TSPLIB", "20k"]:
        s = d[d.c == c]; out.append(f"{c}: {int(s.w.sum())}/{len(s)-int(s.w.sum())-int(s.l.sum())}/{int(s.l.sum())}")
    W, Lw = int(d.w.sum()), int(d.l.sum())
    print(f"{k:7s}", " | ".join(out), f"| all {W}/{len(d)-W-Lw}/{Lw} p={binomtest(W, W+Lw).pvalue:.3g}")

print("\n== (d) noise: same configuration, v3 vs v2, per instance at 100%B and 25%B (W/T/L = v3 better/equal/worse)")
R2 = {(x["inst"], x["config"]): x for x in runs2}
for k3, c2 in (("pop", "LKH-3 POPMUSIC (ascent yok)"), ("cnt", "Topluluk(8) + sayım+"), ("tf5", "TourFold-v2(8)")):
    for fr in (0.25, 1.0):
        w = t = l = 0; ad = []
        for i in inst:
            x3, x2 = R[(i, k3)], R2.get((i, c2))
            if x2 is None: continue
            a, b = g(x3, fr * x3["budget"]), g(x2, fr * x2["budget"])
            if np.isnan(a) or np.isnan(b): continue
            ad.append(abs(a - b))
            w += a < b - 1e-12; l += a > b + 1e-12; t += abs(a - b) <= 1e-12
        print(f"{k3:5s} @{int(fr*100):3d}%: {w}/{t}/{l}  mean |diff| {np.mean(ad):.4f}")
# compare with config-to-config differences
for a_, b_ in (("tf5", "ctrl"), ("tf5", "pop"), ("tfpop", "tf5")):
    ad = [abs(g(R[(i, a_)], R[(i, a_)]["budget"]) - g(R[(i, b_)], R[(i, b_)]["budget"])) for i in inst]
    print(f"between configs {a_} vs {b_} @100%: mean |diff| {np.nanmean(ad):.4f}")

print("\n== (c) does the TourFold benefit grow where POPMUSIC misses more optimal edges? (REF instances)")
cov = pd.read_csv(f"{RES}/v3/results/coverage_v3.csv")
pm = cov[cov.method == "POPMUSIC"].set_index("inst")["missing_per_1k"]
tm = cov[cov.method == "TourFold-v2(8) ilk 5"].set_index("inst")["missing_per_1k"]
rows = []
for i in inst:
    if i not in pm.index: continue
    x, c, p = R[(i, "tf5")], R[(i, "ctrl")], R[(i, "pop")]
    B = x["budget"]
    rows.append((i, cls(iset[i]), pm[i], tm[i], g(x, B) - g(c, B), g(x, B) - g(p, B)))
d = pd.DataFrame(rows, columns=["inst", "c", "pop_miss", "tf_miss", "tf_minus_ctrl", "tf_minus_pop"])
for col in ("tf_minus_ctrl", "tf_minus_pop"):
    r = spearmanr(d.pop_miss - d.tf_miss, d[col])
    print(f"spearman(POPMUSIC missing - TF missing, {col}) = {r.statistic:+.3f} (p={r.pvalue:.3g}, n={len(d)})")
print(d.groupby("c")[["pop_miss", "tf_miss", "tf_minus_ctrl", "tf_minus_pop"]].mean().round(4))
