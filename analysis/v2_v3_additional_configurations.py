"""Additional single-core configurations of Section 5.2: LKH-3 default, edge fixing and the 16-tour ensemble
(run v2), and the two further BetaTour lists of run v3 (top 3 + nearest neighbours, union with POPMUSIC)."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
import sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest
sys.path.insert(0, os.path.join(ROOT, "src"))
import tsplab as T


def report(label, d, tests=True):
    d = np.asarray(d, dtype=float)
    w, l = int((d < -1e-12).sum()), int((d > 1e-12).sum())
    t = len(d) - w - l
    out = f"{label:55s} {w}/{t}/{l}"
    if tests:
        nz = d[np.isfinite(d) & (np.abs(d) > 1e-12)]
        out += f"   sign p = {binomtest(w, w + l).pvalue:.3g}   Wilcoxon p = {wilcoxon(nz).pvalue:.3g}"
    print(out)


# run v2: gap at the end of the budget; a run without any tour counts as an infinite gap
v2 = pd.read_csv(os.path.join(RES, "v2", "results", "lkh_eval_v2.csv"))
G = v2.pivot_table(index="inst", columns="config", values="gap@100%B", dropna=False)
def diff(a, b):
    A, B = G[a].fillna(np.inf), G[b].fillna(np.inf)
    d = np.array((A - B).to_numpy(), dtype=float, copy=True)
    d[np.isnan(d)] = 0.0              # both without a tour: tie
    return d
POP, DEF = "LKH-3 POPMUSIC (ascent yok)", "LKH-3 varsayılan (α+ascent)"
print("== run v2 (51 instances, end of budget)")
report("baseline (LKH-3 POPMUSIC) vs LKH-3 default", diff(POP, DEF), tests=False)
report("BetaTour top 5 vs LKH-3 default", diff("TourFold-v2(8)", DEF), tests=False)
report("BetaTour top 5 vs baseline", diff("TourFold-v2(8)", POP))
report("BetaTour top 5, 16 tours vs baseline", diff("TourFold-v2(16)", POP))
report("counting with edge fixing vs counting", diff("Topluluk(8) omurgası sabit", "Topluluk(8) + sayım+"))
report("BetaTour top 5 vs counting with edge fixing", diff("TourFold-v2(8)", "Topluluk(8) omurgası sabit"))
report("BetaTour top 5 with filtered fixing vs without", diff("TourFold-v2(8) + süzülmüş omurga", "TourFold-v2(8)"))

# run v3: the two further lists
runs = [x for f in sorted(glob.glob(os.path.join(RES, "v3", "results", "lkh_eval_parts", "part_*.json"))) for x in json.load(open(f))]
best = json.load(open(os.path.join(RES, "v3", "results", "lkh_eval_v3.json")))["best"]
R = {(x["inst"], x["key"]): x for x in runs}
inst = sorted({x["inst"] for x in runs})
g = lambda x: T.curve_abs(x["events"], best[x["inst"]], [x["budget"]])[0]
print("== run v3 (51 instances, end of budget)")
for k, label in (("tfnear", "BetaTour top 3 + nearest neighbours"), ("tfuni", "BetaTour top 5 union POPMUSIC (up to 8)")):
    report(f"{label} vs baseline", [g(R[(i, k)]) - g(R[(i, "pop")]) for i in inst])
    report(f"{label} vs control", [g(R[(i, k)]) - g(R[(i, "ctrl")]) for i in inst])
