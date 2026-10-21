"""Run v4: post hoc split at 14,000 nodes, per-instance spread and improvement counts (Section 5.3)."""
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
nafter = 0; nev_tot = 0
for x in inst:
    b, B = best[x["inst"]], x["budget"]
    for k in ("pop", "tf"):
        for r in x[k]["runs"]:
            nafter += sum(1 for c, t in r["events"] if t > B); nev_tot += len(r["events"])
            g = T.curve_abs(r["events"], b, [B])[0]
            fin = min([c for c, t in r["events"] if t <= B] or [np.inf])
            rows.append(dict(cls=cls(x["set"]), set=x["set"], inst=x["inst"], n=x["n"], conf=k, seed=r["seed"], g100=g, fin=fin,
                             nev=len(r["events"]), nevB=sum(1 for c, t in r["events"] if t <= B)))
d = pd.DataFrame(rows)
print("events after B:", nafter, "of", nev_tot)
print("events per run (all / within B):")
print(d.groupby(["cls", "conf"])[["nev", "nevB"]].mean().round(2))
# 14,000 split
def test(sub, label):
    s = sub.pivot_table(index=["inst", "seed"], columns="conf", values="g100")
    dd = (s["tf"] - s["pop"]).values
    w, l = int((dd < -1e-12).sum()), int((dd > 1e-12).sum()); t = len(dd) - w - l
    nz = dd[np.abs(dd) > 1e-12]
    print(f"{label}: n_inst={sub.inst.nunique()} pairs={len(dd)} {w}/{t}/{l} sign p={binomtest(w, w+l).pvalue:.3g} wilcoxon p={wilcoxon(nz).pvalue:.3g}"
          f" mean pop={sub[sub.conf=='pop'].g100.mean():.4f} tf={sub[sub.conf=='tf'].g100.mean():.4f}")
test(d[d.n < 14000], "<14000")
test(d[d.n >= 14000], ">=14000")
test(d, "all")
for c in ["1k", "10k", "TSPLIB", "20k"]:
    test(d[d.cls == c], c)
# per-instance mean single-run gap for TSPLIB >14k
pi = d.groupby(["inst", "n", "cls", "conf"]).g100.mean().unstack("conf").reset_index()
print(pi[pi.n >= 14000].round(4).to_string(index=False))
# per-run W/T/L for TSPLIB >14k instances
for i in ["brd14051", "d15112", "d18512"]:
    s = d[d.inst == i].pivot_table(index="seed", columns="conf", values="g100")
    dd = (s["tf"] - s["pop"]).values
    print(i, f"{(dd<-1e-12).sum()}/{(abs(dd)<=1e-12).sum()}/{(dd>1e-12).sum()}")
# spread per instance
div = d.groupby(["cls", "inst", "n", "conf"]).agg(std0=("fin", lambda v: 100 * np.std(v) / np.mean(v)),
                                                 std1=("fin", lambda v: 100 * np.std(v, ddof=1) / np.mean(v)),
                                                 distinct=("fin", lambda v: len(set(v)))).reset_index()
print("\nspread by class: population SD (ddof=0) and sample SD (ddof=1)")
print(div.groupby(["cls", "conf"])[["std0", "std1", "distinct"]].mean().unstack(1).round(4))
w = div.pivot_table(index=["cls", "inst", "n"], columns="conf", values="std0").reset_index()
w["tf_smaller"] = w.tf < w["pop"] - 1e-12
w["equal"] = (w.tf - w["pop"]).abs() <= 1e-12
print("\ninstances where BetaTour spread < baseline spread, by class:")
print(w.groupby("cls").agg(n=("inst", "size"), tf_smaller=("tf_smaller", "sum"), equal=("equal", "sum")))
print("below 20k (n<20000):", int(w[w.n < 20000].tf_smaller.sum()), "of", int((w.n < 20000).sum()), "equal", int(w[w.n < 20000].equal.sum()))
print(w[(w.n < 20000) & ~w.tf_smaller].round(4).to_string(index=False))
# instance-level: was tf worse (mean gap) on instances >= 14000
print("\ninstances>=14000 where tf mean single-run gap worse:", int((pi[pi.n >= 14000].tf > pi[pi.n >= 14000]["pop"]).sum()), "of", int((pi.n >= 14000).sum()))
print("instances<14000 where tf mean single-run gap better:", int((pi[pi.n < 14000].tf < pi[pi.n < 14000]["pop"] - 1e-12).sum()), "worse", int((pi[pi.n < 14000].tf > pi[pi.n < 14000]["pop"] + 1e-12).sum()), "of", int((pi.n < 14000).sum()))
for c in ["1k", "10k", "TSPLIB", "20k"]:
    s = pi[pi.cls == c]
    print(c, "instances with tf mean gap better/equal/worse:", int((s.tf < s["pop"] - 1e-12).sum()), int(((s.tf - s["pop"]).abs() <= 1e-12).sum()), int((s.tf > s["pop"] + 1e-12).sum()))
