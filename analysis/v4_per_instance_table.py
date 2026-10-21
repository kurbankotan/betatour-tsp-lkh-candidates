"""Run v4: per-instance table of single-run and best-of-six gaps, resume times and first-trial spread."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
import json, glob, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(ROOT, "src"))
import tsplab as T
inst = [json.load(open(f)) for f in sorted(glob.glob(f"{RES}/v4/results/instances/*.json"))]
best = json.load(open(f"{RES}/v4/results/lkh_v4.json"))["best"]
rows = []
for x in inst:
    if x["set"] not in ("uni20k", "clu20k", "uni10k", "clu10k", "tsplib"): continue
    b, B = best[x["inst"]], x["budget"]
    g = {k: [T.curve_abs(r["events"], b, [B])[0] for r in x[k]["runs"]] for k in ("pop", "tf")}
    bo = {k: T.curve_abs([e for r in x[k]["runs"] for e in r["events"]], b, [B])[0] for k in ("pop", "tf")}
    m = x["tf"]["meta"]
    rows.append(dict(set=x["set"], inst=x["inst"], n=x["n"], order="".join(o[0] for o in x["order"]),
                     pop_mean=np.mean(g["pop"]), tf_mean=np.mean(g["tf"]), pop_best=bo["pop"], tf_best=bo["tf"],
                     resume=round(m["resume_at"], 1), first_max=round(m["t_phase1"], 1),
                     first_spread=round(max(m["first_costs"]) / min(m["first_costs"]) * 100 - 100, 3)))
d = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(d.round(4).to_string(index=False))
print("\norder effect (p=pop first): mean tf_mean - pop_mean by order")
print(d.assign(diff=d.tf_mean - d.pop_mean).groupby(["set", "order"])["diff"].mean().round(4))
