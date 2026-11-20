"""Populations for GA-EAX from the trial tours that LKH computes and normally discards (development probe).

usage: bp_pop.py SRC INSTANCE.npy REF.tour OUT_POP T_SECONDS MODE K DUMP [DUMP ...]
MODE last  : the last K distinct trial tours of each dump that ended before T_SECONDS
MODE first : the first K distinct trial tours of each dump
MODE spread: K distinct trial tours of each dump, evenly spaced over the trials that ended before T_SECONDS
"""
import sys, json
import numpy as np

src, inst, ref_file, out, t_lim, mode, k = sys.argv[1:8]
dumps = sys.argv[8:]
t_lim, k = float(t_lim), int(k)
sys.path.insert(0, src)
import tsplab as T

xy = np.load(inst)
n = len(xy)
rec = np.dtype([("trial", "<i8"), ("cost", "<i8"), ("ms", "<i8"), ("ids", "<i4", (n,))])

def keys(t):
    a, b = t, np.roll(t, -1)
    return np.sort(np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b))

ref = keys(T.read_tour(ref_file)) if ref_file != "-" else None
tours, costs, info = [], [], []
for d in dumps:
    a = np.fromfile(d, dtype=rec)
    a = a[a["ms"] <= 1000.0 * t_lim]
    seen, sel = set(), []
    order = range(len(a)) if mode == "first" else range(len(a) - 1, -1, -1)
    if mode == "spread":
        order = [int(i) for i in np.unique(np.linspace(0, len(a) - 1, min(len(a), 3 * k)).round().astype(int))][::-1]
    for i in order:
        t = a["ids"][i].astype(np.int64) - 1
        h = hash(keys(t).tobytes())
        if h in seen:
            continue
        seen.add(h)
        sel.append(i)
        if mode != "spread" and len(sel) == k:
            break
    if mode == "spread" and len(sel) > k:
        sel = [sel[int(j)] for j in np.linspace(0, len(sel) - 1, k).round().astype(int)]
    sel = sorted(sel)
    for i in sel:
        t = a["ids"][i].astype(np.int64) - 1
        assert T.is_valid_tour(t, n)
        c = int(T.tour_length(xy, t))
        assert c == int(a["cost"][i]), (c, int(a["cost"][i]))
        tours.append(t)
        costs.append(c)
    info.append({"dump": d.split("/")[-1], "trials_before_T": int(len(a)), "used": len(sel),
                 "trial_range": [int(a["trial"][sel[0]]), int(a["trial"][sel[-1]])],
                 "best_trial_cost_before_T": int(a["cost"].min()), "last_ms": int(a["ms"][sel[-1]])})
ks = [keys(t) for t in tours]
assert len({x.tobytes() for x in ks}) == len(ks) or len(dumps) > 1
with open(out, "w") as f:
    for t, c in zip(tours, costs):
        f.write(f"{n} {c}\n" + " ".join(map(str, (t + 1).tolist())) + " \n")
allk, cnt = np.unique(np.concatenate(ks), return_counts=True)
rng = np.random.default_rng(0)
pairs = [tuple(rng.choice(len(ks), 2, replace=False)) for _ in range(min(200, len(ks) * 3))]
summary = {"tours": len(tours), "best": min(costs), "mean": float(np.mean(costs)), "worst": max(costs),
           "union_edges": int(len(allk)), "common_to_all": int((cnt == len(ks)).sum()),
           "pairwise_shared": round(float(np.mean([np.isin(ks[i], ks[j]).mean() for i, j in pairs])), 4)}
if ref is not None:
    summary["reference_edges_missing_from_union"] = int((~np.isin(ref, allk)).sum())
print(json.dumps({"population": summary, "dumps": info}))
