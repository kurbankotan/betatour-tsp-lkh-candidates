"""Potential of combining the final tours of independent LKH processes (development probe)."""
import sys, json, time, subprocess, shutil
from pathlib import Path
import numpy as np

src, v19, inst, tourdir, work, bindir = sys.argv[1:7]
names = sys.argv[7].split(",")
mip_seconds = float(sys.argv[8])
eax_bin, eax_tsp = sys.argv[9:11]
sys.path.insert(0, src); sys.path.insert(0, v19)
import tsplab as T
import merge as M
import lkh as L

xy = np.load(inst)
n = len(xy)
tours = [T.read_tour(f"{tourdir}/{name}.tour") for name in names]
assert all(T.is_valid_tour(t, n) for t in tours)
cost = [int(T.tour_length(xy, t)) for t in tours]
best = int(np.argmin(cost))
out = {"names": names, "cost": cost, "best": min(cost), "mean": float(np.mean(cost))}

def keys(t):
    a, b = t, np.roll(t, -1)
    return np.minimum(a, b) * n + np.maximum(a, b)

ks = [keys(t) for t in tours]
allk, cnt = np.unique(np.concatenate(ks), return_counts=True)
out["union_edges"] = int(len(allk))
out["common_to_all"] = int((cnt == len(tours)).sum())
out["share_of_best_tour_common"] = float(np.isin(ks[best], allk[cnt == len(tours)]).mean())
out["pairwise_shared_mean"] = float(np.mean([np.isin(ks[i], ks[j]).mean()
                                             for i in range(len(tours)) for j in range(i + 1, len(tours))]))
print(json.dumps(out), flush=True)

work = Path(work)
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)
parents1 = [(t + 1).tolist() for t in tours]


# (0) GA-EAX with the same tours as its whole population
d = work / "eax"
d.mkdir()
with open(d / "POP", "w") as f:
    for t, c in zip(tours, cost):
        f.write(f"{n} {c}\n" + " ".join(map(str, (t + 1).tolist())) + " \n")
t0 = time.time()
r = subprocess.run([eax_bin, "1", "M", str(len(tours)), "30", eax_tsp, "POP"], cwd=d, stdin=subprocess.DEVNULL,
                   capture_output=True, text=True)
tok = (d / "M_BestSol").read_text().split()
t = np.array(tok[2:2 + n], dtype=np.int64) - 1
assert T.is_valid_tour(t, n)
out["eax_merge"] = {"cost": int(T.tour_length(xy, t)), "wall": round(time.time() - t0, 2),
                    "generations": int((d / "M_Result").read_text().split()[2])}
T.write_tour(d / "merged.tour", t, name="eaxmerge", length=out["eax_merge"]["cost"])
print("eax_merge", json.dumps(out["eax_merge"]), flush=True)
KEYS = ("eax_merge", "gpx2", "union_mip", "lkh_merge_10")

# (1) pairwise GPX2 with LKH's own operator
g = L.gpx2_merge(Path(bindir) / "LKH", xy, parents1, 120.0, work / "gpx2")
out["gpx2"] = {"cost": int(g["cost"]), "seconds": round(g["seconds"], 2), "status": g["status"], **g["stats"]}
print("gpx2", json.dumps(out["gpx2"]), flush=True)

# (2) shortest tour in the union graph with the common edges kept (HiGHS, subtour cuts)
m = M.merge_tours(xy, parents1, mip_seconds)
out["union_mip"] = {"cost": int(m["cost"]), "seconds": round(m["seconds"], 2), "status": m["status"],
                    **{k: v for k, v in m["stats"].items() if not isinstance(v, (list, dict))}}
print("union", json.dumps(out["union_mip"]), flush=True)

# (3) LKH's own merge mode: common edges fixed, LKH searches the rest (may use edges outside the union)
for limit in (10,):
    d = work / f"lkhmerge_{limit}"
    d.mkdir()
    lines = [f"PROBLEM_FILE = {Path(tourdir).parent / 'dev.tsp'}", "RUNS = 1", "TRACE_LEVEL = 1", "SEED = 1",
             "CANDIDATE_SET_TYPE = POPMUSIC", "SUBGRADIENT = NO", "MAX_CANDIDATES = 5",
             f"TIME_LIMIT = {limit}", "MAX_TRIALS = 100000000",
             f"INITIAL_TOUR_FILE = {tourdir}/{names[best]}.tour", f"OUTPUT_TOUR_FILE = {d}/merged.tour"]
    lines += [f"MERGE_TOUR_FILE = {tourdir}/{name}.tour" for name in names]
    (d / "run.par").write_text("\n".join(lines) + "\n")
    t0 = time.time()
    r = subprocess.run([str(Path(bindir) / "LKH"), str(d / "run.par")], cwd=d, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True)
    (d / "run.log").write_text(r.stdout)
    t = T.read_tour(d / "merged.tour")
    assert T.is_valid_tour(t, n)
    out[f"lkh_merge_{limit}"] = {"cost": int(T.tour_length(xy, t)), "wall": round(time.time() - t0, 2)}
    print(f"lkh_merge_{limit}", json.dumps(out[f"lkh_merge_{limit}"]), flush=True)

b = out["best"]
for k in KEYS:
    out[k]["pct_vs_best"] = 100.0 * (out[k]["cost"] - b) / b
(work / "result.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: (out[k]["cost"], round(out[k]["pct_vs_best"], 5)) for k in KEYS}))
