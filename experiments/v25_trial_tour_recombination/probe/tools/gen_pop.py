"""Population of LKH first-trial tours for GA-EAX (development probe).
One LKH run writes the POPMUSIC-5 candidate file; the other runs read it. One trial per run."""
import sys, os, time, subprocess, resource
from pathlib import Path
import numpy as np

src, lkh, problem, inst, out, n_tours, core = sys.argv[1:8]
n_tours, core = int(n_tours), int(core)
sys.path.insert(0, src)
import tsplab as T

xy = np.load(inst)
n = len(xy)
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
cand = out / "pop.cand"
t0 = time.time()
tours, costs = [], []
for i in range(n_tours):
    seed = 2001 + i
    par = out / "run.par"
    tour_file = out / "run.tour"
    if tour_file.exists():
        tour_file.unlink()
    par.write_text("\n".join([
        f"PROBLEM_FILE = {problem}", "RUNS = 1", "TRACE_LEVEL = 0", "MAX_TRIALS = 1", f"SEED = {seed}",
        "CANDIDATE_SET_TYPE = POPMUSIC", "SUBGRADIENT = NO", "MAX_CANDIDATES = 5",
        f"CANDIDATE_FILE = {cand}", f"OUTPUT_TOUR_FILE = {tour_file}"]) + "\n")
    r = subprocess.run(["taskset", "-c", str(core), lkh, str(par)], cwd=out, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-500:]
    t = T.read_tour(tour_file)
    assert T.is_valid_tour(t, n)
    tours.append(t)
    costs.append(int(T.tour_length(xy, t)))
    if i in (0, 1, 9) or (i + 1) % 50 == 0:
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        print(i + 1, costs[-1], f"cpu {ru.ru_utime + ru.ru_stime:.1f} s wall {time.time() - t0:.1f} s", flush=True)
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
with open(out / "POP", "w") as f:
    for t, c in zip(tours, costs):
        f.write(f"{n} {c}\n" + " ".join(map(str, (t + 1).tolist())) + " \n")
keys = [np.minimum(t, np.roll(t, -1)) * n + np.maximum(t, np.roll(t, -1)) for t in tours]
share = float(np.mean([np.isin(keys[i], keys[i + 1]).mean() for i in range(len(tours) - 1)]))
print({"tours": n_tours, "best": min(costs), "mean": float(np.mean(costs)), "worst": max(costs),
       "distinct": len({tuple(k.tolist()) for k in map(np.sort, keys)}),
       "pairwise_shared_edges": round(share, 4),
       "child_cpu_s": round(ru.ru_utime + ru.ru_stime, 1), "wall_s": round(time.time() - t0, 1)})
