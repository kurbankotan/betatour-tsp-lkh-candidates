"""Shortest tour in the union graph of a population of tours, common edges kept (HiGHS with subtour cuts)."""
import sys, json, time
import numpy as np
src, v19, inst, pop, seconds, out = sys.argv[1:7]
sys.path.insert(0, src); sys.path.insert(0, v19)
import tsplab as T
import merge as M
xy = np.load(inst); n = len(xy)
tok = open(pop).read().split()
parents, i = [], 0
while i < len(tok):
    parents.append([int(x) for x in tok[i + 2:i + 2 + n]]); i += 2 + n
t0 = time.time()
m = M.merge_tours(xy, parents, float(seconds))
t = np.array(m["tour"], dtype=np.int64) - 1
assert T.is_valid_tour(t, n)
res = {"parents": len(parents), "cost": int(T.tour_length(xy, t)), "status": m["status"], "wall": round(time.time() - t0, 1),
       **{k: v for k, v in m["stats"].items() if not isinstance(v, (list, dict))}}
T.write_tour(out, t, name="union", length=res["cost"])
print(json.dumps(res))
