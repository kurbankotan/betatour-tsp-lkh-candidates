import sys, numpy as np
src, inst, sol = sys.argv[1:4]
sys.path.insert(0, src); import tsplab as T
xy = np.load(inst); n = len(xy)
tok = open(sol).read().split()
out = []
i = 0
while i < len(tok):
    N, val = int(tok[i]), int(tok[i + 1]); t = np.array(tok[i + 2:i + 2 + N], dtype=np.int64) - 1; i += 2 + N
    assert N == n and T.is_valid_tour(t, n)
    out.append((val, int(T.tour_length(xy, t))))
print(out)
if len(sys.argv) > 4:
    T.write_tour(sys.argv[4], t, name="eax", length=out[-1][1])
