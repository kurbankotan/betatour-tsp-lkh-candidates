"""Where do the final LKH tours differ from the shorter GA-EAX tour? (development probe)"""
import sys, numpy as np
src, inst, ref_file, tourdir = sys.argv[1:5]
extra = sys.argv[5:]
sys.path.insert(0, src); import tsplab as T
xy = np.load(inst); n = len(xy)
def keys(t):
    return np.minimum(t, np.roll(t, -1)) * n + np.maximum(t, np.roll(t, -1))
ref = T.read_tour(ref_file); kr = keys(ref); Lr = int(T.tour_length(xy, ref))
names = [f"s{s}" for s in range(101, 107)]
tours = [T.read_tour(f"{tourdir}/{x}.tour") for x in names]
ks = [keys(t) for t in tours]
allk, cnt = np.unique(np.concatenate(ks), return_counts=True)
common = allk[cnt == 6]
print("reference", Lr, "| union edges", len(allk), "| common to six", len(common))
print("reference edges in union:", int(np.isin(kr, allk).sum()), "of", n, "| missing from every LKH tour:", int((~np.isin(kr, allk)).sum()))
print("common-to-six edges absent from reference:", int((~np.isin(common, kr)).sum()), f"({100 * (~np.isin(common, kr)).mean():.2f}% of common)")
for x, t, k in zip(names, tours, ks):
    L = int(T.tour_length(xy, t)); wrong = ~np.isin(k, kr); inc = np.isin(k, common)
    print(f"{x}: {L} (+{100 * (L - Lr) / Lr:.4f}%) edges not in reference {int(wrong.sum())}, of these common to six "
          f"{int((wrong & inc).sum())} ({100 * (wrong & inc).sum() / max(1, wrong.sum()):.0f}%)")
for f in extra:
    t = T.read_tour(f); k = keys(t); L = int(T.tour_length(xy, t)); wrong = ~np.isin(k, kr)
    print(f"{f.split('/')[-2]}: {L} (+{100 * (L - Lr) / Lr:.4f}%) edges not in reference {int(wrong.sum())}, outside union {int((~np.isin(k, allk)).sum())}")
