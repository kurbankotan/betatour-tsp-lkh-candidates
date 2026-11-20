import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import tsplab as T
out, n, family, seed = sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
xy = T.to_int_coords(T.gen_instance(n, family, np.random.default_rng(seed)))
np.save(out + ".npy", xy)
T.write_tsp(out + ".tsp", xy, name="dev", ew="EUC_2D")
with open(out + "_eax.tsp", "w") as f:
    f.write(f"NAME : dev\nTYPE : TSP\nDIMENSION : {n}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
    for i, (x, y) in enumerate(xy, 1):
        f.write(f"{i} {int(x)} {int(y)}\n")
    f.write("EOF\n")
print(xy.shape, xy.dtype, xy.min(), xy.max(), len({(int(a), int(b)) for a, b in xy}))
