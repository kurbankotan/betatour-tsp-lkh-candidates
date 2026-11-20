"""Best tour length of each LKH log at given times of LKH's own clock (search time, preprocessing not counted)."""
import sys, re
times = [float(x) for x in sys.argv[1].split(",")]
rx = re.compile(r"^\* (\d+): Cost = (\d+), Time = ([0-9.]+) sec")
rows = {}
for path in sys.argv[2:]:
    ev = [(float(m.group(3)), int(m.group(2)), int(m.group(1))) for m in map(rx.match, open(path)) if m]
    pre = re.search(r"Preprocessing time = ([0-9.]+)", open(path).read())
    rows[path.split("/")[-1]] = ([min((c for t, c, _ in ev if t <= T), default=None) for T in times],
                                 [max((k for t, c, k in ev if t <= T), default=None) for T in times],
                                 float(pre.group(1)) if pre else None)
for k, (c, tr, pre) in rows.items():
    print(k, "pre", pre, "best:", c, "last improving trial:", tr)
for i, T in enumerate(times):
    v = [r[0][i] for r in rows.values() if r[0][i] is not None]
    print(f"T={T:g}: best of {len(v)} = {min(v)}, mean = {sum(v) / len(v):.1f}")
