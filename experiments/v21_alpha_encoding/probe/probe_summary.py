"""Summary of the exploratory alpha-encoding probe (see README.md). Reads the LKH logs of probe_setup.py and
probe_arms.py and writes probe_results.csv (one row per instance and arm) and probe_summary.txt.

    python probe_summary.py DATA_DIR [OUT_DIR]

Clock: LKH's own time since the search of each process began. The uninterrupted run is shifted by the end time of
its first trial, so every arm is read on the time since the first trial."""

from __future__ import annotations

import csv
import random
import re
import statistics as st
import sys
from math import comb
from pathlib import Path

TRIAL = re.compile(r"^(\*?)\s*(\d+): Cost = (\d+), Time = ([\d.]+) sec")
SHARE = re.compile(r"first-tour edges trusted \((.+?)\): forward ([\d.]+), backward ([\d.]+)")
ARMS = ["uninterrupted", "native", "rank", "top2", "none", "zero", "schedB_none", "schedB_zero"]
TIMES = [5, 10, 15, 20, 30, 45, 59]


def sign_p(w, l):
    n, k = w + l, min(w, l)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def load(data, tag, arm):
    path = data / tag / f"{arm}.log"
    if not path.exists():
        return None
    rows = [(int(m.group(2)), int(m.group(3)), float(m.group(4)))
            for m in map(TRIAL.match, (line.strip() for line in path.read_text().split("\n"))) if m]
    if not rows:
        return None
    t0, k0 = (rows[0][2], 1) if arm == "uninterrupted" else (0.0, 0)
    best, out = None, []
    for k, cost, t in rows:
        best = cost if best is None else min(best, cost)
        out.append((k - k0, best, t - t0))
    return out


def at(traj, x):
    seen = [b for _, b, t in traj if t <= x]
    return seen[-1] if seen else traj[0][1]


def main():
    data = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    tags = sorted(p.name for p in data.glob("s1??") if (p / "uninterrupted.log").exists())
    runs = {t: {a: load(data, t, a) for a in ARMS} for t in tags}
    shares = {}
    for t in tags:
        texts = [p.read_text() for p in (data / f"{t}.setup.out", data / t / "setup.out") if p.exists()]
        for label, fwd, bwd in SHARE.findall("\n".join(texts)):
            shares[(t, label)] = (float(fwd) + float(bwd)) / 2
    with open(out / "probe_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "arm", "trials", "last_time_s"] + [f"best_at_{x}s" for x in TIMES])
        for t in tags:
            for a in ARMS:
                r = runs[t][a]
                if r:
                    w.writerow([t, a, r[-1][0], f"{r[-1][2]:.2f}"] + [at(r, x) for x in TIMES])
    lines = []
    full = [t for t in tags if all(runs[t][a] and runs[t][a][-1][2] >= 55 for a in ("uninterrupted", "native", "rank"))]
    lines.append(f"{len(full)} instances with the arms uninterrupted, native and rank")
    if not full:                                    # the comparisons need runs of at least 55 s
        (out / "probe_summary.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    for a in ("uninterrupted", "native", "rank"):
        lines.append(f"  trials after 60 s, {a:<14} mean {st.mean(runs[t][a][-1][0] for t in full):6.1f}")
    for label in ("native alpha==0", "rank-1 only"):
        vals = [shares[(t, label)] for t in full if (t, label) in shares]
        if vals:
            lines.append(f"  share of first-trial tour edges with alpha = 0, {label}: {st.mean(vals):.3f} ({len(vals)} instances)")
    rng = random.Random(1)

    def compare(a, b, group, times):
        lines.append(f"{a} minus {b} (% of tour length; positive: {a} is longer), {len(group)} instances")
        for x in times:
            d = [100 * (at(runs[t][a], x) - at(runs[t][b], x)) / at(runs[t][b], x) for t in group]
            worse, better = sum(v > 0 for v in d), sum(v < 0 for v in d)
            boot = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(4000))
            lines.append(f"   after {x:>2} s: mean {st.mean(d):+.4f}, median {st.median(d):+.4f}, 95% CI {boot[100]:+.4f} "
                         f"to {boot[3899]:+.4f}, longer/equal/shorter {worse}/{len(d) - worse - better}/{better}, "
                         f"sign test p = {sign_p(worse, better):.3f}")

    for a, b in (("rank", "native"), ("rank", "uninterrupted"), ("native", "uninterrupted")):
        compare(a, b, full, [10, 15, 20, 30, 45, 59])
    six = [t for t in full if all(runs[t][a] and runs[t][a][-1][2] >= 55 for a in ("top2", "none", "zero"))]
    for a in ("top2", "none", "zero"):
        compare(a, "native", six, [15, 30, 59])
    two = [t for t in six if runs[t]["schedB_none"] and runs[t]["schedB_zero"]]
    if two:
        d = [100 * (runs[t]["schedB_zero"][-1][1] - runs[t]["schedB_none"][-1][1]) / runs[t]["schedB_none"][-1][1] for t in two]
        worse, better = sum(v > 0 for v in d), sum(v < 0 for v in d)
        lines.append(f"two-phase schedule (30 s without kept edges, then 29 s): all candidates trusted minus none trusted, "
                     f"{len(two)} instances: mean {st.mean(d):+.4f}, longer/equal/shorter {worse}/{len(d) - worse - better}/{better}")
    (out / "probe_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
