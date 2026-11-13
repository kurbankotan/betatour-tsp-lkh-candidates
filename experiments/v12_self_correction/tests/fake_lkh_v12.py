#!/usr/bin/env python3
"""Protocol stand-in for LKH-3 used only to test the run-v12 pipeline. It is not a solver.

What it imitates: a seed-dependent 'POPMUSIC' candidate set (written to CANDIDATE_FILE when the file does not exist,
in LKH's format), one trial per MAX_TRIALS = 1 call, in-place rewrites of OUTPUT_TOUR_FILE after every new best tour,
'* trial: Cost = ..., Time = ... sec.' lines (TRACE_LEVEL = 1), candidate rows extended with the edges of every new
best tour, and EUC_2D or CEIL_2D distances. A trial is a segment exchange followed by 2-opt whose first added edge is a
candidate; the closing edge need not be one, which is how edges outside C0 enter tours, as in LKH.

Test-only settings (environment variables): FAKE_LKH_TRIAL_SLEEP_S (default 0.01) pauses each trial.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path

INT_MAX = 2 ** 31 - 1


def read_par(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip().upper()] = value.strip()
    return values


def read_problem(path):
    coords, ew, inside = [], "EUC_2D", False
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.upper() == "EOF":
            break
        if text.upper().startswith("NODE_COORD_SECTION"):
            inside = True
            continue
        if inside:
            parts = text.split()
            coords.append((float(parts[1]), float(parts[2])))
        elif text.upper().startswith("EDGE_WEIGHT_TYPE"):
            ew = text.split(":", 1)[1].strip().upper()
    return coords, ew


class Fake:
    def __init__(self, par):
        self.t0 = time.time()
        self.par = par
        self.xy, self.ew = read_problem(par["PROBLEM_FILE"])
        self.n = len(self.xy)
        self.seed = int(par.get("SEED", 1))
        self.rng = random.Random(self.seed)
        self.k = int(par.get("MAX_CANDIDATES", 5))
        self.max_trials = int(float(par.get("MAX_TRIALS", 100000000)))
        self.time_limit = float(par.get("TIME_LIMIT", 1e9))
        self.sleep = float(os.environ.get("FAKE_LKH_TRIAL_SLEEP_S", "0.01"))
        self.rows = []
        for i in range(self.n):
            near = sorted((j for j in range(self.n) if j != i), key=lambda j: (self.d(i, j), j))[:self.k + 2]
            drop = set(self.rng.sample(range(len(near)), 2))   # seed-dependent deficits, like POPMUSIC
            kept = [j for r, j in enumerate(near) if r not in drop][:self.k]
            self.rows.append([[j, 100 * r] for r, j in enumerate(kept)])
        cand = par.get("CANDIDATE_FILE")
        if cand and not Path(cand).exists():
            lines = [str(self.n)]
            for i, row in enumerate(self.rows):
                lines.append(f"{i + 1} 0 {len(row)} " + " ".join(f"{to + 1} {a}" for to, a in row))
            Path(cand).write_text("\n".join(lines) + "\n-1\nEOF\n")
        self.best, self.best_cost = None, None

    def d(self, a, b):
        (x1, y1), (x2, y2) = self.xy[a], self.xy[b]
        r = math.hypot(x1 - x2, y1 - y2)
        return int(math.ceil(r)) if self.ew == "CEIL_2D" else int(math.floor(r + 0.5))

    def cost(self, tour):
        return sum(self.d(tour[i - 1], tour[i]) for i in range(self.n))

    def two_opt(self, tour):
        n = self.n
        improved = True
        while improved:
            improved = False
            pos = {v: i for i, v in enumerate(tour)}
            for a in range(n):
                sa = tour[(pos[a] + 1) % n]
                for b, _alpha in self.rows[a]:
                    if b == sa or self.d(a, b) >= self.d(a, sa):
                        continue
                    rot = tour[pos[a]:] + tour[:pos[a]]
                    j = rot.index(b)
                    bn = rot[(j + 1) % n]
                    if j < 2 or bn == a:
                        continue
                    delta = self.d(a, b) + self.d(sa, bn) - self.d(a, sa) - self.d(b, bn)
                    if delta < 0:
                        tour = [a] + rot[1:j + 1][::-1] + rot[j + 1:]
                        improved = True
                        break
                if improved:
                    break
        return tour

    def write_tour(self):
        path = self.par.get("OUTPUT_TOUR_FILE")
        if not path:
            return
        text = (f"NAME : fake.{self.best_cost}.tour\nCOMMENT : Length = {self.best_cost}\nTYPE : TOUR\n"
                f"DIMENSION : {self.n}\nTOUR_SECTION\n" + "".join(f"{v + 1}\n" for v in self.best) + "-1\nEOF\n")
        with open(path, "w") as handle:                     # in place, in chunks, like LKH's WriteTour
            for i in range(0, len(text), 256):
                handle.write(text[i:i + 256])
                handle.flush()

    def adjust(self):
        pos = {v: i for i, v in enumerate(self.best)}
        for i in range(self.n):
            for nb in (self.best[pos[i] - 1], self.best[(pos[i] + 1) % self.n]):
                if all(to != nb for to, _ in self.rows[i]):
                    self.rows[i].append([nb, INT_MAX])

    def report(self, trial):
        print(f"* {trial}: Cost = {self.best_cost}, Gap = 0.0000%, Time = {time.time() - self.t0:.2f} sec.",
              flush=True)

    def run(self):
        start = self.rng.randrange(self.n)
        tour, left = [start], set(range(self.n)) - {start}
        while left:
            last = tour[-1]
            nxt = min(left, key=lambda j: (self.d(last, j), j))
            tour.append(nxt)
            left.remove(nxt)
        tour = self.two_opt(tour)
        self.best, self.best_cost = tour, self.cost(tour)
        self.write_tour()
        self.report(1)
        self.adjust()
        trial = 1
        while trial < self.max_trials and time.time() - self.t0 < self.time_limit:
            time.sleep(self.sleep)
            trial += 1
            p, q, r = sorted(self.rng.sample(range(1, self.n), 3))
            cand = self.best[:p] + self.best[q:r] + self.best[p:q] + self.best[r:]
            cand = self.two_opt(cand)
            c = self.cost(cand)
            if c < self.best_cost:
                self.best, self.best_cost = cand, c
                self.write_tour()
                self.report(trial)
                self.adjust()
        print(f"Cost.min = {self.best_cost}, Cost.avg = {self.best_cost}", flush=True)


if __name__ == "__main__":
    Fake(read_par(sys.argv[1])).run()
