#!/usr/bin/env python3
"""Protocol stand-in for LKH-3 used only to test the BetaTour v11 Python pipeline.

Native mode (no BETATOUR_HOOK_DIR) rewrites OUTPUT_TOUR_FILE in place after
each better tour, as LKH does. Hook mode writes the files of the v11 search
hook: native/live candidates, the first-trial barrier, action application,
atomic best tours with events, and telemetry with the hook's key names. The
search is 2-opt over the candidate rows with double-bridge restarts. It is not
a solver and its tour lengths mean nothing.

Test-only settings (environment variables):
  FAKE_LKH_TRUNCATE_AFTER_S  native: after this time, start rewriting the tour
                             file and hang half-way (a kill mid-rewrite)
  FAKE_LKH_TRIAL_SLEEP_S     pause per trial (default 0.03)
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path

INT_MAX = 2**31 - 1
TELEMETRY_KEYS = (
    "telemetry_snapshot_epoch_s", "action_count", "total_candidate_enumerations",
    "action_candidate_enumerations", "distinct_actions_enumerated",
    "top_level_candidate_enumerations", "top_level_action_enumerations",
    "top_level_action_first_gate_passes", "top_level_action_positive_g1",
    "action_accepted_edge_entries", "distinct_actions_entering_accepted_tours",
    "better_tour_events_after_action", "action_incumbent_entries",
    "distinct_actions_ever_in_better_tour", "current_better_tour_actions",
    "trials_completed_after_action", "action_tsv_snapshot_trial",
    "slot_floor_enforced", "slot_checks", "slot_restorations",
    "slot_restoration_displacement", "slot_found_ahead",
    "slot_ahead_at_last_check", "slot_missing")


def read_par(path: Path) -> dict[str, str]:
    values = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_coordinates(path: Path) -> list[tuple[float, float]]:
    coordinates, inside = [], False
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if text == "NODE_COORD_SECTION":
            inside = True
        elif text == "EOF":
            break
        elif inside and text:
            _node, x, y = text.split()[:3]
            coordinates.append((float(x), float(y)))
    return coordinates


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


class FakeLKH:
    def __init__(self, par: dict[str, str]):
        self.par = par
        self.xy = read_coordinates(Path(par["PROBLEM_FILE"]))
        self.n = len(self.xy)
        self.seed = int(par.get("SEED", 1))
        self.max_trials = int(float(par.get("MAX_TRIALS", 100_000_000)))
        self.rng = random.Random(self.seed)
        self.hook = os.environ.get("BETATOUR_HOOK_DIR")
        self.hook = Path(self.hook) if self.hook else None
        self.start = time.time()
        self.trial_sleep = float(os.environ.get("FAKE_LKH_TRIAL_SLEEP_S", "0.03"))
        k = int(par.get("MAX_CANDIDATES", 5))
        self.rows = []
        for i in range(self.n):
            near = sorted((j for j in range(self.n) if j != i), key=lambda j: (self.d(i, j), j))[:k]
            self.rows.append([[j, 10 * rank] for rank, j in enumerate(near)])
        self.action_to: list[int] | None = None
        self.action_slot: list[int] = []
        self.c = dict.fromkeys(TELEMETRY_KEYS, 0)
        self.c["slot_floor_enforced"] = 1
        self.per_action: dict[int, list[int]] = {}
        self.ever_in_best: set[int] = set()
        self.last_full = 0.0
        self.best: list[int] = []
        self.best_cost = None

    def d(self, a: int, b: int) -> int:
        (x1, y1), (x2, y2) = self.xy[a], self.xy[b]
        return int(math.floor(math.hypot(x1 - x2, y1 - y2) + 0.5))

    def cost(self, tour: list[int]) -> int:
        return sum(self.d(tour[i - 1], tour[i]) for i in range(self.n))

    def is_action(self, a: int, b: int) -> bool:
        return bool(self.action_to) and (self.action_to[a] == b + 1 or self.action_to[b] == a + 1)

    def two_opt(self, tour: list[int]) -> list[int]:
        n = self.n
        improved = True
        while improved:
            improved = False
            position = {node: index for index, node in enumerate(tour)}
            for a in range(n):
                successor = tour[(position[a] + 1) % n]
                for b, _alpha in self.rows[a]:
                    self.c["total_candidate_enumerations"] += 1
                    self.c["top_level_candidate_enumerations"] += 1
                    action = bool(self.action_to) and self.action_to[a] == b + 1
                    if action:
                        self.c["action_candidate_enumerations"] += 1
                        self.c["top_level_action_enumerations"] += 1
                        self.per_action[a][0] += 1
                    gain = self.d(a, successor) - self.d(a, b)
                    if gain <= 0 or b == successor:
                        continue
                    if action:
                        self.c["top_level_action_first_gate_passes"] += 1
                        self.c["top_level_action_positive_g1"] += 1
                        self.per_action[a][1] += 1
                    rotated = tour[position[a]:] + tour[:position[a]]
                    j = rotated.index(b)
                    b_next = rotated[(j + 1) % n]
                    if j < 2 or b_next == a:
                        continue
                    delta = (self.d(a, b) + self.d(successor, b_next)
                             - self.d(a, successor) - self.d(b, b_next))
                    if delta < 0:
                        tour = [a] + rotated[1:j + 1][::-1] + rotated[j + 1:]
                        for x, y in ((a, b), (successor, b_next)):
                            if self.is_action(x, y):
                                source = x if self.action_to[x] == y + 1 else y
                                self.c["action_accepted_edge_entries"] += 1
                                self.per_action[source][2] += 1
                        improved = True
                        break
                if improved:
                    break
        return tour

    def nearest_neighbour_tour(self) -> list[int]:
        start = self.seed % self.n
        tour, left = [start], set(range(self.n)) - {start}
        while left:
            last = tour[-1]
            nxt = min(left, key=lambda j: (self.d(last, j), j))
            tour.append(nxt)
            left.remove(nxt)
        return tour

    def tour_text_hook(self) -> str:
        return f"{self.n}\n" + "".join(f"{node + 1}\n" for node in self.best)

    def tour_text_tsplib(self) -> str:
        return (f"NAME : fake.{self.best_cost}.tour\nCOMMENT : Length = {self.best_cost}\n"
                f"TYPE : TOUR\nDIMENSION : {self.n}\nTOUR_SECTION\n"
                + "".join(f"{node + 1}\n" for node in self.best) + "-1\nEOF\n")

    def write_output_in_place(self, cut: bool = False) -> None:
        path = self.par.get("OUTPUT_TOUR_FILE")
        if not path:
            return
        text = self.tour_text_tsplib()
        with open(path, "w") as handle:
            stop = len(text) // 2 if cut else len(text)
            for index in range(0, stop, 512):
                handle.write(text[index:min(index + 512, stop)])
                handle.flush()
            if cut:
                while True:
                    time.sleep(3600)

    def write_candidates(self, name: str) -> None:
        lines = [str(self.n)]
        for i, row in enumerate(self.rows):
            lines.append(f"{i + 1} {len(row)}" + "".join(f" {to + 1} {alpha}" for to, alpha in row))
        atomic_write(self.hook / name, "\n".join(lines) + "\n")

    def tour_edges(self, tour: list[int]) -> set[tuple[int, int]]:
        return {(min(tour[i - 1], tour[i]), max(tour[i - 1], tour[i])) for i in range(self.n)}

    def record_best(self, tour: list[int], cost: int, after_action: bool) -> None:
        previous = self.tour_edges(self.best) if self.best else set()
        self.best, self.best_cost = list(tour), cost
        if self.hook:
            atomic_write(self.hook / "best.tour", self.tour_text_hook())
            with open(self.hook / "best.events", "a") as events:
                events.write(f"{time.time():.6f} {cost}\n")
            if after_action:
                self.c["better_tour_events_after_action"] += 1
                now = self.tour_edges(self.best)
                current = 0
                for source, target in enumerate(self.action_to or []):
                    if not target:
                        continue
                    edge = (min(source, target - 1), max(source, target - 1))
                    if edge in now:
                        current += 1
                        self.ever_in_best.add(source)
                        if edge not in previous:
                            self.c["action_incumbent_entries"] += 1
                            self.per_action[source][3] += 1
                self.c["current_better_tour_actions"] = current
                self.write_telemetry(full=False)
        else:
            self.write_output_in_place()

    def adjust_candidates(self) -> None:
        position = {node: index for index, node in enumerate(self.best)}
        for i in range(self.n):
            neighbours = (self.best[position[i] - 1], self.best[(position[i] + 1) % self.n])
            row = self.rows[i]
            for neighbour in neighbours:
                if all(to != neighbour for to, _ in row):
                    row.append([neighbour, INT_MAX])
            for neighbour in neighbours:  # imitate a reorder that favours tour edges
                index = next(k for k, (to, _) in enumerate(row) if to == neighbour)
                row.insert(0, row.pop(index))

    def reassert_slots(self) -> None:
        if not self.action_to:
            return
        self.c["slot_checks"] += 1
        self.c["slot_ahead_at_last_check"] = 0
        for source, target in enumerate(self.action_to):
            if not target:
                continue
            row = self.rows[source]
            index = next(k for k, (to, _) in enumerate(row) if to == target - 1)
            destination = min(self.action_slot[source] - 1, len(row) - 1)
            if index < destination:
                self.c["slot_found_ahead"] += 1
                self.c["slot_ahead_at_last_check"] += 1
            elif index > destination:
                row.insert(destination, row.pop(index))
                self.c["slot_restorations"] += 1
                self.c["slot_restoration_displacement"] += index - destination
                self.per_action[source][4] += 1

    def write_telemetry(self, full: bool) -> None:
        self.c["telemetry_snapshot_epoch_s"] = time.time()
        actions = [s for s, t in enumerate(self.action_to or []) if t]
        self.c["action_count"] = len(actions)
        self.c["distinct_actions_enumerated"] = sum(self.per_action[s][0] > 0 for s in actions)
        self.c["distinct_actions_entering_accepted_tours"] = sum(self.per_action[s][2] > 0 for s in actions)
        self.c["distinct_actions_ever_in_better_tour"] = len(self.ever_in_best)
        lines = [f"{key} {self.c[key]:.6f}" if key == "telemetry_snapshot_epoch_s"
                 else f"{key} {int(self.c[key])}" for key in TELEMETRY_KEYS]
        atomic_write(self.hook / "telemetry.txt", "\n".join(lines) + "\n")
        if full:
            self.last_full = time.time()
            rows = ["from\tto\tenumerations\tfirst_gate_passes\tpositive_g1\t"
                    "accepted_edge_entries\tincumbent_entries\tever_in_better_tour\t"
                    "currently_in_better_tour\tslot_restorations"]
            now = self.tour_edges(self.best)
            for s in actions:
                t = self.action_to[s] - 1
                e, g, a, inc, r = self.per_action[s]
                rows.append(f"{s + 1}\t{t + 1}\t{e}\t{g}\t{g}\t{a}\t{inc}\t{int(s in self.ever_in_best)}\t"
                            f"{int((min(s, t), max(s, t)) in now)}\t{r}")
            atomic_write(self.hook / "action_telemetry.tsv", "\n".join(rows) + "\n")

    def wait_and_apply(self) -> None:
        action_path = self.hook / "action.txt"
        timeout = float(os.environ.get("BETATOUR_HOOK_TIMEOUT_S", "300"))
        waited = time.time()
        while not action_path.exists():
            if time.time() - waited > timeout:
                sys.exit(97)
            time.sleep(0.01)
        lines = action_path.read_text().split("\n")
        if int(lines[0]) != self.n:
            sys.exit(97)
        self.action_to = [0] * self.n
        self.action_slot = [0] * self.n
        added = moved = 0
        for line in lines[1:self.n + 1]:
            source, target, slot, alpha = (int(x) for x in line.split())
            if not target:
                continue
            row = self.rows[source - 1]
            if any(to == target - 1 for to, _ in row):
                sys.exit(97)
            destination = min(slot - 1, len(row))
            moved += destination != len(row)
            row.insert(destination, [target - 1, alpha])
            added += 1
            self.action_to[source - 1] = target
            self.action_slot[source - 1] = slot
            self.per_action[source - 1] = [0, 0, 0, 0, 0]
        self.write_candidates("post.candidates")
        atomic_write(self.hook / "applied.stats",
                     f"added {added}\nexisting 0\nmoved {moved}\nrejected 0\n"
                     f"wait_wall_s {time.time() - waited:.6f}\n")
        self.write_telemetry(full=True)

    def run(self) -> None:
        if self.hook:
            self.write_candidates("native.candidates")
        tour = self.two_opt(self.nearest_neighbour_tour())
        self.record_best(tour, self.cost(tour), after_action=False)
        trial = 1
        if self.hook:
            self.adjust_candidates()
            self.write_candidates("live.candidates")
            atomic_write(self.hook / "first.tour", self.tour_text_hook())
            self.wait_and_apply()
        truncate_after = os.environ.get("FAKE_LKH_TRUNCATE_AFTER_S")
        while trial < self.max_trials:
            if not self.hook and truncate_after and time.time() - self.start > float(truncate_after):
                self.write_output_in_place(cut=True)
            time.sleep(self.trial_sleep)
            trial += 1
            candidate = list(self.best)
            cuts = sorted(self.rng.sample(range(1, self.n), 3))
            p, q, r = cuts
            candidate = candidate[:p] + candidate[q:r] + candidate[p:q] + candidate[r:]
            candidate = self.two_opt(candidate)
            cost = self.cost(candidate)
            if cost < self.best_cost:
                self.record_best(candidate, cost, after_action=self.action_to is not None)
                if self.hook:
                    self.adjust_candidates()
            if self.hook and self.action_to is not None:
                self.c["trials_completed_after_action"] += 1
                self.reassert_slots()
                full = time.time() - self.last_full >= 1.0
                if full:
                    self.c["action_tsv_snapshot_trial"] = trial
                self.write_telemetry(full=full)
        if not self.hook:
            self.write_output_in_place()


if __name__ == "__main__":
    FakeLKH(read_par(Path(sys.argv[1]))).run()
