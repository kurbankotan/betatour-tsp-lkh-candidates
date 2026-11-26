"""Certified candidate graphs: decision pilot (not a run of the BetaTour study).

Question. Can a learned model propose sparse candidate lists whose completeness a lower-bound certificate proves? An
edge is eliminated only when every tour that uses it is provably longer than a known tour; the edges that remain form
a graph that contains every optimal tour. A node's candidate list is certified complete when it contains all of the
node's remaining edges.

Certificates.
C1  The 1-tree bound with LKH-3's node penalties (pi) and the alpha-values of all edges of the complete graph, in
    exact integer arithmetic (this module). A tour that uses edge e costs at least W + alpha(e) (in units of
    1/PRECISION), where W is the bound.
C2  Concorde's root LP with cutting planes: its lower bound and the reduced costs that `concorde -z k` writes (the k
    lowest per node). A tour that uses edge e costs at least LB + rc(e) when rc are the reduced costs of a dual
    solution of value LB. Floating point; used only when the checks of `c2_checks` pass against a proven optimal tour.

All functions are pure numpy except the Concorde download, build and run helpers.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

VERSION = "certpilot-1"
PRECISION = 100          # LKH-3 default: node penalties are in units of 1/PRECISION of a distance unit
BIG = np.iinfo(np.int64).max // 4

# ---------------------------------------------------------------------------------------------- Concorde build
CONCORDE_SOURCES = {
    "co031219.tgz": ["https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz",
                     "http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz"],
    "qsopt.a": ["https://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.PIC.a",
                "http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.PIC.a"],
    "qsopt.h": ["https://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.h",
                "http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.h"],
}
CFLAGS = "-O3 -fcommon -fPIC"


def download(urls, dest, timeout=180, min_bytes=200):
    """First URL that answers; returns (url, sha256)."""
    last = None
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (certpilot)"})
            with urllib.request.urlopen(request, timeout=timeout) as response, open(dest, "wb") as f:
                shutil.copyfileobj(response, f)
            data = Path(dest).read_bytes()
            if len(data) < min_bytes:
                raise RuntimeError(f"only {len(data)} bytes")
            return url, hashlib.sha256(data).hexdigest()
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"download failed ({urls[0]}): {last}")


def _run(cmd, cwd, env=None, timeout=1800):
    proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          errors="replace", timeout=timeout)
    return proc.returncode, proc.stdout


def build_concorde(root, jobs=4):
    """Concorde (co031219) with the QSopt LP solver, from the official University of Waterloo pages. Returns
    {'binary', 'sources': {file: {'url', 'sha256'}}, 'cflags', 'attempts'}. A finished build is reused."""
    root = Path(root)
    binary = root / "concorde" / "TSP" / "concorde"
    marker = root / "build.json"
    if binary.exists() and marker.exists():
        info = json.loads(marker.read_text())
        info["binary"] = str(binary)
        return info
    shutil.rmtree(root, ignore_errors=True)
    (root / "qsopt").mkdir(parents=True)
    sources = {"co031219.tgz": download(CONCORDE_SOURCES["co031219.tgz"], root / "co031219.tgz", min_bytes=100000),
               "qsopt.a": download(CONCORDE_SOURCES["qsopt.a"], root / "qsopt" / "qsopt.a", min_bytes=100000),
               "qsopt.h": download(CONCORDE_SOURCES["qsopt.h"], root / "qsopt" / "qsopt.h")}
    with tarfile.open(root / "co031219.tgz") as tar:
        try:
            tar.extractall(root, filter="data")
        except TypeError:
            tar.extractall(root)
    src = root / "concorde"
    assert (src / "configure").exists(), "Concorde source folder is missing"
    qsopt = str((root / "qsopt").resolve())
    attempts = []
    for libs in ("", "-lpthread -lm"):
        env = dict(os.environ, CFLAGS=CFLAGS)
        if libs:
            env["LIBS"] = libs
        code, out = _run(["./configure", f"--with-qsopt={qsopt}"], src, env)
        if code != 0:
            # An old config.sub/config.guess may not know this machine: use the system's newer copies and retry.
            for name in ("config.sub", "config.guess"):
                found = sorted(glob.glob(f"/usr/share/misc/{name}") + glob.glob(f"/usr/share/automake-*/{name}")
                               + glob.glob(f"/usr/share/libtool/build-aux/{name}"))
                if found:
                    shutil.copy2(found[-1], src / name)
            code, out = _run(["./configure", f"--with-qsopt={qsopt}"], src, env)
        attempts.append({"libs": libs, "configure": code})
        if code != 0:
            continue
        code, out = _run(["make", "-j", str(max(1, int(jobs)))], src, env)
        if code != 0 or not binary.exists():
            code, out = _run(["make"], src, env)
        attempts[-1]["make"] = code
        if code == 0 and binary.exists():
            break
    if not binary.exists():
        raise RuntimeError("Concorde build failed:\n" + out[-4000:])
    info = {"sources": {k: {"url": u, "sha256": s} for k, (u, s) in sources.items()}, "cflags": CFLAGS,
            "attempts": attempts}
    marker.write_text(json.dumps(info, indent=1))
    info["binary"] = str(binary)
    return info


# ------------------------------------------------------------------------------------------------ Concorde runs
NUM = r"(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
RX = {"lp": re.compile(r"LP Value\s+\d+:\s*" + NUM),
      "exact": re.compile(r"Exact lower bound:\s*" + NUM),
      "final": re.compile(r"Final lower bound\s*" + NUM + r",\s*upper bound\s*" + NUM),
      "optimal": re.compile(r"Optimal Solution:\s*" + NUM),
      "bbnodes": re.compile(r"Number of bbnodes:\s*(\d+)"),
      "time": re.compile(r"Total Running Time:\s*" + NUM)}


def concorde_usage(binary):
    """The help text that concorde prints when it is run without arguments."""
    try:
        proc = subprocess.run([str(binary)], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace", timeout=60)
        return proc.stdout
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout
        return (out.decode("latin-1", "replace") if isinstance(out, bytes) else (out or "")) + "\n[timeout]"


def parse_concorde(text):
    lp = [float(x) for x in RX["lp"].findall(text)]
    exact = [float(x) for x in RX["exact"].findall(text)]
    final = [(float(a), float(b)) for a, b in RX["final"].findall(text)]
    optimal = RX["optimal"].findall(text)
    bbnodes = RX["bbnodes"].findall(text)
    total = RX["time"].findall(text)
    return {"lp_last": lp[-1] if lp else None, "lp_count": len(lp), "exact_last": exact[-1] if exact else None,
            "final_lb": final[-1][0] if final else None, "final_ub": final[-1][1] if final else None,
            "optimal": float(optimal[-1]) if optimal else None, "bbnodes": int(bbnodes[-1]) if bbnodes else None,
            "total_time": float(total[-1]) if total else None}


def root_lower_bound(parsed):
    """The smallest of the root LP value, the exact lower bound and the final lower bound that the run printed. A
    smaller value keeps the certificate valid (it can only eliminate fewer edges)."""
    values = [v for v in (parsed.get("lp_last"), parsed.get("exact_last"), parsed.get("final_lb")) if v is not None]
    return min(values) if values else None


def run_concorde(binary, problem_file, workdir, args, timeout):
    """Runs concorde on a copy of the problem named p.tsp in an empty folder (its files are written there)."""
    workdir = Path(workdir)
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    shutil.copy2(problem_file, workdir / "p.tsp")
    cmd = [str(binary)] + [str(a) for a in args] + ["p.tsp"]
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=workdir, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace", timeout=timeout)
        text, code, timed_out = proc.stdout, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout
        text = out.decode("latin-1", "replace") if isinstance(out, bytes) else (out or "")
        code, timed_out = None, True
    wall = time.time() - start
    (workdir / "stdout.txt").write_text(text)
    result = {"cmd": [Path(cmd[0]).name] + cmd[1:], "returncode": code, "timed_out": timed_out, "wall": wall,
              "workdir": str(workdir)}
    result.update(parse_concorde(text))
    return result


def read_concorde_tour(path, n):
    """Concorde's tour file: n, then the n nodes (0-based, in input order)."""
    values = np.array(Path(path).read_text().split(), dtype=np.int64)
    assert len(values) >= n + 1 and values[0] == n, "unexpected tour file"
    tour = values[1:n + 1]
    assert np.array_equal(np.sort(tour), np.arange(n)), "tour file is not a permutation"
    return tour


def find_rcn(workdir):
    found = sorted(Path(workdir).glob("*.rcn"))
    return found[0] if found else None


def read_rcn(path, n):
    """Reduced costs written by `concorde -z k`. Accepts an optional 'ncount ecount' header and lines 'u v rc'.
    Returns the distinct undirected edges (key = min*n + max, sorted), their reduced costs (the smaller value when an
    edge is listed twice; the largest difference between two listings is reported), the number of lines and the
    number of listed edges at each node."""
    with open(path) as f:
        head = f.readline().split()
    skip = 1 if len(head) == 2 else 0
    try:
        import pandas as pd
        frame = pd.read_csv(path, sep=r"\s+", header=None, skiprows=skip, usecols=[0, 1, 2],
                            names=["u", "v", "rc"], dtype={"u": np.int64, "v": np.int64, "rc": np.float64})
        u, v, rc = frame["u"].to_numpy(), frame["v"].to_numpy(), frame["rc"].to_numpy()
    except Exception:  # noqa: BLE001
        data = np.loadtxt(path, skiprows=skip, ndmin=2)
        u, v, rc = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float64)
    lines = int(len(u))
    if skip:
        assert int(head[0]) == n, f"header says {head[0]} nodes, expected {n}"
    decimals = 0                     # printing precision of the reduced costs (from the first lines)
    with open(path) as f:
        if skip:
            f.readline()
        for _, line in zip(range(5000), f):
            parts = line.split()
            if len(parts) >= 3 and "." in parts[2]:
                decimals = max(decimals, len(parts[2].lower().split("e")[0].split(".")[1]))
    assert lines > 0 and ((u >= 0) & (u < n) & (v >= 0) & (v < n)).all(), "node numbers out of range"
    keep = u != v
    u, v, rc = u[keep], v[keep], rc[keep]
    key = np.minimum(u, v) * n + np.maximum(u, v)
    order = np.lexsort((rc, key))
    key, rc = key[order], rc[order]
    first = np.r_[True, key[1:] != key[:-1]]
    starts = np.flatnonzero(first)
    low = rc[starts]
    high = np.maximum.reduceat(rc, starts)
    keys = key[starts]
    counts = np.bincount(np.r_[keys // n, keys % n], minlength=n)
    return {"keys": keys, "rc": low, "lines": lines, "header": head if skip else None,
            "dup_max_diff": float((high - low).max()) if len(low) else 0.0, "counts": counts,
            "quantum": 10.0 ** (-decimals)}


def node_thresholds(keys, rc, n, k):
    """For a dump that lists the k lowest reduced costs of every node: the k-th lowest value at each node. An edge
    that is not listed has a reduced cost of at least max(thr[u], thr[v]). Nodes with fewer than k listed edges get
    -inf (no bound)."""
    ends = np.r_[keys // n, keys % n]
    values = np.r_[rc, rc]
    order = np.lexsort((values, ends))
    ends, values = ends[order], values[order]
    counts = np.bincount(ends, minlength=n)
    starts = np.searchsorted(ends, np.arange(n))
    thr = np.full(n, -np.inf)
    has = counts >= k
    thr[has] = values[starts[has] + k - 1]
    return thr


def write_full_edge_file(path, D):
    """The complete graph in Concorde's edge-file format: 'n m', then 'u v length' (0-based)."""
    n = len(D)
    iu, ju = np.triu_indices(n, 1)
    with open(path, "w") as f:
        f.write(f"{n} {len(iu)}\n")
        np.savetxt(f, np.column_stack([iu, ju, D[iu, ju]]), fmt="%d")


# ---------------------------------------------------------------------------------------------- edges and tours
def tour_keys(tour, n):
    t = np.asarray(tour, dtype=np.int64)
    a, b = t, np.roll(t, -1)
    return np.sort(np.minimum(a, b) * n + np.maximum(a, b))


def lookup(keys, query):
    """Positions of query in the sorted array keys and a mask of the ones found."""
    if len(keys) == 0:
        return np.zeros(len(query), dtype=np.int64), np.zeros(len(query), dtype=bool)
    pos = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
    return pos, keys[pos] == query


def distance_matrix(xy, ew="EUC_2D"):
    """Integer distances as LKH-3 and Concorde compute them (EUC_2D: nearest integer; CEIL_2D: ceiling)."""
    xy = np.asarray(xy, dtype=np.float64)
    dx = xy[:, None, 0] - xy[None, :, 0]
    dy = xy[:, None, 1] - xy[None, :, 1]
    d = np.sqrt(dx * dx + dy * dy)
    if ew == "EUC_2D":
        return np.floor(d + 0.5).astype(np.int64)
    if ew == "CEIL_2D":
        return np.ceil(d).astype(np.int64)
    raise ValueError(f"unsupported EDGE_WEIGHT_TYPE {ew}")


# ------------------------------------------------------------------------------------------- C1: 1-tree bound
def one_tree(D, pi, precision=PRECISION):
    """Minimum 1-tree of the transformed costs C = precision*D + pi_i + pi_j (LKH-3's form: a minimum spanning tree
    plus, at the leaf whose second-smallest edge is longest, that edge) and the alpha-values of all edges.

    A tour is a 1-tree, so a tour that uses edge e costs at least (L1 + alpha(e) - 2*sum(pi)) / precision, where
    L1 is the cost of the minimum 1-tree and L1 + alpha(e) the cost of the minimum 1-tree that contains e. W =
    L1 - 2*sum(pi) is the bound (all values are integers, in units of 1/precision)."""
    D = np.asarray(D, dtype=np.int64)
    pi = np.asarray(pi, dtype=np.int64)
    n = len(D)
    C = D * precision + pi[:, None] + pi[None, :]
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = C[0].copy()
    parent = np.zeros(n, dtype=np.int64)
    order = np.zeros(n, dtype=np.int64)
    beta = np.full((n, n), -BIG, dtype=np.int64)  # largest tree edge on the path between two nodes (none: -BIG;
    #                                               transformed costs can be negative, so 0 is not a neutral start)
    for t in range(1, n):
        v = int(np.argmin(np.where(in_tree, BIG, best)))
        p = int(parent[v])
        prev = order[:t]
        row = np.maximum(beta[p, prev], C[p, v])
        beta[v, prev] = row
        beta[prev, v] = row
        order[t] = v
        in_tree[v] = True
        update = (~in_tree) & (C[v] < best)
        best[update] = C[v][update]
        parent[update] = v
    parent[0] = -1
    children = order[1:]
    mst = int(C[children, parent[children]].sum())
    degree = np.bincount(parent[children], minlength=n) + 1
    degree[0] -= 1
    leaves = np.flatnonzero(degree == 1)
    neighbour = parent.copy()
    if degree[0] == 1:
        neighbour[0] = int(np.flatnonzero(parent == 0)[0])
    rows = C[leaves].copy()
    rows[np.arange(len(leaves)), leaves] = BIG
    rows[np.arange(len(leaves)), neighbour[leaves]] = BIG
    second = rows.min(1)
    pick = int(np.argmax(second))
    special, m2 = int(leaves[pick]), int(second[pick])
    alpha = C - beta
    alpha[special, :] = np.maximum(C[special, :] - m2, 0)
    alpha[:, special] = alpha[special, :]
    np.fill_diagonal(alpha, BIG)
    L1 = mst + m2
    return {"W": int(L1 - 2 * int(pi.sum())), "L1": int(L1), "alpha": alpha, "special": special, "mst": mst,
            "precision": int(precision)}


def c1_remaining(tree, ub):
    """Edges that C1 cannot eliminate for an upper bound ub (an integer tour length): W + alpha(e) <= precision*ub.
    Returns the sorted undirected keys."""
    alpha = tree["alpha"]
    n = len(alpha)
    limit = tree["precision"] * int(round(ub)) - tree["W"]
    a, b = np.nonzero(np.triu(alpha <= limit, 1))
    return np.sort(a * n + b)


# -------------------------------------------------------------------------------------- C2: Concorde's root LP
def c2_remaining(keys, rc, lb, ub, n, tol, thr=None):
    """Edges that C2 cannot eliminate for an upper bound ub: listed edges with rc <= ub - lb + tol, and the nodes at
    which unlisted edges might remain (their k-th lowest reduced cost is not above the gap); None for a complete
    dump."""
    gap = ub - lb + tol
    remaining = keys[rc <= gap]
    unresolved = None if thr is None else ~(thr > gap)
    return remaining, unresolved


def c2_checks(keys, rc, lb, n, opt, opt_tour, tol, thr=None, cap=None, sum_negative=0.0, quantum=0.0):
    """Checks of a C2 certificate against a proven optimal tour.
    1. lb <= opt.
    2. No edge of the optimal tour is eliminated (rc > opt - lb + tol).
    3. Dual consistency: opt >= lb + sum of rc over the tour edges + sum of max(0, -rc) over all edges. This holds
       for the reduced costs of a dual solution whose value is at least lb; reduced costs without the cut duals, or
       a bound that belongs to another dual solution, violate it in general.
    keys/rc may hold only the edges with rc <= cap; an edge that is absent then has rc > cap (complete dump) or at
    least max(thr) (partial dump). quantum: printing precision of the reduced costs (each may be off by half of it).
    """
    t_keys = tour_keys(opt_tour, n)
    pos, found = lookup(keys, t_keys)
    a, b = t_keys // n, t_keys % n
    rc_tour = np.where(found, rc[pos], np.nan)
    if thr is None:                       # complete dump: an absent edge has rc > cap (impossible without a cap)
        floor_absent = np.full(len(t_keys), np.inf if cap is None else float(cap))
    else:                                 # partial dump: unlisted (rc >= max thr) or listed above the stored cap
        floor_absent = np.maximum(thr[a], thr[b])
        if cap is not None:
            floor_absent = np.minimum(floor_absent, float(cap))
    gap = opt - lb + tol
    eliminated = (found & (rc_tour > gap)) | (~found & (floor_absent > gap))
    rc_low = np.where(found, rc_tour, np.where(np.isfinite(floor_absent), floor_absent, 0.0))
    slack = float(opt - lb - rc_low.sum() - sum_negative)
    tol_sum = tol * (1.0 + 0.01 * n) + quantum * 2 * n
    out = {"lb_le_opt": bool(lb <= opt + tol), "opt_edges_eliminated": int(eliminated.sum()),
           "opt_edges_unlisted": int((~found).sum()), "dual_slack": slack, "dual_ok": bool(slack >= -tol_sum),
           "root_gap_pct": float(100.0 * (opt - lb) / opt)}
    out["passed"] = bool(out["lb_le_opt"] and out["opt_edges_eliminated"] == 0 and out["dual_ok"])
    return out


# -------------------------------------------------------------------------------------- rankings and measures
def ranking_from_key(key, depth):
    """Per row, the columns with the smallest keys (best first); the diagonal must already be excluded by a large
    key."""
    n = key.shape[0]
    depth = min(depth, n - 1)
    part = np.argpartition(key, depth - 1, axis=1)[:, :depth]
    order = np.argsort(np.take_along_axis(key, part, 1), axis=1, kind="stable")
    return np.take_along_axis(part, order, 1)


def alpha_ranking(alpha, D, depth):
    """All neighbours by alpha-value, ties by distance, then by node number."""
    n = len(D)
    off = ~np.eye(n, dtype=bool)
    if int(alpha[off].max()) < (1 << 29) and int(D.max()) < (1 << 21) and n <= (1 << 12):
        with np.errstate(over="ignore"):
            key = (alpha << 33) + (D << 12) + np.arange(n, dtype=np.int64)[None, :]
        np.fill_diagonal(key, np.iinfo(np.int64).max)
        return ranking_from_key(key, depth)
    depth = min(depth, n - 1)
    out = np.empty((n, depth), dtype=np.int64)
    cols = np.arange(n)
    for i in range(n):
        o = np.lexsort((cols, D[i], alpha[i]))
        out[i] = o[o != i][:depth]
    return out


def nearest_ranking(D, depth):
    n = len(D)
    key = D * n + np.arange(n)[None, :]
    np.fill_diagonal(key, np.iinfo(np.int64).max)
    return ranking_from_key(key, depth)


def certified_k(ranking, remaining, n, unresolved=None):
    """Per node: the shortest prefix of its ranking that contains all of its remaining edges (in the node's own
    list). ranking.shape[1] + 1 when some remaining edge is not in the ranking (or the node is unresolved)."""
    ranking = np.asarray(ranking, dtype=np.int64)
    L = ranking.shape[1]
    rows = np.repeat(np.arange(n, dtype=np.int64), L)
    cols = ranking.ravel()
    valid = cols >= 0
    list_keys = rows[valid] * n + cols[valid]
    list_pos = np.tile(np.arange(1, L + 1, dtype=np.int64), n)[valid]
    order = np.argsort(list_keys)
    list_keys, list_pos = list_keys[order], list_pos[order]
    a, b = remaining // n, remaining % n
    src, dst = np.r_[a, b], np.r_[b, a]
    pos, found = lookup(list_keys, src * n + dst)
    need = np.where(found, list_pos[pos], L + 1)
    k = np.zeros(n, dtype=np.int64)
    np.maximum.at(k, src, need)
    if unresolved is not None:
        k[unresolved] = L + 1
    return k


def edge_coverage(lists, remaining, n):
    """Share of the remaining edges that appear in the list of at least one end."""
    lists = np.asarray(lists, dtype=np.int64)
    L = lists.shape[1]
    rows = np.repeat(np.arange(n, dtype=np.int64), L)
    cols = lists.ravel()
    valid = cols >= 0
    keys = np.unique(np.minimum(rows[valid], cols[valid]) * n + np.maximum(rows[valid], cols[valid]))
    _, found = lookup(keys, remaining)
    return float(found.mean()) if len(remaining) else 1.0


def degree_summary(remaining, n):
    degree = np.bincount(np.r_[remaining // n, remaining % n], minlength=n)
    return {"edges_per_node": float(len(remaining) / n), "mean_degree": float(degree.mean()),
            "median_degree": float(np.median(degree)), "max_degree": int(degree.max()),
            "share_degree_le5": float((degree <= 5).mean()), "share_degree_le8": float((degree <= 8).mean())}


def k_summary(k, L):
    inf = L + 1
    out = {"le5": float((k <= 5).mean()), "not_in_list": float((k >= inf).mean())}
    for m in (8, 10, 16):
        if L >= m:
            out[f"le{m}"] = float((k <= m).mean())
    out["median"] = float(np.median(k)) if (k < inf).mean() > 0.5 else None
    out["p90"] = float(np.quantile(k, 0.9)) if (k < inf).mean() > 0.9 else None
    return out


def auc(score, label):
    """Probability that a random positive scores above a random negative (ties count one half)."""
    from scipy.stats import rankdata
    label = np.asarray(label, dtype=bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def graph_edges(NB, values, n, reduce="max"):
    """Distinct undirected edges of a (n, K) neighbour table with one value per edge (max over both directions)."""
    NB = np.asarray(NB, dtype=np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), NB.shape[1])
    cols = NB.ravel()
    vals = np.asarray(values, dtype=np.float64).ravel()
    valid = (cols >= 0) & (cols != rows)
    keys = np.minimum(rows[valid], cols[valid]) * n + np.maximum(rows[valid], cols[valid])
    uniq, inverse = np.unique(keys, return_inverse=True)
    out = np.full(len(uniq), -np.inf)
    np.maximum.at(out, inverse, vals[valid])
    return uniq, out
