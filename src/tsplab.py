"""
tsplab - utilities for "learned guidance + LKH-3" TSP experiments.

  * instance generation (uniform / clustered / mixed), integer scaling
  * TSPLIB I/O: problem, tour, parameter, LKH candidate and penalty (pi) files
  * LKH-3 download/build, runner, output parser (anytime trace)
  * Held-Karp 1-tree bound with LKH's 1-tree definition (unit checks)
  * candidate-set coverage metrics
  * dataset shards for variable-size instances
  * process-parallel map with error capture
"""
from __future__ import annotations

import glob
import gzip
import io
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import traceback
import urllib.request

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

COORD_SCALE = 1_000_000          # synthetic instances: unit square -> integer grid
LKH_PRECISION = 100              # LKH default PRECISION
CAND_K = 5                       # LKH default MAX_CANDIDATES

# Known optimal tour lengths (TSPLIB, symmetric, 1k-20k nodes).
TSPLIB_OPT = {
    "pr1002": 259045, "si1032": 92650, "u1060": 224094, "vm1084": 239297,
    "pcb1173": 56892, "d1291": 50801, "rl1304": 252948, "rl1323": 270199,
    "nrw1379": 56638, "fl1400": 20127, "u1432": 152970, "fl1577": 22249,
    "d1655": 62128, "vm1748": 336556, "u1817": 57201, "rl1889": 316536,
    "d2103": 80450, "u2152": 64253, "u2319": 234256, "pr2392": 378032,
    "pcb3038": 137694, "fl3795": 28772, "fnl4461": 182566, "rl5915": 565530,
    "rl5934": 556045, "pla7397": 23260728, "rl11849": 923288,
    "usa13509": 19982859, "brd14051": 469385, "d15112": 1573084,
    "d18512": 645238, "dsj1000": 18660188,
}


# ----------------------------------------------------------------------------- env
def env_info():
    info = {"nproc": os.cpu_count()}
    try:
        with open("/proc/meminfo") as f:
            kb = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1))
        info["ram_gb"] = round(kb / 1024 ** 2, 1)
    except Exception:
        info["ram_gb"] = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        info["gpu"] = out or None
    except Exception:
        info["gpu"] = None
    info["numpy"] = np.__version__
    return info


# ---------------------------------------------------------------------- generation
def normalize_unit(xy):
    xy = np.asarray(xy, dtype=np.float64)
    mn = xy.min(0)
    span = float((xy.max(0) - mn).max())
    return (xy - mn) / (span if span > 0 else 1.0)


def gen_uniform(n, rng):
    return rng.random((n, 2))


def gen_clustered(n, rng, n_clusters=None, sigma=None):
    c = int(n_clusters if n_clusters is not None else rng.integers(3, 16))
    s = float(sigma if sigma is not None else math.exp(rng.uniform(math.log(0.02), math.log(0.08))))
    centers = rng.random((c, 2))
    lab = rng.integers(0, c, n)
    return normalize_unit(centers[lab] + rng.normal(0.0, s, (n, 2)))


def gen_instance(n, dist, rng):
    if dist == "uniform":
        return gen_uniform(n, rng)
    if dist == "clustered":
        return gen_clustered(n, rng)
    if dist == "mixed":
        return gen_uniform(n, rng) if rng.random() < 0.5 else gen_clustered(n, rng)
    raise ValueError(dist)


def to_int_coords(xy01, scale=COORD_SCALE):
    return np.rint(np.asarray(xy01, dtype=np.float64) * scale).astype(np.int64)


# ------------------------------------------------------------------------ distances
def edge_len(a, b, ew="EUC_2D"):
    d = np.sqrt(((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2).sum(-1))
    if ew == "EUC_2D":
        return np.floor(d + 0.5)
    if ew == "CEIL_2D":
        return np.ceil(d)
    if ew in ("FLOAT", "EXACT"):
        return d
    raise ValueError(f"unsupported EDGE_WEIGHT_TYPE {ew}")


def tour_length(xy, tour, ew="EUC_2D"):
    t = np.asarray(tour)
    xy = np.asarray(xy)
    return float(edge_len(xy[t], xy[np.roll(t, -1)], ew).sum())


def is_valid_tour(tour, n):
    if tour is None:
        return False
    t = np.asarray(tour)
    return t.shape == (n,) and np.array_equal(np.sort(t), np.arange(n))


def gap_pct(cost, ref):
    return 100.0 * (cost - ref) / ref


# --------------------------------------------------------------------- TSPLIB I/O
def _open_text(path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path), encoding="latin-1")
    return open(path, encoding="latin-1")


def _is_int_token(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def write_tsp(path, xy, name="instance", ew="EUC_2D", comment=None):
    xy = np.asarray(xy)
    head = [f"NAME : {name}", "TYPE : TSP"]
    if comment:
        head.append(f"COMMENT : {comment}")
    head += [f"DIMENSION : {len(xy)}", f"EDGE_WEIGHT_TYPE : {ew}", "NODE_COORD_SECTION"]
    if np.issubdtype(xy.dtype, np.integer):
        body = "\n".join(f"{i + 1} {x} {y}" for i, (x, y) in enumerate(xy.tolist()))
    else:
        body = "\n".join(f"{i + 1} {x!r} {y!r}" for i, (x, y) in enumerate(xy.tolist()))
    with open(path, "w") as f:
        f.write("\n".join(head) + "\n" + body + "\nEOF\n")


def read_tsplib(path):
    hdr, ids, coords = {}, [], []
    section = None
    with _open_text(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.upper() == "EOF":
                break
            up = line.upper()
            if section == "NODE_COORD_SECTION":
                parts = line.split()
                if len(parts) >= 3 and _is_int_token(parts[0]):
                    ids.append(int(parts[0]))
                    coords.append((float(parts[1]), float(parts[2])))
                    continue
                section = None
            if "_SECTION" in up.split(":")[0]:
                section = up.split(":")[0].strip()
                continue
            if section is not None:
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                hdr[k.strip().upper()] = v.strip()
    xy = np.array(coords, dtype=np.float64)
    if ids:
        xy = xy[np.argsort(np.array(ids))]
    return {
        "name": hdr.get("NAME", os.path.basename(str(path)).split(".")[0]),
        "type": hdr.get("TYPE"),
        "dimension": int(hdr.get("DIMENSION", len(xy))),
        "edge_weight_type": hdr.get("EDGE_WEIGHT_TYPE"),
        "comment": hdr.get("COMMENT"),
        "xy": xy,
    }


def write_par(path, params):
    with open(path, "w") as f:
        for k, v in params.items():
            if v is None:
                continue
            for x in (v if isinstance(v, (list, tuple)) else [v]):
                if isinstance(x, bool):
                    x = "YES" if x else "NO"
                f.write(f"{k} = {x}\n")


def read_tour(path):
    vals, in_sec = [], False
    with _open_text(path) as f:
        for line in f:
            s = line.strip()
            if not in_sec:
                if s.upper().startswith("TOUR_SECTION"):
                    in_sec = True
                continue
            for tok in s.split():
                if tok.upper() == "EOF":
                    break
                v = int(tok)
                if v == -1:
                    return np.array(vals, dtype=np.int64) - 1
                vals.append(v)
    return np.array(vals, dtype=np.int64) - 1


def write_tour(path, tour, name="tour", length=None):
    t = np.asarray(tour, dtype=np.int64) + 1
    head = [f"NAME : {name}"]
    if length is not None:
        head.append(f"COMMENT : Length = {int(length)}")
    head += ["TYPE : TOUR", f"DIMENSION : {len(t)}", "TOUR_SECTION"]
    with open(path, "w") as f:
        f.write("\n".join(head) + "\n" + "\n".join(map(str, t.tolist())) + "\n-1\nEOF\n")


def write_candidates(path, cand, alpha=None, dad=None):
    """LKH CANDIDATE_FILE. cand: (n,K) 0-based node ids, -1 padded, best first.
    alpha: same shape integer ranks/scores (default rank*100). dad: (n,) 0-based or -1."""
    cand = np.asarray(cand, dtype=np.int64)
    n = len(cand)
    lines = [str(n)]
    for i in range(n):
        row = cand[i]
        valid = row >= 0
        row = row[valid]
        if alpha is None:
            al = np.arange(len(row), dtype=np.int64) * 100
        else:
            al = np.asarray(alpha[i], dtype=np.int64)[valid]
        d = 0 if dad is None or dad[i] < 0 else int(dad[i]) + 1
        pairs = np.empty(2 * len(row), dtype=np.int64)
        pairs[0::2] = row + 1
        pairs[1::2] = al
        lines.append(f"{i + 1} {d} {len(row)} " + " ".join(map(str, pairs.tolist())))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n-1\nEOF\n")


def read_candidates(path, k_pad=None):
    with open(path) as f:
        toks = f.read().split()
    n = int(toks[0])
    pos = 1
    lists, alists = [None] * n, [None] * n
    dad = np.full(n, -1, dtype=np.int64)
    while pos < len(toks) and _is_int_token(toks[pos]):
        tid = int(toks[pos])
        pos += 1
        if tid == -1:
            break
        d, cnt = int(toks[pos]), int(toks[pos + 1])
        pos += 2
        v = np.array(toks[pos:pos + 2 * cnt], dtype=np.int64).reshape(cnt, 2)
        pos += 2 * cnt
        order = np.argsort(v[:, 1], kind="stable")
        lists[tid - 1], alists[tid - 1] = v[order, 0] - 1, v[order, 1]
        dad[tid - 1] = d - 1
    K = max([len(x) for x in lists if x is not None] + [0])
    K = max(K, k_pad or 0)
    C = np.full((n, K), -1, dtype=np.int64)
    A = np.full((n, K), -1, dtype=np.int64)
    for i, x in enumerate(lists):
        if x is not None and len(x):
            C[i, :len(x)] = x
            A[i, :len(x)] = alists[i]
    return {"n": n, "cand": C, "alpha": A, "dad": dad}


def write_pi(path, pi):
    pi = np.rint(np.asarray(pi, dtype=np.float64)).astype(np.int64)
    with open(path, "w") as f:
        f.write(f"{len(pi)}\n" + "\n".join(f"{i + 1} {v}" for i, v in enumerate(pi.tolist())) + "\n-1\nEOF\n")


def read_pi(path):
    with open(path) as f:
        toks = f.read().split()
    n = int(toks[0])
    pi = np.zeros(n, dtype=np.int64)
    pos = 1
    for _ in range(n):
        i, v = int(toks[pos]), int(toks[pos + 1])
        pos += 2
        pi[i - 1] = v
    return pi


# -------------------------------------------------------------------------- LKH
LKH_BASES = ["http://webhotel4.ruc.dk/~keld/research/LKH-3/",
             "https://webhotel4.ruc.dk/~keld/research/LKH-3/"]
LKH_FALLBACK_VERSIONS = ["3.0.13", "3.0.12", "3.0.11", "3.0.10", "3.0.9", "3.0.8"]


def _urlget(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (tsplab)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _extract(tgz, dest):
    with tarfile.open(tgz) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:
            tf.extractall(dest)


def build_lkh(dest="/content/lkh", version=None, verbose=True):
    """Download the latest LKH-3 from the author's site and compile it. Returns the binary path."""
    os.makedirs(dest, exist_ok=True)
    found = sorted(glob.glob(os.path.join(dest, "LKH-3*", "LKH")))
    if found and version is None:
        return found[-1]
    versions = [version] if version else []
    if not versions:
        for base in LKH_BASES:
            try:
                html = _urlget(base, 30).decode("latin-1")
                vs = sorted({tuple(map(int, m)) for m in re.findall(r"LKH-3\.(\d+)\.(\d+)\.tgz", html)},
                            reverse=True)
                versions = [f"3.{a}.{b}" for a, b in vs]
                if versions:
                    break
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print("index fetch failed:", base, e)
        versions += [v for v in LKH_FALLBACK_VERSIONS if v not in versions]
    last = None
    for v in versions:
        for base in LKH_BASES:
            try:
                data = _urlget(f"{base}LKH-{v}.tgz", 180)
            except Exception as e:  # noqa: BLE001
                last = e
                continue
            tgz = os.path.join(dest, f"LKH-{v}.tgz")
            with open(tgz, "wb") as f:
                f.write(data)
            _extract(tgz, dest)
            src = os.path.join(dest, f"LKH-{v}")
            r = subprocess.run(["make"], cwd=src, capture_output=True, text=True)
            binp = os.path.join(src, "LKH")
            if r.returncode != 0 or not os.path.exists(binp):
                raise RuntimeError(f"LKH build failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
            if verbose:
                print(f"LKH-{v} built at {binp}")
            return binp
    raise RuntimeError(f"Could not download LKH-3 ({last})")


_RX = {
    "lb": re.compile(r"Lower bound\s*=\s*(-?[\d.]+)"),
    "ascent": re.compile(r"Ascent time\s*=\s*([\d.]+)\s*sec"),
    "pre": re.compile(r"Preprocessing time\s*=\s*([\d.]+)\s*sec"),
    "imp": re.compile(r"^\*\s*(\d+):\s*Cost\s*=\s*(-?\d+)[^\n]*?Time\s*=\s*([\d.]+)\s*sec", re.M),
    "run": re.compile(r"^Run\s+(\d+):\s*Cost\s*=\s*(-?\d+)[^\n]*?Time\s*=\s*([\d.]+)\s*sec", re.M),
    "costmin": re.compile(r"Cost\.min\s*=\s*(-?\d+)"),
    "timetotal": re.compile(r"Time\.total\s*=\s*([\d.]+)\s*sec"),
    "cand": re.compile(r"Cand\.min\s*=\s*(\d+),\s*Cand\.avg\s*=\s*([\d.]+),\s*Cand\.max\s*=\s*(\d+)"),
    "trials": re.compile(r"Trials\.min\s*=\s*(\d+)"),
}


def parse_lkh_output(text):
    def f1(key, cast=float):
        m = _RX[key].search(text)
        return cast(m.group(1)) if m else None

    out = {
        "lower_bound": f1("lb"),
        "ascent_time": f1("ascent"),
        "pre_time": f1("pre"),
        "cost_min": f1("costmin", int),
        "time_total": f1("timetotal"),
        "trials": f1("trials", int),
        "events": [(int(a), int(b), float(c)) for a, b, c in _RX["imp"].findall(text)],
        "runs": [(int(a), int(b), float(c)) for a, b, c in _RX["run"].findall(text)],
    }
    m = _RX["cand"].search(text)
    out["cand_stats"] = (int(m.group(1)), float(m.group(2)), int(m.group(3))) if m else None
    return out


def run_lkh(lkh_bin, problem_file, params, workdir, timeout=None, keep_stdout=False):
    os.makedirs(workdir, exist_ok=True)
    par_path = os.path.join(workdir, "run.par")
    p = {"PROBLEM_FILE": problem_file, "RUNS": 1, "TRACE_LEVEL": 1}
    p.update(params or {})
    p.setdefault("OUTPUT_TOUR_FILE", os.path.join(workdir, "run.tour"))
    write_par(par_path, p)
    t0, t_start = time.perf_counter(), time.time()      # t_start: wall clock, comparable across processes
    try:
        proc = subprocess.run([lkh_bin, par_path], cwd=workdir, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, errors="replace", timeout=timeout)
        rc, so, se, to = proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as e:
        so = e.stdout.decode("latin-1", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        se = e.stderr.decode("latin-1", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc, to = None, True
    wall = time.perf_counter() - t0
    res = parse_lkh_output(so)
    res.update(wall=wall, returncode=rc, timed_out=to, stderr=(se or "")[-2000:], t_start=t_start)
    res["stdout"] = so if keep_stdout else so[-4000:]
    tp = p["OUTPUT_TOUR_FILE"]
    res["tour"] = read_tour(tp) if os.path.exists(tp) else None
    return res


def lkh_solve(lkh_bin, xy=None, problem_file=None, ew="EUC_2D", params=None, workroot=None,
              want_cands=False, want_pi=False, cand_file=None, pi_file=None, timeout=None,
              keep_stdout=False):
    """Run LKH on coordinates (written to a temp TSPLIB file) or on an existing problem file.
    cand_file / pi_file: existing files to be READ by LKH (learned guidance).
    want_cands / want_pi: let LKH WRITE its own candidate set / penalties and return them."""
    os.makedirs(workroot or tempfile.gettempdir(), exist_ok=True)
    wd = tempfile.mkdtemp(dir=workroot, prefix="lkh_")
    try:
        if problem_file is None:
            problem_file = os.path.join(wd, "p.tsp")
            write_tsp(problem_file, xy, ew=ew)
        p = dict(params or {})
        out_cand = out_pi = None
        if cand_file is not None:
            p["CANDIDATE_FILE"] = cand_file
        elif want_cands:
            out_cand = p["CANDIDATE_FILE"] = os.path.join(wd, "out.cand")
        if pi_file is not None:
            p["PI_FILE"] = pi_file
        elif want_pi:
            out_pi = p["PI_FILE"] = os.path.join(wd, "out.pi")
        res = run_lkh(lkh_bin, problem_file, p, wd, timeout=timeout, keep_stdout=keep_stdout)
        if out_cand and os.path.exists(out_cand):
            res["cand"] = read_candidates(out_cand, k_pad=p.get("MAX_CANDIDATES", CAND_K))
        if out_pi and os.path.exists(out_pi):
            res["pi"] = read_pi(out_pi)
        return res
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def anytime_curve(res, ref, budgets):
    """Gap (%) of the best tour found within each total-time budget (preprocessing + search)."""
    pre = res.get("pre_time") or 0.0
    ev = sorted(res.get("events") or [], key=lambda e: e[2])
    out = []
    for b in budgets:
        best = None
        for _, c, t in ev:
            if pre + t <= b:
                best = c if best is None else min(best, c)
        out.append(gap_pct(best, ref) if best is not None else np.nan)
    return out


TSPLIB_BASES = ["https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/",
                "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"]


def _valid_tsp(path):
    try:
        if os.path.getsize(path) == 0:
            return False
        inst = read_tsplib(path)
        return inst["dimension"] > 0 and (len(inst["xy"]) == inst["dimension"] or inst["edge_weight_type"] == "EXPLICIT")
    except Exception:  # noqa: BLE001
        return False


def _local_tsplib(dest, names):
    """{name: dest/name.tsp} for every name found under dest as a valid .tsp or .tsp.gz (any subfolder).
    Empty files (left behind by interrupted downloads) are removed so that they never count as present."""
    have = {}
    for nm in names:
        out = os.path.join(dest, nm + ".tsp")
        if os.path.exists(out):
            if _valid_tsp(out):
                have[nm] = out
                continue
            os.remove(out)
        for cand in sorted(glob.glob(os.path.join(dest, "**", nm + ".tsp*"), recursive=True)):
            if cand == out or not (cand.endswith(".tsp") or cand.endswith(".tsp.gz")):
                continue
            try:
                if os.path.getsize(cand) == 0:
                    os.remove(cand)
                    continue
                if cand.endswith(".gz"):
                    with gzip.open(cand, "rb") as fi, open(out + ".part", "wb") as fo:
                        shutil.copyfileobj(fi, fo)
                    os.replace(out + ".part", out)
                else:
                    shutil.copy(cand, out)
                if _valid_tsp(out):
                    have[nm] = out
                    break
                os.remove(out)
            except Exception:  # noqa: BLE001
                continue
    return have


def download_tsplib(dest, names=None, verbose=True, timeout=30, bases=None):
    """TSPLIB problems as {name: path.tsp}.
    1) files already under dest (.tsp, .tsp.gz or ALL_tsp.tar.gz, e.g. copied there by the user);
    2) the official TSPLIB95 server. Each address is first probed with one small file and a short timeout,
       so an unreachable server costs about a minute instead of one long timeout per file."""
    os.makedirs(dest, exist_ok=True)
    names = list(names or TSPLIB_OPT)
    tgz = os.path.join(dest, "ALL_tsp.tar.gz")
    if os.path.exists(tgz):
        if os.path.getsize(tgz) == 0:
            os.remove(tgz)                      # left behind by an interrupted download
        else:
            try:
                _extract(tgz, dest)
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print("could not extract", tgz, e)
    have = _local_tsplib(dest, names)
    if len(have) == len(names):
        return have
    for base in (bases or TSPLIB_BASES):
        probe = next(nm for nm in names if nm not in have)
        try:
            data = _urlget(base + probe + ".tsp.gz", timeout)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print("TSPLIB server not answering at", base, "-", e)
            continue
        with open(os.path.join(dest, probe + ".tsp.gz"), "wb") as f:
            f.write(data)
        try:
            data = _urlget(base + "ALL_tsp.tar.gz", max(timeout, 120))
            with open(tgz + ".part", "wb") as f:
                f.write(data)
            os.replace(tgz + ".part", tgz)
            _extract(tgz, dest)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print("ALL_tsp.tar.gz not available, fetching single files:", e)
        have = _local_tsplib(dest, names)
        for nm in [m for m in names if m not in have]:
            try:
                data = _urlget(base + nm + ".tsp.gz", timeout)
                with open(os.path.join(dest, nm + ".tsp.gz"), "wb") as f:
                    f.write(data)
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print("could not get", nm, e)
        break
    return _local_tsplib(dest, names)


def time_to_gap(res, ref, target_pct):
    """Earliest total time (preprocessing + search) at which the gap reaches target_pct; inf if never."""
    pre = res.get("pre_time") or 0.0
    for _, c, t in sorted(res.get("events") or [], key=lambda e: e[2]):
        if gap_pct(c, ref) <= target_pct + 1e-12:
            return pre + t
    return float("inf")


# ------------------------------------------------------------- Held-Karp 1-tree
def _row_cost(xy, u, ew, precision, pi):
    r = edge_len(xy[u], xy, ew) * precision
    if pi is not None:
        r = r + pi[u] + pi
    return r


def held_karp_bound(xy, pi=None, ew="EUC_2D", precision=1.0, chunk=256):
    """w(pi) with LKH's minimum 1-tree: MST + the longest second-nearest edge among MST leaves.
    Transformed cost C'_ij = precision*d_ij + pi_i + pi_j.  Returns bound in distance units."""
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    pia = None if pi is None else np.asarray(pi, dtype=np.float64)
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int64)
    best[0] = 0.0
    total = 0.0
    for _ in range(n):
        u = int(np.argmin(best))
        total += best[u]
        in_tree[u] = True
        best[u] = np.inf
        row = _row_cost(xy, u, ew, precision, pia)
        upd = (~in_tree) & (row < best)
        best[upd] = row[upd]
        parent[upd] = u
    deg = np.bincount(parent[parent >= 0], minlength=n) + (parent >= 0)
    nbr = parent.copy()
    root_child = np.where(parent == 0)[0]
    if deg[0] == 1:
        nbr[0] = root_child[0]
    leaves = np.where(deg == 1)[0]
    best_val, best_leaf, best_to = -np.inf, -1, -1
    for s in range(0, len(leaves), chunk):
        L = leaves[s:s + chunk]
        D = edge_len(xy[L][:, None, :], xy[None, :, :], ew) * precision
        if pia is not None:
            D = D + pia[L][:, None] + pia[None, :]
        D[np.arange(len(L)), L] = np.inf
        D[np.arange(len(L)), nbr[L]] = np.inf
        j = D.argmin(1)
        v = D[np.arange(len(L)), j]
        k = int(v.argmax())
        if v[k] > best_val:
            best_val, best_leaf, best_to = float(v[k]), int(L[k]), int(j[k])
    deg = deg.copy()
    deg[best_leaf] += 1
    deg[best_to] += 1
    w = total + best_val - (2.0 * pia.sum() if pia is not None else 0.0)
    return {"bound": w / precision, "degree": deg, "parent": parent, "special": (best_leaf, best_to)}


# ----------------------------------------------------------------- candidates
def knn(xy, k):
    from scipy.spatial import cKDTree
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    k = min(k, n - 1)
    d, idx = cKDTree(xy).query(xy, k=k + 1)
    self_mask = idx == np.arange(n)[:, None]
    no_self = ~self_mask.any(1)
    self_mask[no_self, -1] = True
    keep = ~self_mask
    return idx[keep].reshape(n, k), d[keep].reshape(n, k)


def tour_succ_pred(tour):
    t = np.asarray(tour)
    succ = np.empty(len(t), dtype=np.int64)
    pred = np.empty(len(t), dtype=np.int64)
    succ[t] = np.roll(t, -1)
    pred[t] = np.roll(t, 1)
    return succ, pred


def tour_edge_labels(nbr_idx, tour):
    """(n,k) bool: is edge (i, nbr_idx[i,j]) in the tour (either direction)."""
    succ, pred = tour_succ_pred(tour)
    return (nbr_idx == succ[:, None]) | (nbr_idx == pred[:, None])


def candidate_coverage(cand, tour, ks=None):
    cand = np.asarray(cand)
    n, K = cand.shape
    succ, pred = tour_succ_pred(tour)
    out = {}
    for k in (ks or range(1, K + 1)):
        c = cand[:, :k]
        in_s = (c == succ[:, None]).any(1)
        in_p = (c == pred[:, None]).any(1)
        und = in_s | in_p[succ]
        out[int(k)] = {"directed": float((in_s.sum() + in_p.sum()) / (2 * n)),
                       "undirected": float(und.mean()),
                       "missing": int(n - und.sum())}
    return out


def pad_cols(a, K, fill=-1):
    a = np.asarray(a)
    if a.shape[1] >= K:
        return a[:, :K]
    return np.concatenate([a, np.full((a.shape[0], K - a.shape[1]), fill, dtype=a.dtype)], 1)


# --------------------------------------------------------------------- shards
def save_shard(path, records):
    ns = np.array([int(r["n"]) for r in records], dtype=np.int64)
    data = {"offsets": np.concatenate([[0], np.cumsum(ns)])}
    for k, v0 in records[0].items():
        if k == "n":
            continue
        if isinstance(v0, np.ndarray) and v0.ndim >= 1 and len(v0) == records[0]["n"]:
            data["node__" + k] = np.concatenate([np.asarray(r[k]) for r in records], 0)
        else:
            vals = [np.nan if r.get(k) is None else r.get(k) for r in records]
            data["scal__" + k] = np.array(vals)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz" if path.endswith(".npz") else path + ".tmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)


class Shard:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.off = z["offsets"]
        self.node = {k[6:]: z[k] for k in z.files if k.startswith("node__")}
        self.scal = {k[6:]: z[k] for k in z.files if k.startswith("scal__")}

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, i):
        a, b = int(self.off[i]), int(self.off[i + 1])
        r = {k: v[a:b] for k, v in self.node.items()}
        r.update({k: (v[i].item() if getattr(v[i], "ndim", 1) == 0 else v[i]) for k, v in self.scal.items()})
        r["n"] = b - a
        return r


def load_shards(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        s = Shard(p)
        out.extend(s[i] for i in range(len(s)))
    return out


# ---------------------------------------------------------------- parallel map
def _safe_call(args):
    fn, kw = args
    try:
        return fn(**kw)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e), "traceback": traceback.format_exc()}


def pmap(fn, kwargs_list, workers=None, desc=None, ctx="fork"):
    """Process-parallel map; exceptions are returned as {'error': ...} instead of raised."""
    import concurrent.futures as cf
    import multiprocessing as mp
    workers = max(1, workers or os.cpu_count() or 1)
    if workers == 1 or len(kwargs_list) <= 1:
        return [_safe_call((fn, kw)) for kw in tqdm(kwargs_list, desc=desc)]
    out = [None] * len(kwargs_list)
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context(ctx)) as ex:
        futs = {ex.submit(_safe_call, (fn, kw)): i for i, kw in enumerate(kwargs_list)}
        for f in tqdm(cf.as_completed(futs), total=len(futs), desc=desc):
            out[futs[f]] = f.result()
    return out


def errors_of(results):
    return [r for r in results if isinstance(r, dict) and "error" in r]


def call_task(fn, kw):
    """Run fn(**kw). Lets a single pool run different task functions; picklable under 'spawn'
    (functions defined inside a notebook cell are not)."""
    return fn(**kw)


# ------------------------------------------------------------ task functions
def task_record(lkh_bin, workroot, seed, xy=None, problem_file=None, ew="EUC_2D", params=None,
                dist=None, name=None, require_files=True, ensemble_M=0):
    """Solve one instance with LKH and keep: tour, alpha-candidates, penalties (pi), timings.
    ensemble_M > 0 also stores a TourFold ensemble: M single-trial tours on shared POPMUSIC candidates."""
    p = {"SEED": int(seed), "MAX_CANDIDATES": CAND_K}
    p.update(params or {})
    res = lkh_solve(lkh_bin, xy=None if problem_file else xy, problem_file=problem_file, ew=ew,
                    params=p, workroot=workroot, want_cands=True, want_pi=True)
    n = len(xy)
    if not is_valid_tour(res["tour"], n):
        raise RuntimeError(f"invalid tour (rc={res['returncode']}): {res['stdout'][-600:]}")
    cost = tour_length(xy, res["tour"], ew)
    if res.get("cost_min") is not None and abs(cost - res["cost_min"]) > 0.5:
        raise RuntimeError(f"cost mismatch {cost} vs {res['cost_min']}")
    xy_store = np.asarray(xy, dtype=np.int32) if np.issubdtype(np.asarray(xy).dtype, np.integer) \
        else np.asarray(xy, dtype=np.float64)
    rec = {"n": n, "xy": xy_store, "tour": res["tour"].astype(np.int32), "cost": cost,
           "lower_bound": res["lower_bound"], "ascent_time": res["ascent_time"],
           "pre_time": res["pre_time"], "wall": res["wall"], "seed": int(seed),
           "dist": dist or "", "name": name or "", "ew": ew}
    if require_files and ("cand" not in res or "pi" not in res):
        raise RuntimeError("LKH did not write CANDIDATE_FILE/PI_FILE: " + res["stdout"][-600:])
    if "cand" in res:
        rec["cand"] = pad_cols(res["cand"]["cand"], CAND_K).astype(np.int32)
        rec["cand_alpha"] = pad_cols(res["cand"]["alpha"], CAND_K).astype(np.int64)
    if "pi" in res:
        rec["pi"] = res["pi"].astype(np.int64)
    if ensemble_M:
        ens = task_ensemble(lkh_bin, workroot, ensemble_M, seed0=int(seed) * 1000 + 17, xy=xy,
                            problem_file=problem_file, ew=ew)
        rec["ens_succ"] = ens["succ"]
        rec["ens_cost"] = ens["costs"].astype(np.float64)
        rec["ens_wall"] = float(ens["walls"].sum())
        rec["pop_cand"] = ens["cand"]
    return rec


def task_train_record(lkh_bin, xy_int, seed, workroot, params=None, dist=None):
    return task_record(lkh_bin, workroot, seed, xy=xy_int, params=params, dist=dist)


def task_candidates(lkh_bin, workroot, xy=None, problem_file=None, ew="EUC_2D", ctype="ALPHA",
                    subgradient=False, k=CAND_K, seed=1):
    p = {"CANDIDATE_SET_TYPE": ctype, "SUBGRADIENT": "YES" if subgradient else "NO",
         "MAX_CANDIDATES": k, "MAX_TRIALS": 1, "SEED": seed}
    res = lkh_solve(lkh_bin, xy=xy, problem_file=problem_file, ew=ew, params=p, workroot=workroot,
                    want_cands=True)
    if "cand" not in res:
        raise RuntimeError(f"no candidate file written ({ctype}): {res['stdout'][-800:]}")
    return {"cand": pad_cols(res["cand"]["cand"], k), "pre_time": res["pre_time"],
            "wall": res["wall"], "cand_stats": res["cand_stats"]}


def task_anytime(lkh_bin, workroot, budget, seed, xy=None, problem_file=None, ew="EUC_2D",
                 params=None, cand_file=None, pi_file=None, timeout=None):
    """timeout: wall-clock cap in seconds (LKH is stopped there; the trace printed so far is kept). LKH's own
    TIME_LIMIT does not count preprocessing, so a run with a long ascent needs the cap to stay within budget."""
    p = {"TIME_LIMIT": float(budget), "MAX_TRIALS": 100_000_000, "SEED": int(seed)}
    p.update(params or {})
    res = lkh_solve(lkh_bin, xy=xy, problem_file=problem_file, ew=ew, params=p, workroot=workroot,
                    cand_file=cand_file, pi_file=pi_file, timeout=budget * 3 + 600 if timeout is None else timeout)
    return {k: res.get(k) for k in ("events", "pre_time", "ascent_time", "lower_bound", "wall",
                                    "cost_min", "time_total", "cand_stats", "returncode",
                                    "timed_out", "t_start")}


# ------------------------------------------------------------ tour ensembles (TourFold)
def task_ensemble(lkh_bin, workroot, M, seed0, xy, problem_file=None, ew="EUC_2D", ctype="POPMUSIC",
                  k=CAND_K, extra_params=None):
    """M independent single-trial LKH tours that share one ascent-free candidate set.
    Run 0 builds the candidate set (POPMUSIC by default) and writes it; runs 1..M-1 read it together with
    zero penalties, so no run pays for a subgradient ascent. Returns successor arrays (n, M), tour costs,
    per-run wall times (sequential, one core), the shared candidate set and the best tour."""
    os.makedirs(workroot or tempfile.gettempdir(), exist_ok=True)
    wd = tempfile.mkdtemp(dir=workroot, prefix="ens_")
    try:
        n = len(xy)
        if problem_file is None:
            problem_file = os.path.join(wd, "p.tsp")
            write_tsp(problem_file, xy, ew=ew)
        cand_path, pi_path = os.path.join(wd, "c.cand"), os.path.join(wd, "zero.pi")
        base = {"MAX_TRIALS": 1, "SUBGRADIENT": "NO", "MAX_CANDIDATES": k}
        base.update(extra_params or {})
        tours, costs, walls = [], [], []
        cand = None
        for m in range(M):
            p = dict(base, SEED=int(seed0) + m)
            if m == 0:
                p.update(CANDIDATE_SET_TYPE=ctype, CANDIDATE_FILE=cand_path)
            else:
                p.update(CANDIDATE_FILE=cand_path, PI_FILE=pi_path)
            res = run_lkh(lkh_bin, problem_file, p, os.path.join(wd, f"r{m}"))
            if not is_valid_tour(res["tour"], n):
                raise RuntimeError(f"ensemble run {m} gave no valid tour: {res['stdout'][-500:]}")
            if m == 0:
                if not os.path.exists(cand_path):
                    raise RuntimeError("LKH did not write the ensemble candidate file: " + res["stdout"][-500:])
                cand = read_candidates(cand_path, k_pad=k)
                write_pi(pi_path, np.zeros(n))
            tours.append(res["tour"])
            costs.append(tour_length(xy, res["tour"], ew))
            walls.append(res["wall"])
        succ = np.stack([tour_succ_pred(t)[0] for t in tours], 1).astype(np.int32)
        b = int(np.argmin(costs))
        return {"succ": succ, "costs": np.array(costs), "walls": np.array(walls),
                "cand": pad_cols(cand["cand"], k).astype(np.int32), "best": tours[b].astype(np.int32),
                "best_cost": float(costs[b])}
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def task_first_trial(lkh_bin, workroot, seed, xy, problem_file=None, ew="EUC_2D", ctype="POPMUSIC", k=CAND_K,
                     timeout=None):
    """The first trial of an ascent-free LKH run: candidate generation (POPMUSIC by default) and one trial.
    With the same seed it is the first trial of the corresponding full run, so a multi-start search can pause
    here, read its tours as a TourFold ensemble and resume. Returns the tour, its length, the run's own candidate
    lists (n, k) and the trace (preprocessing time, improvement events, wall time)."""
    p = {"CANDIDATE_SET_TYPE": ctype, "SUBGRADIENT": "NO", "MAX_TRIALS": 1, "MAX_CANDIDATES": int(k),
         "SEED": int(seed)}
    res = lkh_solve(lkh_bin, xy=None if problem_file else xy, problem_file=problem_file, ew=ew, params=p,
                    workroot=workroot, want_cands=True, timeout=timeout)
    n = len(xy)
    if not is_valid_tour(res["tour"], n):
        raise RuntimeError(f"first trial gave no valid tour (rc={res['returncode']}): {res['stdout'][-500:]}")
    if "cand" not in res:
        raise RuntimeError("LKH did not write the candidate file: " + res["stdout"][-500:])
    return {"tour": res["tour"].astype(np.int32), "cost": float(tour_length(xy, res["tour"], ew)),
            "cand": pad_cols(res["cand"]["cand"], k).astype(np.int32), "events": res["events"],
            "pre_time": res["pre_time"], "wall": res["wall"], "cost_min": res["cost_min"],
            "t_start": res["t_start"]}


def worker_ping(seconds=0.0):
    """Starts a pool worker (and its imports) ahead of timed work; returns the worker's process id."""
    time.sleep(seconds)
    return os.getpid()


def ensemble_pred(succ):
    n, M = succ.shape
    pred = np.empty_like(succ)
    ar = np.arange(n, dtype=succ.dtype)
    for m in range(M):
        pred[succ[:, m], m] = ar
    return pred


def ensemble_frequency(nb, succ):
    """freq[i, a] = share of ensemble tours that use edge (i, nb[i, a]) in either direction."""
    pred = ensemble_pred(succ)
    hit = (nb[:, :, None] == succ[:, None, :]) | (nb[:, :, None] == pred[:, None, :])
    return hit.mean(-1)


def frequency_candidates(nb, d, freq, k=CAND_K):
    """'Count' baseline: rank each node's kNN edges by ensemble frequency, ties broken by distance."""
    order = np.lexsort((d, -freq), axis=1)
    return np.take_along_axis(nb, order[:, :k], 1)


def mix_candidates(first, rest, k_first=3, k=CAND_K, fill=None):
    """Mixed candidate lists. Per node: the first k_first entries of `first`, then the entries of `rest` (in
    their order) that are not in the list yet, then - if `fill` is given - entries of `fill` not in the list yet,
    until k entries. Inputs are (n, *) arrays of 0-based node ids, -1 padded; a node never lists itself.
    Returns (n, k), -1 padded where fewer than k distinct neighbours are available."""
    first = np.asarray(first, dtype=np.int64)
    n = len(first)
    parts = [first[:, :k_first], np.asarray(rest, dtype=np.int64).reshape(n, -1)]
    if fill is not None:
        parts.append(np.asarray(fill, dtype=np.int64).reshape(n, -1))
    L = np.concatenate(parts, 1)
    if L.shape[1] < k:
        L = np.concatenate([L, np.full((n, k - L.shape[1]), -1, dtype=np.int64)], 1)
    W = L.shape[1]
    dup = ((L[:, :, None] == L[:, None, :]) & np.tri(W, W, -1, dtype=bool)[None]).any(2)   # seen earlier in row
    valid = (L >= 0) & ~dup & (L != np.arange(n)[:, None])
    order = np.argsort(~valid, axis=1, kind="stable")[:, :k]
    return np.where(np.take_along_axis(valid, order, 1), np.take_along_axis(L, order, 1), -1)


def fixing_stats(nb, score, tour, thresholds):
    """Precision / recall of 'fix edge if score >= t' against a reference tour, per threshold.
    Edges are undirected: an edge counts once and is fixed if either direction passes."""
    n, K = nb.shape
    lab = tour_edge_labels(nb, tour)
    i = np.repeat(np.arange(n), K)
    j = nb.reshape(-1)
    key = np.minimum(i, j) * n + np.maximum(i, j)
    s, y = score.reshape(-1), lab.reshape(-1)
    uniq, inv = np.unique(key, return_inverse=True)
    best = np.full(len(uniq), -np.inf)
    np.maximum.at(best, inv, s)
    is_tour = np.zeros(len(uniq), dtype=bool)
    is_tour[inv[y]] = True
    out = []
    for t in thresholds:
        sel = best >= t
        tp, nf = int((sel & is_tour).sum()), int(sel.sum())
        out.append({"t": float(t), "fixed": nf, "tp": tp, "n": int(n),
                    "precision": tp / nf if nf else float("nan"), "recall": tp / n})
    return out


# ------------------------------------------------------------ guided LKH runs (TourFold evaluation)
def select_fixed_edges(nb, prob, tau):
    """Greedy, consistent set of edges to fix: highest probability first, degree <= 2, no cycles.
    prob[i, a] is the probability of edge (i, nb[i, a]); both directions are merged by max."""
    n, K = nb.shape
    i = np.repeat(np.arange(n), K)
    j = nb.reshape(-1)
    p = prob.reshape(-1)
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    key = lo * n + hi
    uniq, inv = np.unique(key, return_inverse=True)
    best = np.full(len(uniq), -np.inf)
    np.maximum.at(best, inv, p)
    order = np.argsort(-best, kind="stable")
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    deg = np.zeros(n, dtype=np.int64)
    out = []
    for e in order:
        if best[e] < tau or len(out) >= n - 1:
            break
        a, c = int(uniq[e] // n), int(uniq[e] % n)
        if deg[a] >= 2 or deg[c] >= 2:
            continue
        ra, rc = find(a), find(c)
        if ra == rc:
            continue
        parent[ra] = rc
        deg[a] += 1
        deg[c] += 1
        out.append((a, c))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def add_fixed_edges(src_problem, dst_problem, edges):
    """Copy a TSPLIB problem file and append a FIXED_EDGES_SECTION (0-based edges in, 1-based out)."""
    with _open_text(src_problem) as f:
        lines = [ln.rstrip("\n") for ln in f]
    while lines and lines[-1].strip() in ("", "EOF"):
        lines.pop()
    body = "\n".join(f"{a + 1} {b + 1}" for a, b in np.asarray(edges).tolist())
    with open(dst_problem, "w") as f:
        f.write("\n".join(lines) + "\nFIXED_EDGES_SECTION\n" + (body + "\n" if body else "") + "-1\nEOF\n")


def task_guided(lkh_bin, workroot, budget, seed, xy, cand, pi=None, init_tour=None, problem_file=None,
                ew="EUC_2D", fixed_edges=None, alpha=None, extra_params=None, return_tour=False, timeout=None):
    """LKH with externally supplied candidates (+ penalties, initial tour, fixed edges); no ascent."""
    os.makedirs(workroot or tempfile.gettempdir(), exist_ok=True)
    wd = tempfile.mkdtemp(dir=workroot, prefix="g_")
    try:
        n = len(xy)
        base = problem_file
        if base is None:
            base = os.path.join(wd, "p.tsp")
            write_tsp(base, xy, ew=ew)
        pf = base
        if fixed_edges is not None and len(fixed_edges):
            pf = os.path.join(wd, "fixed.tsp")
            add_fixed_edges(base, pf, fixed_edges)
        cf, pif = os.path.join(wd, "g.cand"), os.path.join(wd, "g.pi")
        write_candidates(cf, cand, alpha)
        write_pi(pif, np.zeros(n) if pi is None else pi)
        params = {"SUBGRADIENT": "NO", "TIME_LIMIT": float(budget), "MAX_TRIALS": 100_000_000,
                  "SEED": int(seed), "MAX_CANDIDATES": int(np.asarray(cand).shape[1])}
        if init_tour is not None:
            tp = os.path.join(wd, "init.tour")
            write_tour(tp, init_tour)
            params["INITIAL_TOUR_FILE"] = tp
        params.update(extra_params or {})
        params["CANDIDATE_FILE"], params["PI_FILE"] = cf, pif
        res = run_lkh(lkh_bin, pf, params, os.path.join(wd, "run"),
                      timeout=budget * 3 + 600 if timeout is None else timeout)
        ok = is_valid_tour(res["tour"], n)
        cost = tour_length(xy, res["tour"], ew) if ok else None
        out = {k: res.get(k) for k in ("events", "pre_time", "wall", "cost_min", "returncode", "timed_out",
                                       "t_start")}
        out["cost_check"] = cost
        if fixed_edges is not None and len(fixed_edges):
            # LKH gives fixed edges length 0, so its reported costs miss their total length. Every tour of the
            # run contains all fixed edges, hence one constant shift restores true lengths for all events.
            fe = np.asarray(fixed_edges, dtype=np.int64).reshape(-1, 2)
            xa = np.asarray(xy)
            out["fixed_len"] = float(edge_len(xa[fe[:, 0]], xa[fe[:, 1]], ew).sum())
            if cost is not None and res.get("cost_min") is not None:
                shift = float(cost - res["cost_min"])
            else:
                shift = out["fixed_len"]
            out["cost_shift"] = shift
            out["events"] = [(i, int(round(c + shift)), t) for i, c, t in (res.get("events") or [])]
            if res.get("cost_min") is not None:
                out["cost_min"] = int(round(res["cost_min"] + shift))
        if not ok:
            out["stdout_tail"] = (res.get("stdout") or "")[-800:]
        if return_tour:
            out["tour"] = res["tour"] if ok else None
        return out
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def abs_events(res, offset=0.0):
    """(cost, absolute time) pairs: offset + LKH preprocessing + time inside the run."""
    pre = (res.get("pre_time") or 0.0) + offset
    return [(int(c), pre + float(t)) for _, c, t in (res.get("events") or [])]


def curve_abs(events, ref, budgets):
    ev = sorted(events, key=lambda e: e[1])
    out = []
    for b in budgets:
        cs = [c for c, t in ev if t <= b]
        out.append(gap_pct(min(cs), ref) if cs else np.nan)
    return out


def time_to_gap_abs(events, ref, target_pct):
    for c, t in sorted(events, key=lambda e: e[1]):
        if gap_pct(c, ref) <= target_pct + 1e-12:
            return t
    return float("inf")
