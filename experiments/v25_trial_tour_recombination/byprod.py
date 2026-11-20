"""Helpers of run v25 (pilot): the trial tours that LKH-3 discards, recorded and recombined with GA-EAX.

Every trial of LKH-3 ends in a locally optimal tour. LKH-3 merges it with its best tour and keeps only the best
tour. This module

  * patches FindTour.c of LKH-3.0.14 so that a copy of LKH-3 writes the tour of every trial and every new best tour
    to a file (nothing of the search state is changed),
  * patches env.cpp of GA-EAX (Nagata and Kobayashi, 2013) so that it prints a wall-clock time stamp per generation
    and stops at a wall-clock deadline with its best tour,
  * patches evaluator.cpp of GA-EAX so that the lists of nearest cities are found with a grid instead of a scan of
    all cities for every list entry (the same lists; at 50,000 nodes the scan takes about a minute),
  * reads the recorded tours, builds populations from them and runs GA-EAX on a population,
  * holds the paired statistics used by the analysis cell.

The patches are applied by exact string replacement after the SHA-256 of the original file has been compared with
the value below, so another source version is refused and never patched wrongly.

Pure Python and NumPy; tsplab is not imported here.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from math import comb

import numpy as np

VERSION = "v25-byprod-1"

# ----------------------------------------------------------------------------------------------- source patches
LKH_FINDTOUR_SHA256 = "567e72acc9b7392488dd158be4df0267560424204a0e88d2d9b62f923a7c0eee"      # LKH-3.0.14
EAX_COMMIT = "015dfbe9f267230f78787bd244af393ffc018900"                                     # GA-EAX, 2021-12-29
EAX_ENV_SHA256 = {"Normal": "f923055933b26b3bde02bd1467abca92e99a357d590e62f8044f047ffc8fa6e5",
                  "Large": "5c58951f29477c34f0bd10364147e70ce475899799324775c7df98ddabeae35c"}
EAX_EVALUATOR_SHA256 = {"Normal": "85c07c041c7ea32d42d3b180d1eed30d25707bfeb01393c010803ffbcdcd540c",
                        "Large": "dd1582b36c0c595209cf6d67c287f20000dc24737cd252b474bb03e1763b9767"}
EAX_SOURCES = ["main.cpp", "env.cpp", "cross.cpp", "evaluator.cpp", "indi.cpp", "rand.cpp", "kopt.cpp", "sort.cpp"]
# SHA-256 over the names and contents of all .cpp and .h files of a version (see source_digest).
EAX_TREE_SHA256 = {"Normal": "d4d016b831745a54bd39c458c447e55ac4039e8606220d6fbb2b4ecd130b5813",
                   "Large": "bc5785bdb728b8d46472480eaa5e962faa547b99b02b4d47055ac58971af92af"}

_LKH_FUNCTION = r'''
#include <sys/time.h>

/* Run v25 (pilot): record tours without changing the search.
 * Kind 0: the locally optimal tour of the trial that has just ended (before it is merged with the best tour).
 * Kind 1: the new best tour of the run.
 * Record: five 64-bit integers (kind, trial, cost, milliseconds of LKH's clock since the start of the search,
 * milliseconds of the wall clock since the epoch), then Dimension 32-bit node numbers in tour order. */
static void BetaTourRecord(int Kind, GainType RecordCost, double RecordEntryTime)
{
    static FILE *File = 0;
    static int Tried = 0;
    static int *Buffer = 0;
    long long Head[5];
    struct timeval Now;
    int i;

    if (!Tried) {
        char *Name = getenv("BETATOUR_TRIAL_DUMP");
        Tried = 1;
        if (Name)
            File = fopen(Name, "wb");
        if (File)
            Buffer = (int *) malloc((size_t) Dimension * sizeof(int));
    }
    if (!File || !Buffer)
        return;
    if (Kind == 0) {
        Node *N = FirstNode;
        for (i = 0; i < Dimension; i++) {
            Buffer[i] = N->Id;
            N = N->Suc;
        }
    } else
        for (i = 0; i < Dimension; i++)
            Buffer[i] = BetterTour[i + 1];
    gettimeofday(&Now, 0);
    Head[0] = Kind;
    Head[1] = Trial;
    Head[2] = (long long) RecordCost;
    Head[3] = (long long) (1000.0 * (GetTime() - RecordEntryTime));
    Head[4] = (long long) Now.tv_sec * 1000 + Now.tv_usec / 1000;
    fwrite(Head, sizeof(long long), 5, File);
    fwrite(Buffer, sizeof(int), (size_t) Dimension, File);
    fflush(File);
}
'''

_LKH_EDITS = [
    ('#include "LKH.h"\n',
     '#include "LKH.h"\n' + _LKH_FUNCTION),
    ("        Cost = LinKernighan();\n        if (GetTime() - EntryTime < TimeLimit &&",
     "        Cost = LinKernighan();\n        BetaTourRecord(0, Cost, EntryTime);\n"
     "        if (GetTime() - EntryTime < TimeLimit &&"),
    ("            RecordBetterTour();\n            if ((BetterPenalty < BestPenalty ||",
     "            RecordBetterTour();\n            BetaTourRecord(1, BetterCost, EntryTime);\n"
     "            if ((BetterPenalty < BestPenalty ||"),
]

_EAX_OLD = '    printf( "%d: %d %lf\\n", fCurNumOfGen, fBestValue, fAverageValue );\n'
_EAX_NEW = r'''    {
      /* Run v25 (pilot): wall-clock time stamp of every generation and an optional wall-clock deadline. */
      struct timeval pilotNow;
      gettimeofday( &pilotNow, NULL );
      double pilotTime = (double)pilotNow.tv_sec + 1e-6 * (double)pilotNow.tv_usec;
      printf( "%d: %d %lf %.3f %d\n", fCurNumOfGen, fBestValue, fAverageValue, pilotTime, fStage );
      fflush( stdout );
      const char* pilotDeadline = getenv( "EAX_DEADLINE" );
      if( pilotDeadline != NULL && pilotTime >= atof( pilotDeadline ) ) break;
    }
'''
_EAX_HEAD = "#include <stdio.h>\n#include <stdlib.h>\n#include <sys/time.h>\n"

_EAX_NEAREST = r'''
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <utility>

/* Run v25 (pilot): the nearest cities of every city, found with a grid instead of a scan of all cities.
 * The result is the list of the original loops: ascending rounded distance and, among equal distances, the larger
 * city number first. nearDis may be NULL. Requires N > K + 1. */
static void PilotNearest( const double* x, const double* y, int N, int K, int** nearCity, int** nearDis )
{
  double xmin = x[0], xmax = x[0], ymin = y[0], ymax = y[0];
  for( int i = 1; i < N; ++i ){
    if( x[i] < xmin ) xmin = x[i];
    if( x[i] > xmax ) xmax = x[i];
    if( y[i] < ymin ) ymin = y[i];
    if( y[i] > ymax ) ymax = y[i];
  }
  double span = ( xmax - xmin > ymax - ymin ) ? xmax - xmin : ymax - ymin;
  int G = (int)sqrt( (double)N / 2.0 );
  if( G < 1 ) G = 1;
  double cell = span / (double)G;
  if( !( cell > 0.0 ) ){ cell = 1.0; G = 1; }
  std::vector<int> cx( N ), cy( N ), head( G * G + 1, 0 ), order( N );
  for( int i = 0; i < N; ++i ){
    int a = (int)( ( x[i] - xmin ) / cell ), b = (int)( ( y[i] - ymin ) / cell );
    cx[i] = a < 0 ? 0 : ( a >= G ? G - 1 : a );
    cy[i] = b < 0 ? 0 : ( b >= G ? G - 1 : b );
    ++head[ cx[i] * G + cy[i] + 1 ];
  }
  for( int c = 0; c < G * G; ++c ) head[ c + 1 ] += head[ c ];
  {
    std::vector<int> next( head.begin(), head.end() - 1 );
    for( int i = 0; i < N; ++i ) order[ next[ cx[i] * G + cy[i] ]++ ] = i;
  }
  std::vector< std::pair<int,int> > cand;
  for( int i = 0; i < N; ++i ){
    cand.clear();
    for( int r = 0; ; ++r ){
      for( int a = cx[i] - r; a <= cx[i] + r; ++a ){
        if( a < 0 || a >= G ) continue;
        for( int b = cy[i] - r; b <= cy[i] + r; ++b ){
          if( b < 0 || b >= G ) continue;
          if( abs( a - cx[i] ) != r && abs( b - cy[i] ) != r ) continue;   /* the cells of ring r only */
          for( int p = head[ a * G + b ]; p < head[ a * G + b + 1 ]; ++p ){
            int j = order[ p ];
            if( j == i ) continue;
            int d = (int)(sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))+0.5);
            cand.push_back( std::make_pair( d, -j ) );
          }
        }
      }
      if( cx[i] - r <= 0 && cx[i] + r >= G - 1 && cy[i] - r <= 0 && cy[i] + r >= G - 1 ) break;   /* whole grid */
      if( (int)cand.size() >= K ){
        std::nth_element( cand.begin(), cand.begin() + ( K - 1 ), cand.end() );
        /* A city outside the rings 0..r is farther than r * cell from city i. */
        if( (double)r * cell > (double)cand[ K - 1 ].first + 1.0 ) break;
      }
    }
    std::partial_sort( cand.begin(), cand.begin() + K, cand.end() );
    nearCity[ i ][ 0 ] = i;
    if( nearDis != NULL ) nearDis[ i ][ 0 ] = 0;
    for( int k = 1; k <= K; ++k ){
      nearCity[ i ][ k ] = -cand[ k - 1 ].second;
      if( nearDis != NULL ) nearDis[ i ][ k ] = cand[ k - 1 ].first;
    }
  }
}
'''
_EAX_EVALUATOR_HEAD = '#ifndef __EVALUATOR__\n#include "evaluator.h"\n#endif\n'
_EAX_NORMAL_START = "  for( ci = 0; ci < Ncity; ++ci )\n  {\n    for( j3 = 0; j3 < Ncity; ++j3 ) checkedN[ j3 ] = 0;"
_EAX_LARGE_START = '  for( int i = 0; i < Ncity ; ++i )\n  {\n    if( strcmp( fType, "EUC_2D" ) == 0  ) {'
_EAX_LARGE_END = "  delete [] DisTmp;\n}"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replace_once(text, old, new):
    assert text.count(old) == 1, "the source does not hold the expected line exactly once: " + old[:60].strip()
    return text.replace(old, new)


def patch_lkh_findtour(text):
    """FindTour.c of LKH-3.0.14 -> the same file with the recording calls (see _LKH_FUNCTION)."""
    assert sha256_text(text) == LKH_FINDTOUR_SHA256, "FindTour.c is not the file of LKH-3.0.14"
    for old, new in _LKH_EDITS:
        text = _replace_once(text, old, new)
    return text


def patch_eax_env(text, version):
    """env.cpp of GA-EAX (version 'Normal' or 'Large') -> time stamps and deadline (see _EAX_NEW)."""
    assert sha256_text(text) == EAX_ENV_SHA256[version], "env.cpp is not the file of the pinned GA-EAX commit"
    return _EAX_HEAD + _replace_once(text, _EAX_OLD, _EAX_NEW)


def patch_eax_evaluator(text, version):
    """evaluator.cpp of GA-EAX -> the lists of nearest cities come from PilotNearest (EUC_2D, more cities than the
    list length); the original loops stay in place for every other case."""
    assert sha256_text(text) == EAX_EVALUATOR_SHA256[version], "evaluator.cpp is not the file of the pinned GA-EAX commit"
    assert text.startswith(_EAX_EVALUATOR_HEAD)
    if version == "Normal":
        assert text.count(_EAX_NORMAL_START) == 1
        start = text.index(_EAX_NORMAL_START)
        end = text.index("\n}\n", start)
        call = ('  if( strcmp( type, "EUC_2D" ) == 0 && Ncity > fNearNumMax + 1 )\n'
                "    PilotNearest( x, y, Ncity, fNearNumMax, fNearCity, NULL );\n  else\n  {\n")
        text = text[:start] + call + text[start:end] + "\n  }" + text[end:]
    else:
        assert text.count(_EAX_LARGE_START) == 1 and text.count(_EAX_LARGE_END) == 1
        start, end = text.index(_EAX_LARGE_START), text.index(_EAX_LARGE_END)
        call = ("  if( Ncity > fNearNumMax + 1 )\n"
                "    PilotNearest( x, y, Ncity, fNearNumMax, fNearCity, fEdgeDisOrder );\n  else\n  {\n")
        text = text[:start] + call + text[start:end].rstrip("\n") + "\n  }\n\n" + text[end:]
    return _EAX_EVALUATOR_HEAD + _EAX_NEAREST + text[len(_EAX_EVALUATOR_HEAD):]


def source_digest(folder, suffixes=(".cpp", ".h")):
    """One SHA-256 over the names and contents of the source files of a folder."""
    h = hashlib.sha256()
    for name in sorted(os.listdir(folder)):
        if name.endswith(suffixes):
            h.update(name.encode("ascii") + b"\0")
            with open(os.path.join(folder, name), "rb") as f:
                h.update(f.read() + b"\0")
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------ processes
def run_process(cmd, cwd, env=None, timeout=None):
    """Run a process; on a timeout it is stopped and the output printed so far is kept."""
    full_env = dict(os.environ)
    full_env.update({k: str(v) for k, v in (env or {}).items()})
    t_start = time.time()
    try:
        done = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=full_env, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        out, code, timed_out = done.stdout, done.returncode, False
    except subprocess.TimeoutExpired as exc:
        out, code, timed_out = exc.stdout or b"", None, True
    return dict(stdout=out.decode("latin-1", "replace"), returncode=code, timed_out=timed_out, t_start=t_start,
                wall=time.time() - t_start)


def write_par(path, params):
    with open(path, "w") as f:
        for key, value in params.items():
            f.write(f"{key} = {value}\n")


_PRE_RX = re.compile(r"Preprocessing time = ([0-9.]+) sec")
_TRIAL_RX = re.compile(r"^(\*?)[ \t]*(\d+):[ \t]*Cost[ \t]*=[ \t]*(-?\d+)[^\n]*?Time[ \t]*=[ \t]*([\d.]+)[ \t]*sec", re.M)


def lkh_preprocessing_s(stdout):
    m = _PRE_RX.search(stdout)
    return float(m.group(1)) if m else None


def lkh_trial_costs(stdout):
    """(trial, cost) of every trial line of a TRACE_LEVEL = 2 output (improving trials carry a star)."""
    return [(int(k), int(c)) for _, k, c, _ in _TRIAL_RX.findall(stdout)]


# ------------------------------------------------------------------------------------------------- recorded tours
def record_dtype(n):
    return np.dtype([("kind", "<i8"), ("trial", "<i8"), ("cost", "<i8"), ("lkh_ms", "<i8"), ("epoch_ms", "<i8"),
                     ("ids", "<i4", (int(n),))])


def read_records(path, n):
    """All complete records of a file written by the recording copy of LKH (a last, cut record is dropped)."""
    dtype = record_dtype(n)
    count = os.path.getsize(path) // dtype.itemsize if os.path.exists(path) else 0
    if count == 0:
        return np.zeros(0, dtype=dtype)
    return np.fromfile(path, dtype=dtype, count=count)


def tour_of(record):
    """0-based tour of a record."""
    return record["ids"].astype(np.int64) - 1


def is_permutation(tour, n):
    tour = np.asarray(tour)
    return tour.shape == (n,) and tour.min() == 0 and tour.max() == n - 1 and len(np.unique(tour)) == n


def edge_keys(tour, n):
    """Sorted undirected edge keys of a tour."""
    a = np.asarray(tour, dtype=np.int64)
    b = np.roll(a, -1)
    return np.sort(np.minimum(a, b) * n + np.maximum(a, b))


def improvements(records, start_ms):
    """[[seconds since start_ms, cost], ...] of the new best tours of one process, in time order."""
    best = records[records["kind"] == 1]
    return [[round((int(e) - start_ms) / 1000.0, 3), int(c)] for e, c in zip(best["epoch_ms"], best["cost"])]


def best_before(curve, limit):
    """Smallest cost of a [[t, cost], ...] curve at times <= limit, or None."""
    costs = [c for t, c in curve if t <= limit + 1e-9]
    return min(costs) if costs else None


def select_population(record_list, cut_ms, per_process, n, trial_tours=True):
    """Population at wall-clock time cut_ms: per process its best tour at that time and, with trial_tours, its last
    (per_process - 1) distinct trial tours that ended before that time. Returns 0-based tours, the costs recorded by
    LKH and one information dictionary per process."""
    tours, costs, info = [], [], []
    for records in record_list:
        before = records[records["epoch_ms"] <= cut_ms]
        best = before[before["kind"] == 1]
        trials = before[before["kind"] == 0]
        chosen, seen = [], set()
        if len(best):
            chosen.append(best[-1])
            seen.add(edge_keys(tour_of(best[-1]), n).tobytes())
        used_trials = []
        if trial_tours:
            for i in range(len(trials) - 1, -1, -1):
                if len(chosen) >= per_process:
                    break
                key = edge_keys(tour_of(trials[i]), n).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(trials[i])
                used_trials.append(int(trials[i]["trial"]))
        for rec in chosen:
            tours.append(tour_of(rec))
            costs.append(int(rec["cost"]))
        info.append(dict(trials_before_cut=int(len(trials)), best_tours_before_cut=int(len(best)),
                         best_cost_at_cut=int(best[-1]["cost"]) if len(best) else None,
                         tours_used=int(len(chosen)),
                         trial_range=[min(used_trials), max(used_trials)] if used_trials else None))
    return tours, costs, info


def population_stats(tours, n, pairs=200, seed=0):
    keys = [edge_keys(t, n) for t in tours]
    union, count = np.unique(np.concatenate(keys), return_counts=True)
    rng = np.random.default_rng(seed)
    shared = []
    if len(keys) >= 2:
        for _ in range(min(pairs, len(keys) * (len(keys) - 1) // 2)):
            i, j = rng.choice(len(keys), 2, replace=False)
            shared.append(float(np.isin(keys[i], keys[j], assume_unique=True).mean()))
    return dict(tours=int(len(tours)), distinct=int(len({k.tobytes() for k in keys})),
                union_edges=int(len(union)), common_edges=int((count == len(keys)).sum()),
                pairwise_shared=float(np.mean(shared)) if shared else 1.0)


# ---------------------------------------------------------------------------------------------------------- GA-EAX
def write_eax_problem(path, xy):
    """EUC_2D instance in the layout that GA-EAX reads (keyword, colon and value separated by blanks)."""
    xy = np.asarray(xy)
    integral = np.issubdtype(xy.dtype, np.integer) or bool(np.all(xy == np.round(xy)))
    with open(path, "w") as f:
        f.write(f"NAME : instance\nTYPE : TSP\nDIMENSION : {len(xy)}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
        if integral:
            f.writelines(f"{i} {int(x)} {int(y)}\n" for i, (x, y) in enumerate(xy, 1))
        else:
            f.writelines(f"{i} {float(x)!r} {float(y)!r}\n" for i, (x, y) in enumerate(xy, 1))
        f.write("EOF\n")


def write_population(path, tours, costs):
    with open(path, "w") as f:
        for tour, cost in zip(tours, costs):
            f.write(f"{len(tour)} {int(cost)}\n" + " ".join(map(str, (np.asarray(tour) + 1).tolist())) + " \n")


_EAX_LINE = re.compile(r"^(\d+): (-?\d+) ([0-9.eE+-]+) ([0-9.]+) (\d)$", re.M)


def parse_eax_output(stdout, launch_epoch):
    """Generation lines of the patched GA-EAX: improvements as [[seconds since launch, best], ...]."""
    rows = [(int(g), int(b), float(a), float(t), int(s)) for g, b, a, t, s in _EAX_LINE.findall(stdout)]
    curve, best = [], None
    for g, b, a, t, s in rows:
        if best is None or b < best:
            best = b
            curve.append([round(t - launch_epoch, 3), b])
    return dict(generations=rows[-1][0] if rows else None, lines=len(rows), curve=curve,
                first_line_s=round(rows[0][3] - launch_epoch, 3) if rows else None,
                last_line_s=round(rows[-1][3] - launch_epoch, 3) if rows else None,
                last_best=rows[-1][1] if rows else None, last_average=rows[-1][2] if rows else None,
                last_stage=rows[-1][4] if rows else None)


def read_eax_best(path, n):
    """Last tour of a <name>_BestSol file: (0-based tour, length reported by GA-EAX)."""
    with open(path) as f:
        tokens = f.read().split()
    assert len(tokens) >= n + 2 and len(tokens) % (n + 2) == 0, "unexpected GA-EAX tour file"
    block = tokens[-(n + 2):]
    assert int(block[0]) == n
    return np.array(block[2:], dtype=np.int64) - 1, int(block[1])


def run_eax(binary, problem_file, workdir, name, pop_size, kids=30, population_file=None, deadline_epoch=None,
            grace_s=120.0):
    """One GA-EAX run. With population_file the run starts from these tours (pop_size = their number); without it
    GA-EAX builds its own population. deadline_epoch: wall-clock time at which GA-EAX stops with its best tour.
    Returns the parsed output, the best tour (0-based) and the length reported by GA-EAX."""
    os.makedirs(workdir, exist_ok=True)
    for suffix in ("_BestSol", "_Result"):
        path = os.path.join(workdir, name + suffix)
        if os.path.exists(path):
            os.remove(path)
    cmd = [binary, "1", name, str(int(pop_size)), str(int(kids)), problem_file]
    if population_file is not None:
        cmd.append(population_file)
    env = {} if deadline_epoch is None else {"EAX_DEADLINE": f"{deadline_epoch:.3f}"}
    timeout = None if deadline_epoch is None else max(5.0, deadline_epoch - time.time()) + grace_s
    res = run_process(cmd, workdir, env=env, timeout=timeout)
    out = parse_eax_output(res["stdout"], res["t_start"])
    out.update(returncode=res["returncode"], timed_out=res["timed_out"], wall_s=float(res["wall"]),
               launch_epoch=float(res["t_start"]), deadline_s=None if deadline_epoch is None
               else float(deadline_epoch - res["t_start"]), tail=res["stdout"][-400:])
    best_file = os.path.join(workdir, name + "_BestSol")
    out["finished"] = bool(res["returncode"] == 0 and os.path.exists(best_file))
    out["stopped_by_deadline"] = bool(out["finished"] and deadline_epoch is not None and out["last_line_s"] is not None
                                      and res["t_start"] + out["last_line_s"] >= deadline_epoch - 1e-3)
    return out, best_file


# ------------------------------------------------------------------------------------------------------ statistics
def sign_test_p(wins, losses):
    """Exact two-sided sign test (ties excluded)."""
    m = wins + losses
    if m == 0:
        return 1.0
    k = min(wins, losses)
    return float(min(1.0, 2.0 * sum(comb(m, i) for i in range(k + 1)) / 2.0 ** m))


def paired_stats(values, bootstrap_seed, replicates=5000):
    """Paired percentage differences (first minus second arm): wins are negative values (the first arm is shorter)."""
    x = np.asarray(values, dtype=np.float64)
    assert x.ndim == 1 and len(x) and np.isfinite(x).all()
    wins, losses = int((x < -1e-12).sum()), int((x > 1e-12).sum())
    rng = np.random.default_rng(bootstrap_seed)
    means = x[rng.integers(0, len(x), size=(replicates, len(x)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return dict(instances=int(len(x)), win=wins, tie=int(len(x) - wins - losses), loss=losses,
                mean_delta_pct=float(x.mean()), median_delta_pct=float(np.median(x)),
                ci95_low_pct=float(low), ci95_high_pct=float(high), sign_p=sign_test_p(wins, losses))
