"""Tests of byprod.py with the real LKH-3.0.14 source and the real GA-EAX source.

    python experiments/v26_recombination_confirmation/tests/test_byprod.py --src src \\
        --lkh-source /path/to/LKH-3.0.14 --eax-source /path/to/GA-EAX --work /tmp/v26_unit

--lkh-source: a source tree of LKH-3.0.14 that holds a built LKH binary (the unmodified solver).
--eax-source: a clone of https://github.com/nagata-yuichi/GA-EAX at the commit pinned in byprod.py.

What is checked:
  1. The recording copy of LKH runs the same search as the unmodified LKH (same cost at every trial of a run with a
     fixed number of trials), and every recorded tour is a valid tour of the recorded length.
  2. The last best-tour record equals the tour that LKH writes itself.
  3. The patched GA-EAX starts from recorded tours, returns a valid tour that is not longer than the best tour it
     was given, reports its true length, and stops at a wall-clock deadline with its best tour.
  4. The population builder returns the tours it promises.
  5. The grid search for the nearest cities returns exactly the lists of GA-EAX's own loops (uniform, clustered,
     lattice with repeated points, decimal coordinates, points on a line, identical points), and GA-EAX built with
     it gives the same result, generation for generation, as GA-EAX built without it.
  6. (Run v26.) The recording copy writes its candidate file without changing the search; with BETATOUR_BEST_ONLY it
     writes only the new best tours. LKH-3's tour merging with MERGE_TOUR_FILEs reads that candidate file, gives the
     same search with and without recording, keeps the edges common to the merged tours and does not lengthen the
     best of them. The helpers for common edges, distinct tours and the Holm correction return the expected values.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import byprod as B


def build_recording_lkh(source, dest):
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("OBJ", "LKH", "*.o"))
    (dest / "SRC" / "OBJ").mkdir()
    path = dest / "SRC" / "FindTour.c"
    path.write_text(B.patch_lkh_findtour(path.read_text()))
    done = subprocess.run(["make", "-j", "2"], cwd=dest, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert done.returncode == 0 and (dest / "LKH").exists(), done.stdout[-2000:]
    return dest / "LKH"


HARNESS = r'''
#include <stdio.h>
/* The selection of GA-EAX's own loops (evaluator.cpp, version Large; version Normal selects in the same way). */
static void Original( const double* x, const double* y, int N, int K, int** nearCity, int** nearDis )
{
  int* DisTmp = new int[ N ];
  int* checkedN = new int[ N ];
  for( int i = 0; i < N; ++i ){
    for( int j = 0; j < N; ++j )
      DisTmp[ j ] = (int)(sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))+0.5);
    for( int j3 = 0; j3 < N; ++j3 ) checkedN[ j3 ] = 0;
    checkedN[ i ] = 1;
    nearCity[ i ][ 0 ] = i;
    nearDis[ i ][ 0 ] = 0;
    int city_num = 0, min_dis;
    for( int j1 = 1; j1 <= K; ++j1 ){
      min_dis = 100000000;
      for( int j2 = 0; j2 < N; ++j2 )
        if( DisTmp[ j2 ] <= min_dis && checkedN[ j2 ] == 0 ){ city_num = j2; min_dis = DisTmp[ j2 ]; }
      nearCity[ i ][ j1 ] = city_num;
      nearDis[ i ][ j1 ] = min_dis;
      checkedN[ city_num ] = 1;
    }
  }
}

int main( int argc, char** argv )
{
  int N = atoi( argv[2] ), K = atoi( argv[3] );
  double* x = new double[ N ];
  double* y = new double[ N ];
  FILE* fp = fopen( argv[1], "r" );
  for( int i = 0; i < N; ++i ) if( fscanf( fp, "%lf %lf", &x[i], &y[i] ) != 2 ) return 2;
  fclose( fp );
  int** a = new int*[ N ]; int** b = new int*[ N ]; int** c = new int*[ N ]; int** d = new int*[ N ];
  for( int i = 0; i < N; ++i ){ a[i] = new int[K+1]; b[i] = new int[K+1]; c[i] = new int[K+1]; d[i] = new int[K+1]; }
  Original( x, y, N, K, a, b );
  PilotNearest( x, y, N, K, c, d );
  for( int i = 0; i < N; ++i )
    for( int k = 0; k <= K; ++k )
      if( a[i][k] != c[i][k] || b[i][k] != d[i][k] ){
        printf( "DIFFERENT at city %d entry %d: %d (%d) / %d (%d)\n", i, k, a[i][k], b[i][k], c[i][k], d[i][k] );
        return 1;
      }
  printf( "EQUAL\n" );
  return 0;
}
'''


def check_nearest(work, T):
    """The grid search against GA-EAX's own selection, entry by entry."""
    folder = work / "nearest"
    folder.mkdir()
    (folder / "harness.cpp").write_text(B._EAX_NEAREST + HARNESS)
    done = subprocess.run(["g++", "-O3", "-w", "-o", "harness", "harness.cpp", "-lm"], cwd=folder,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert done.returncode == 0, done.stdout[-2000:]
    rng = np.random.default_rng(77)
    cases = {
        "uniform": T.to_int_coords(T.gen_instance(4000, "uniform", rng)).astype(float),
        "clustered": T.to_int_coords(T.gen_instance(4000, "clustered", rng)).astype(float),
        "lattice": rng.integers(0, 25, size=(3000, 2)).astype(float) * 10.0,
        "decimal": rng.random((3000, 2)) * 5000.0,
        "line": np.stack([rng.random(2000) * 1e5, np.zeros(2000)], axis=1),
        "identical": np.full((300, 2), 7.0),
        "tiny": rng.random((103, 2)) * 50.0,
    }
    for name, xy in cases.items():
        path = folder / (name + ".txt")
        with open(path, "w") as f:
            f.writelines(f"{float(a)!r} {float(b)!r}\n" for a, b in xy)
        for k in (50, 100):
            out = subprocess.run([str(folder / "harness"), str(path), str(len(xy)), str(k)], stdout=subprocess.PIPE,
                                 text=True)
            assert out.returncode == 0 and out.stdout.strip() == "EQUAL", (name, k, out.stdout)
    return sorted(cases)


def build_eax(source, dest, version, grid=True):
    folder = source / "GA_EAX_1.0" / version
    assert B.source_digest(folder) == B.EAX_TREE_SHA256[version], "GA-EAX source differs from the pinned commit"
    dest.mkdir(parents=True)
    for path in folder.iterdir():
        if path.suffix in {".cpp", ".h"}:
            shutil.copy2(path, dest / path.name)
    env = dest / "env.cpp"
    env.write_text(B.patch_eax_env(env.read_text(), version))
    if grid:
        evaluator = dest / "evaluator.cpp"
        evaluator.write_text(B.patch_eax_evaluator(evaluator.read_text(), version))
    done = subprocess.run(["g++", "-O3", "-w", "-o", "eax"] + B.EAX_SOURCES + ["-lm"], cwd=dest,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert done.returncode == 0 and (dest / "eax").exists(), done.stdout[-2000:]
    return dest / "eax"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--lkh-source", type=Path, required=True)
    parser.add_argument("--eax-source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=2000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.src))
    import tsplab as T

    work = args.work
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    plain = args.lkh_source / "LKH"
    assert plain.exists(), "build the unmodified LKH first"
    recording = build_recording_lkh(args.lkh_source, work / "lkh_rec")
    eax = {v: build_eax(args.eax_source, work / ("eax_" + v.lower()), v) for v in ("Normal", "Large")}
    plain_eax = {v: build_eax(args.eax_source, work / ("eax_plain_" + v.lower()), v, grid=False)
                 for v in ("Normal", "Large")}
    print("5a. grid search equals GA-EAX's own lists on:", ", ".join(check_nearest(work, T)))

    n = args.nodes
    xy = T.to_int_coords(T.gen_instance(n, "clustered", np.random.default_rng(4242)))
    problem = work / "p.tsp"
    T.write_tsp(str(problem), xy, name="unit", ew="EUC_2D")
    B.write_eax_problem(work / "p_eax.tsp", xy)

    def lkh(binary, tag, dump, extra=None, env=None, merge=(), trials=25):
        par = work / f"{tag}.par"
        B.write_par(par, {"PROBLEM_FILE": problem, "RUNS": 1, "TRACE_LEVEL": 2, "MAX_TRIALS": trials, "SEED": 101,
                          "CANDIDATE_SET_TYPE": "POPMUSIC", "SUBGRADIENT": "NO", "MAX_CANDIDATES": 5,
                          **(extra or {}), "OUTPUT_TOUR_FILE": work / f"{tag}.tour"}, merge_tour_files=merge)
        full_env = dict(env or {})
        if dump:
            full_env["BETATOUR_TRIAL_DUMP"] = dump
        res = B.run_process([binary, par], work, env=full_env or None, timeout=300)
        assert res["returncode"] == 0, res["stdout"][-500:]
        return res

    a = lkh(plain, "plain", None)
    b = lkh(recording, "rec", work / "rec.bin", extra={"CANDIDATE_FILE": work / "cand.txt"})
    assert (work / "cand.txt").exists()
    costs_a, costs_b = B.lkh_trial_costs(a["stdout"]), B.lkh_trial_costs(b["stdout"])
    assert len(costs_a) == 25 and costs_a == costs_b, "the recording copy did not run the same search"
    print("1. same cost at each of", len(costs_a), "trials: ok")

    rec = B.read_records(work / "rec.bin", n)
    trials, best = rec[rec["kind"] == 0], rec[rec["kind"] == 1]
    assert len(trials) == 25 and list(trials["trial"]) == list(range(1, 26))
    for r in rec:
        tour = B.tour_of(r)
        assert B.is_permutation(tour, n) and int(T.tour_length(xy, tour)) == int(r["cost"])
    assert np.all(np.diff(best["cost"]) < 0) and np.all(np.diff(rec["epoch_ms"]) >= 0)
    assert abs(int(rec["epoch_ms"][-1]) / 1000.0 - (b["t_start"] + b["wall"])) < 5.0
    own = T.read_tour(str(work / "rec.tour"))
    assert np.array_equal(B.edge_keys(own, n), B.edge_keys(B.tour_of(best[-1]), n))
    assert int(best[-1]["cost"]) == min(c for _, c in costs_b) == int(T.tour_length(xy, own))
    cut = B.read_records(work / "rec.bin", n)
    with open(work / "cut.bin", "wb") as f:                 # a file whose last record was cut off
        f.write((work / "rec.bin").read_bytes()[:-1000])
    assert len(B.read_records(work / "cut.bin", n)) == len(cut) - 1
    print("2. records:", len(trials), "trial tours and", len(best), "best tours, all valid; last best = LKH's tour: ok")

    cut_ms = int(rec["epoch_ms"][-1])
    tours, costs, info = B.select_population([rec], cut_ms, 12, n)
    assert len(tours) == 12 and costs[0] == int(best[-1]["cost"]) and info[0]["tours_used"] == 12
    assert len({B.edge_keys(t, n).tobytes() for t in tours}) == 12
    only_best, best_costs, _ = B.select_population([rec, rec], cut_ms, 12, n, trial_tours=False)
    assert len(only_best) == 2 and best_costs == [int(best[-1]["cost"])] * 2
    early = int(trials["epoch_ms"][9])
    tours_early, costs_early, info_early = B.select_population([rec], early, 6, n)
    assert info_early[0]["trials_before_cut"] == 10 and max(info_early[0]["trial_range"]) <= 10
    stats = B.population_stats(tours, n)
    assert stats["tours"] == stats["distinct"] == 12 and n <= stats["union_edges"] and stats["common_edges"] <= n
    print("4. population builder: ok", stats)

    B.write_population(work / "pop12", tours, costs)
    results = {}
    for version in ("Normal", "Large"):
        out, best_file = B.run_eax(eax[version], work / "p_eax.tsp", work / ("run_" + version), "T", len(tours),
                                   population_file=work / "pop12", deadline_epoch=time.time() + 60)
        assert out["finished"] and not out["timed_out"], out["tail"]
        tour, value = B.read_eax_best(best_file, n)
        assert B.is_permutation(tour, n) and int(T.tour_length(xy, tour)) == value == out["last_best"]
        assert value <= min(costs) and out["curve"][0][1] == min(costs) and out["curve"][-1][1] == value
        results[version] = (value, out["generations"], out["stopped_by_deadline"])
        ref, ref_file = B.run_eax(plain_eax[version], work / "p_eax.tsp", work / ("ref_" + version), "T", len(tours),
                                  population_file=work / "pop12", deadline_epoch=time.time() + 60)
        ref_tour, ref_value = B.read_eax_best(ref_file, n)
        assert (ref_value, ref["generations"]) == (value, out["generations"]) and np.array_equal(ref_tour, tour), \
            "GA-EAX with the grid search did not give the result of GA-EAX without it"
        assert [c for _, c in ref["curve"]] == [c for _, c in out["curve"]]
    print("3a. GA-EAX from 12 recorded tours:", results, "| best input", min(costs))
    print("5b. the same tour, generation count and improvements with and without the grid search: ok")

    deadline = time.time() + 1.0
    out, best_file = B.run_eax(eax["Normal"], work / "p_eax.tsp", work / "run_deadline", "D", 300,
                               deadline_epoch=deadline)
    tour, value = B.read_eax_best(best_file, n)
    assert out["finished"] and out["stopped_by_deadline"] and B.is_permutation(tour, n)
    assert int(T.tour_length(xy, tour)) == value
    print(f"3b. GA-EAX from scratch with a deadline of 1 s: stopped after {out['wall_s']:.1f} s at generation "
          f"{out['generations']} with a valid tour: ok")
    assert abs(B.sign_test_p(10, 2) - 0.03857421875) < 1e-12 and B.sign_test_p(0, 0) == 1.0

    # 6. run v26: best-only recording, LKH-3's tour merging, helpers
    c = lkh(recording, "best_only", work / "best_only.bin", env={"BETATOUR_BEST_ONLY": "1"})
    only = B.read_records(work / "best_only.bin", n)
    assert B.lkh_trial_costs(c["stdout"]) == costs_a and set(only["kind"].tolist()) == {1}
    assert list(only["cost"]) == list(best["cost"]) and np.array_equal(only["ids"], best["ids"])
    chosen, seen = [], set()
    for r in trials[::-1]:
        key = B.edge_keys(B.tour_of(r), n).tobytes()
        if key not in seen:
            seen.add(key)
            chosen.append((int(r["cost"]), B.tour_of(r)))
        if len(chosen) == 6:
            break
    chosen.sort(key=lambda x: x[0])
    files = []
    for i, (cost, tour) in enumerate(chosen):
        files.append(work / f"m{i}.tour")
        T.write_tour(str(files[-1]), tour, name=f"m{i}", length=cost)
    common = B.common_edge_keys([t for _, t in chosen], n)
    assert B.distinct_tours([t for _, t in chosen] + [chosen[0][1][::-1]], n) == len(chosen)
    expected = set(B.edge_keys(chosen[0][1], n).tolist())
    for _, tour in chosen[1:]:
        expected &= set(B.edge_keys(tour, n).tolist())
    assert sorted(expected) == common.tolist()
    stamp = ((work / "cand.txt").stat().st_mtime_ns, (work / "cand.txt").read_bytes())
    extra = {"CANDIDATE_FILE": work / "cand.txt", "INITIAL_TOUR_FILE": files[0]}
    m1 = lkh(plain, "merge_plain", None, extra=extra, merge=files, trials=30)
    m2 = lkh(recording, "merge_rec", work / "merge.bin", extra=extra, env={"BETATOUR_BEST_ONLY": "1"}, merge=files,
             trials=30)
    assert ((work / "cand.txt").stat().st_mtime_ns, (work / "cand.txt").read_bytes()) == stamp, "candidate file rewritten"
    assert "MERGE_TOUR_FILE = " in (work / "merge_rec.par").read_text()
    assert B.lkh_trial_costs(m1["stdout"]) == B.lkh_trial_costs(m2["stdout"]) and len(B.lkh_trial_costs(m1["stdout"])) == 30
    merged = B.read_records(work / "merge.bin", n)
    final = T.read_tour(str(work / "merge_rec.tour"))
    assert set(merged["kind"].tolist()) == {1} and B.is_permutation(final, n)
    assert int(T.tour_length(xy, final)) == int(merged[-1]["cost"]) <= chosen[0][0]
    assert np.isin(common, B.edge_keys(final, n)).all()
    print(f"6. merging {len(chosen)} trial tours ({len(common)} common edges) with LKH-3: {chosen[0][0]} -> "
          f"{int(merged[-1]['cost'])}; candidate file read, same search with best-only recording, common edges kept: ok")
    got = B.holm([0.01, 0.04, 0.03, 0.002])
    assert np.allclose(got, [0.03, 0.06, 0.06, 0.008]), got
    assert B.holm([0.5, 0.5]) == [1.0, 1.0] and B.holm([]) == []
    stats = B.paired_stats([-1.0, -2.0, 0.0, 1.0], 1)
    assert (stats["win"], stats["tie"], stats["loss"]) == (2, 1, 1)
    print("all tests passed")


if __name__ == "__main__":
    main()
