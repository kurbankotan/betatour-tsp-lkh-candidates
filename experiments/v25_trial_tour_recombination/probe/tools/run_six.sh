#!/bin/bash
# six uninterrupted POPMUSIC-5 runs, one after another on one core (settings of the notebooks' pop_full arm)
R=$1; LKH=$2; B=$3
for s in 101 102 103 104 105 106; do
  cat > $R/probe/lkh/s$s.par <<PAR
PROBLEM_FILE = $R/probe/dev_uni10k.tsp
RUNS = 1
TRACE_LEVEL = 1
TIME_LIMIT = $B
MAX_TRIALS = 100000000
SEED = $s
CANDIDATE_SET_TYPE = POPMUSIC
SUBGRADIENT = NO
MAX_CANDIDATES = 5
OUTPUT_TOUR_FILE = $R/probe/lkh/s$s.tour
PAR
  taskset -c 1 $LKH $R/probe/lkh/s$s.par > $R/probe/lkh/s$s.log 2>&1 < /dev/null
done
echo done > $R/probe/lkh/ALL_DONE
