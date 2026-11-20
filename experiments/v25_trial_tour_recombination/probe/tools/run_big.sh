#!/bin/bash
# recording LKH runs on the large instance (POPMUSIC-5, uninterrupted)
R=$1; CORE=$2; B=$3; shift 3
for s in "$@"; do
  cat > $R/big/s$s.par <<PAR
PROBLEM_FILE = $R/big/uni50k.tsp
RUNS = 1
TRACE_LEVEL = 1
TIME_LIMIT = $B
MAX_TRIALS = 100000000
SEED = $s
CANDIDATE_SET_TYPE = POPMUSIC
SUBGRADIENT = NO
MAX_CANDIDATES = 5
OUTPUT_TOUR_FILE = $R/big/s$s.tour
PAR
  BETATOUR_TRIAL_DUMP=$R/big/s$s.bin taskset -c $CORE $R/lkh_dump/LKH $R/big/s$s.par > $R/big/s$s.log 2>&1 < /dev/null
done
echo done > $R/big/DONE_$CORE
