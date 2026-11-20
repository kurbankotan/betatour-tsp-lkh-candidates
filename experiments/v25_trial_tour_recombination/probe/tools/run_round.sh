#!/bin/bash
# continue LKH from a merged tour for B seconds with the recording copy (own POPMUSIC-5 lists per seed)
R=$1; CORE=$2; B=$3; DIR=$4; INIT=$5; shift 5
for s in "$@"; do
  cat > $DIR/s$s.par <<PAR
PROBLEM_FILE = $R/probe/dev_uni10k.tsp
RUNS = 1
TRACE_LEVEL = 1
TIME_LIMIT = $B
MAX_TRIALS = 100000000
SEED = $s
CANDIDATE_SET_TYPE = POPMUSIC
SUBGRADIENT = NO
MAX_CANDIDATES = 5
INITIAL_TOUR_FILE = $INIT
OUTPUT_TOUR_FILE = $DIR/s$s.tour
PAR
  BETATOUR_TRIAL_DUMP=$DIR/s$s.bin taskset -c $CORE $R/lkh_dump/LKH $DIR/s$s.par > $DIR/s$s.log 2>&1 < /dev/null
done
echo done > $DIR/DONE_$CORE
