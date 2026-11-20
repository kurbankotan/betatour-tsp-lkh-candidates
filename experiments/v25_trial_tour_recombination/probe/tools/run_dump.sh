#!/bin/bash
# uninterrupted POPMUSIC-5 runs with the recording copy of LKH (every trial's tour is written)
R=$1; CORE=$2; B=$3; shift 3
for s in "$@"; do
  sed -e "s/TIME_LIMIT = 120/TIME_LIMIT = $B/" -e "s#probe/lkh/s$s.tour#bp/s$s.tour#" -e "s/SEED = 101/SEED = $s/" -e "s#probe/lkh/s101.tour#bp/s$s.tour#" $R/probe/lkh/s101.par > $R/bp/s$s.par
  BETATOUR_TRIAL_DUMP=$R/bp/s$s.bin taskset -c $CORE $R/lkh_dump/LKH $R/bp/s$s.par > $R/bp/s$s.log 2>&1 < /dev/null
done
echo done > $R/bp/DONE_$CORE
