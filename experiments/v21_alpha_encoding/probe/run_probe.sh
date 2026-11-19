#!/bin/bash
# The probe as it was run in a two-CPU sandbox (LKH-3.0.14 built from the pinned archive): two LKH processes at a time.
#   bash run_probe.sh /path/to/LKH
# Instances s101 to s109 have six arms (run as three concurrent pairs, the pairing rotated by instance) and the
# two-phase schedule; instances s110 to s118 have the arms uninterrupted, native and rank (three concurrent processes
# on two CPUs). Instance seed: 7 for s101, 7 * s + 1 for the others; LKH seed: s.
set -e
LKH=$1
HERE=$(cd "$(dirname "$0")" && pwd)
python3 $HERE/probe_setup.py $LKH 10000 7 101 s101 > s101.setup.out 2>&1
python3 $HERE/probe_arms.py $LKH s101 101 60 native rank >> s101.setup.out 2>&1
python3 $HERE/probe_arms.py $LKH s101 101 60 top2 zero >> s101.setup.out 2>&1
python3 $HERE/probe_arms.py $LKH s101 101 60 uninterrupted none >> s101.setup.out 2>&1
i=0
for s in 102 103 104 105 106 107 108 109; do
  python3 $HERE/probe_setup.py $LKH 10000 $((s*7+1)) $s s$s > s$s.setup.out 2>&1
  case $((i % 3)) in
    0) P1="uninterrupted rank"; P2="native none"; P3="top2 zero";;
    1) P1="uninterrupted native"; P2="rank top2"; P3="none zero";;
    2) P1="uninterrupted none"; P2="rank zero"; P3="native top2";;
  esac
  python3 $HERE/probe_arms.py $LKH s$s $s 60 $P1 >> s$s.setup.out 2>&1
  python3 $HERE/probe_arms.py $LKH s$s $s 60 $P2 >> s$s.setup.out 2>&1
  python3 $HERE/probe_arms.py $LKH s$s $s 60 $P3 >> s$s.setup.out 2>&1
  i=$((i+1))
done
for s in 101 102 103 104 105 106 107 108 109; do
  python3 $HERE/probe_schedule.py $LKH s$s $s 30 29 zero none
done
for s in 110 111 112 113 114 115 116 117 118; do
  python3 $HERE/probe_setup.py $LKH 10000 $((s*7+1)) $s s$s > s$s.setup.out 2>&1
  python3 $HERE/probe_arms.py $LKH s$s $s 60 uninterrupted native rank >> s$s.setup.out 2>&1
done
python3 $HERE/probe_summary.py . .
