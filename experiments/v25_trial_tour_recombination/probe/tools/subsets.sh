#!/bin/bash
# GA-EAX started from the first N LKH first-trial tours (N = population size)
SP=$1; R=$2; shift 2
mkdir -p $R/probe/subsets; cd $R/probe/subsets
for N in "$@"; do
  head -n $((2 * N)) $R/probe/seeded/POP > POP_$N
  s=$(date +%s.%N)
  taskset -c 0 $R/eax_build/jikken 1 SUB$N $N 30 $R/probe/dev_uni10k_eax.tsp POP_$N > sub$N.log 2>&1
  e=$(date +%s.%N)
  echo "N=$N wall=$(echo "$e - $s" | bc) $(tail -1 sub$N.log)" >> summary.txt
done
echo done >> summary.txt
