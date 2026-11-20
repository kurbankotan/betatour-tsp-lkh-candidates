#!/bin/bash
# robustness: the same cascade on the uniform instance with the other 100 first-trial tours (tours 101-200)
R=$1
while [ ! -f $R/usa/pop_c/POP ]; do sleep 2; done
mkdir -p $R/probe/second && cd $R/probe/second && tail -n 200 $R/probe/seeded/POP > POP_B100
taskset -c 0 $R/eax_build/jikken 1 SECOND 100 30 $R/probe/dev_uni10k_eax.tsp POP_B100 > eax.log 2>&1
echo done > ALL_DONE
