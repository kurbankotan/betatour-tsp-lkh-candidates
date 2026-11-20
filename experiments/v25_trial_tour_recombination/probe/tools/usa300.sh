#!/bin/bash
R=$1
while [ ! -f $R/usa/pop_b/POP ] || [ ! -f $R/usa/pop_c/POP ]; do sleep 2; done
mkdir -p $R/usa/eax300 && cd $R/usa/eax300 && cat $R/usa/pop/POP $R/usa/pop_b/POP $R/usa/pop_c/POP > POP300
taskset -c 1 $R/eax_build/jikken 1 U300 300 30 $R/usa/usa13509.tsp POP300 > eax.log 2>&1
echo done > ALL_DONE
