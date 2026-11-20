#!/bin/bash
# LKH first-trial tours as the initial population of GA-EAX (one core)
SP=$1; R=$2; N=$3
python3 $R/tools/gen_pop.py $SP/v20h/pkg/repo/src $R/tools/bin/LKH $R/probe/dev.tsp $R/probe/dev_uni10k.npy $R/probe/seeded $N 1 > $R/probe/seeded_gen.log 2>&1
cd $R/probe/seeded && taskset -c 1 timeout 900 $R/eax_build/jikken 1 SEEDED $N 30 $R/probe/dev_uni10k_eax.tsp POP > eax_seeded.log 2>&1
echo done > $R/probe/seeded/ALL_DONE
