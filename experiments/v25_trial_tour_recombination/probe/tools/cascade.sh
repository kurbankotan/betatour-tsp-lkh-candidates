#!/bin/bash
# LKH first-trial tours (one core) -> GA-EAX started from them
SP=$1; R=$2; D=$3; TSP=$4; NPY=$5; N=$6; CORE=$7
python3 $R/tools/gen_pop.py $SP/v20h/pkg/repo/src $R/tools/bin/LKH $TSP $NPY $D/pop $N $CORE > $D/gen.log 2>&1
cd $D/pop && taskset -c $CORE $R/eax_build/jikken 1 CASC $N 30 $TSP POP > eax.log 2>&1
echo done > $D/ALL_DONE
