#!/bin/bash
# GA-EAX on populations of discarded trial tours; one line per population in summary_<core>.txt
SP=$1; R=$2; CORE=$3; shift 3
S=$SP/v20h/pkg/repo/src; I=$R/probe/dev_uni10k.npy; REF=$R/probe/eax/eax_final.tour
cd $R/bp/pops
for spec in "$@"; do
  IFS=: read name T mode K seeds <<< "$spec"
  dumps=""; for s in ${seeds//,/ }; do dumps="$dumps $R/bp/s$s.bin"; done
  python3 $R/tools/bp_pop.py $S $I $REF $name.pop $T $mode $K $dumps > $name.json 2>&1
  N=$(($(wc -l < $name.pop) / 2))
  taskset -c $CORE $R/eax_build/jikken 1 $name $N 30 $R/probe/dev_uni10k_eax.tsp $name.pop > $name.log 2>&1
  echo "$name N=$N $(tail -1 $name.log)" >> summary_$CORE.txt
done
echo done >> summary_$CORE.txt
