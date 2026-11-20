#!/bin/bash
SP=$1; R=$2; CORE=$3; shift 3
S=$SP/v20h/pkg/repo/src; I=$R/big/uni50k.npy
mkdir -p $R/big/pops; cd $R/big/pops
for spec in "$@"; do
  IFS=: read name T mode K <<< "$spec"
  dumps=""; for s in 101 102 103 104 105 106; do dumps="$dumps $R/big/s$s.bin"; done
  python3 $R/tools/bp_pop.py $S $I - $name.pop $T $mode $K $dumps > $name.json 2>&1
  N=$(($(wc -l < $name.pop) / 2))
  taskset -c $CORE $R/eax_build_large/jikken 1 $name $N 30 $R/big/uni50k_eax.tsp $name.pop > $name.log 2>&1
  echo "$name N=$N $(tail -1 $name.log)" >> summary_$CORE.txt
done
echo done >> summary_$CORE.txt
