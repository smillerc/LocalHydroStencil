#!/bin/bash
set -e
set -u
# nt=6/
# for nt in {1,2,4,6,8,12,16}; do
for nt in {12,16}; do
  echo "-----"
  JULIA_EXCLUSIVE=1 julia --project=.. -t $nt -O2 cpu_bench.jl # > $nt.txt
  echo "-----"
done