#!/bin/bash

# echo "FLOPS_DP"
# (monitoring double precision FLOPS on CPU 0-31)
# JULIA_EXCLUSIVE=1 likwid-perfctr -c 0-31 -g FLOPS_DP -m julia --project=.. -O3 -t 32 cpu_bench.jl 
# JULIA_EXCLUSIVE=1 likwid-perfctr -c 0 -g FLOPS_DP -m julia --project=.. -O3 -t 1 cpu_bench.jl 

echo "MEM_DP"
# (monitoring memory ops on CPU 0-31)

# $ likwid-perfctr -C E:N:36:1:2 -g MEM_DP -m ./a.out
JULIA_EXCLUSIVE=1 likwid-perfctr -C E:N:36 -g MEM_DP -m julia --project=.. -O3 -t 16 cpu_bench.jl 
# JULIA_EXCLUSIVE=1 likwid-perfctr -c 0 -g MEM_DP -m julia --project=.. -O3 -t 1 cpu_bench.jl 