#!/bin/bash

# source /opt/intel/oneapi/vtune/latest/vtune-vars.sh intel64
source /opt/intel/oneapi/advisor/latest/advixe-vars.sh

JULIA_EXCLUSIVE=1 ENABLE_JITPROFILING=1 INTEL_JIT_BACKWARD_COMPATIBILITY=1 advixe-cl \
    --no-auto-finalize \
    -collect roofline \
    -project-dir=./advi3 --search-dir src:r=../src -- julia --project=.. -g1 -t 16 cpu_bench.jl 

