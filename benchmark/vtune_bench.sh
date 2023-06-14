#!/bin/bash

source /opt/intel/oneapi/vtune/latest/vtune-vars.sh intel64

export JULIA_EXCLUSIVE=1
ENABLE_JITPROFILING=1 INTEL_JIT_BACKWARD_COMPATIBILITY=1 vtune -collect hotspots -start-paused -- julia --project=.. -O3 -t 16 cpu_bench.jl 