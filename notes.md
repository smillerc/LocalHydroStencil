benchmarking for 3stage RK3
with SArrays in the mesh for facelen and norms: 
nthreads: 6
BenchmarkTools.Trial: 24 samples with 1 evaluation.
 Range (min … max):  205.900 ms … 213.401 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     208.876 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   209.125 ms ±   2.031 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▁     ▁▁▁▁▁  ▁▁ ▁ ▁▁▁      ▁ ▁ ▁  █▁   ▁ ▁    █       ▁     ▁  
  █▁▁▁▁▁█████▁▁██▁█▁███▁▁▁▁▁▁█▁█▁█▁▁██▁▁▁█▁█▁▁▁▁█▁▁▁▁▁▁▁█▁▁▁▁▁█ ▁
  206 ms           Histogram: frequency by time          213 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.

with normal Arrays in the mesh for facelen and norms: median 211 ms
nthreads: 6
BenchmarkTools.Trial: 24 samples with 1 evaluation.
 Range (min … max):  208.667 ms … 221.145 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     211.312 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   212.574 ms ±   3.738 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▃ ▃         ▃ ▃                           █                    
  █▇█▇▇▁▇▁▇▁▇▁█▇█▁▇▁▁▇▁▁▁▁▁▁▁▁▇▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▇▇▁▁▁▁▁▁▁▁▁▁▁▇ ▁
  209 ms           Histogram: frequency by time          221 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.