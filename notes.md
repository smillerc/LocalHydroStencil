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


 # Using the current version as-is (commit fe30017)
  - `check_uniform=false`
  - `dx = 0.001`
 ```
 BenchmarkTools.Trial: 97 samples with 1 evaluation.
 Range (min … max):  49.461 ms … 61.341 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     51.281 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   51.814 ms ±  2.096 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

     ▄█▄ ▅                                                     
  ▅▅██████▃▆▆█████▆▆▁▃▁▅▁▅▅▃▁▃▁▁▅▃▅▃▁▁▅▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃ ▁
  49.5 ms         Histogram: frequency by time        59.9 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
 ```



`nzones=640000`
`nthreads=6`
 ```
 Benchmarking with muscl orig

BenchmarkTools.Trial: 97 samples with 1 evaluation.
 Range (min … max):  50.815 ms … 55.168 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     51.672 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   51.981 ms ±  1.039 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

      ▆█ ▁                                                     
  ▄▆▃▁██▆█▃█▆█▇▆▆▆█▄▃▄▁▆▃▁▃▁▃▃▁▆▁▁▄▁▃▄▁▁▁▁▁▁▃▁▁▁▁▄▁▁▁▁▃▁▁▁▁▃▄ ▁
  50.8 ms         Histogram: frequency by time        55.2 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.

Benchmarking with muscl_sarr_turbo_split2

BenchmarkTools.Trial: 118 samples with 1 evaluation.
 Range (min … max):  40.980 ms … 52.887 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     41.776 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   42.355 ms ±  1.560 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

     ▁▃█ ▂▁▆                                                   
  ▇▆▃███▆███▇██▄▄▆▁▄▆▄▄▄▁▆▄▁▁▁▁▁▃▁▁▁▃▁▃▃▁▃▃▁▁▃▃▁▁▃▄▄▄▃▃▃▁▁▃▁▃ ▃
  41 ms           Histogram: frequency by time        45.7 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
 ```
 `speedup=1.23x`


`nzones=640000`
`nthreads=12`
 ```
 Benchmarking with muscl orig

BenchmarkTools.Trial: 171 samples with 1 evaluation.
 Range (min … max):  27.394 ms … 39.236 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     28.951 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   29.301 ms ±  1.884 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

    ▁   ▁▃▄█▁▃▁                                                
  ▇▅██▇████████▄▁▃▄▁▄▃▁▁▃▁▁▃▁▁▃▁▁▁▃▃▃▃▁▁▁▁▃▃▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▃ ▃
  27.4 ms         Histogram: frequency by time        37.7 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.

Benchmarking with muscl_sarr_turbo_split2

BenchmarkTools.Trial: 211 samples with 1 evaluation.
 Range (min … max):  21.965 ms … 30.001 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     23.268 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   23.681 ms ±  1.367 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

     ▁  ▂▁ ▃█▄▆   ▂                                            
  ▆▇▇█▆█████████▇▆█▇▄▆▁▆▁▁▁▄▁▁▁▁▆▄▄▁▇▁▄▁▄▄▄▆▄▁▁▄▁▆▄▄▁▁▁▁▁▁▁▄▄ ▆
  22 ms        Histogram: log(frequency) by time      29.1 ms <

 Memory estimate: 0 bytes, allocs estimate: 0.
 ```
  `speedup=1.24x`