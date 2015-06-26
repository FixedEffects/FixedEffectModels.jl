### Simple benchmark 
![benchmark](https://cdn.rawgit.com/matthieugomez/FixedEffectModels.jl/master/files/result2.svg)

Code to reproduce this graph:

  Julia
  ```julia
  using DataFrames, FixedEffectModels
  N = 10000000
  K = 100
  df = DataFrame(
    id1 =  pool(rand(1:(N/K), N)),
    id2 =  pool(rand(1:K, N)),
     y =  randn(N),
    x1 =  randn(N),
    x2 =  randn(N)
  )
  @time reg(y ~ x1, df)
  # elapsed time: 0.819534136 seconds (1061291480 bytes allocated, 22.60% gc time)
  @time reg(y ~ x1, df, VcovCluster(:id2))
  # elapsed time: 1.024682163 seconds (1075061296 bytes allocated, 26.10% gc time)
  @time reg(y ~ x1 |> id1, df)
  # elapsed time: 1.133261605 seconds (1188282120 bytes allocated, 22.28% gc time)
  @time reg(y ~ x1|> id1 + id2, df)
  #elapsed time: 1.707240818 seconds (1202049072 bytes allocated, 13.95% gc time)
  ````

  R (lfe package)
  ```R
  library(lfe)
  N = 10000000
  K = 100
  df = data.frame(
    id1 =  as.factor(sample(N/K, N, replace = TRUE)),
    id2 =  as.factor(sample(K, N, replace = TRUE)),
    y =  runif(N),
    x1 =  runif(N)
  )
  system.time(lm(y ~ x1, df))
  # user  system elapsed 
  # 11.970   0.746  12.635 
  system.time(felm(y ~ x1|0|0|id2, df))
  #  user  system elapsed 
  # 9.160   1.437  10.543 
  system.time(felm(y ~ x1|id1, df))
  #   user  system elapsed 
  # 19.971   1.595  22.112 
  system.time(felm(y ~ x1|(id1 + id2), df))
  #   user  system elapsed 
  # 23.980   1.950  24.942 
  ```



  Stata
  ```
  clear all
  local N = 10000000
  local K = 100
  set obs `N'
  gen  id1 =  floor(runiform() * (`N'+1)/`K')
  gen  id2 =  floor(runiform() * (`K'+1))
  gen   y =  runiform()
  gen   x1 =  runiform()
  gen   x2 =  runiform()
  timer clear

  timer on 1
  reg y x1 x2
  timer off 1

  timer on 2
  reg y x1 x2, cl(id2)
  timer off 2

  timer on 3
  areg y x1 x2, a(id1)
  timer off 3

  timer on 4
  reghdfe y x1 x2, a(id1 id2)
  timer off 4

  . . timer list
     1:      1.61 /        1 =       1.4270
     3:     10.42 /        1 =      10.4190
     4:     15.97 /        1 =      15.9690
     6:     67.24 /        1 =      67.2430
  ````






`reg` is fast because Julia allows to write fast code, not because of a superior algorithm. `reg`, `reghdfe` (Stata) and `lfe`  roughly use the same repeated demeaning procedure by default. When the demean procedure is slow to converge, however, `reghdfe` and `lfe` switch to different algorithms. For some "hard" datasets, these commands may therefore be faster.
