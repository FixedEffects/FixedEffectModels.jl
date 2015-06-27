### Simple benchmark 
![benchmark](https://cdn.rawgit.com/matthieugomez/FixedEffectModels.jl/master/benchmark/result3.svg)

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
  @time reg(y ~ x1 + x2, df)
  # elapsed time: 1.272440851 seconds (1388059428 bytes allocated, 25.67% gc time)
  @time reg(y ~ x1 + x2, df, VcovCluster(:id2))
  # elapsed time: 1.313059285 seconds (1401312428 bytes allocated)
  @time reg(y ~ x1 + x2 |> id1, df)
  # elapsed time: 2.246762796 seconds (1675339864 bytes allocated, 41.52% gc time)
  @time reg(y ~ x1 + x2 |> id1, df, VcovCluster(:id1))
  # elapsed time: 2.874408954 seconds (1690411832 bytes allocated, 33.30% gc time)
  @time reg(y ~ x1 + x2|> id1 + id2, df)
  # elapsed time: 2.750196351 seconds (1689108032 bytes allocated, 22.54% gc time)
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
    x1 =  runif(N),
    x2 =  runif(N)
  )
  system.time(felm(y ~ x1 + x2, df))
  #>    user  system elapsed 
  #>  12.660   1.227  13.779 
  system.time(felm(y ~ x1 + x2|0|0|id2, df))
  #>    user  system elapsed 
  #>  12.530   1.289  13.751 
  system.time(felm(y ~ x1 + x2|id1, df))
  #>    user  system elapsed 
  #>  20.750   1.516  21.847 
  system.time(felm(y ~ x1 + x2|id1|0|id1, df)) 
  #>    user  system elapsed 
  #>  33.163   2.025  34.639
  system.time(felm(y ~ x1 + x2|(id1 + id2), df))
  #>    user  system elapsed 
  #>  26.603   1.954  26.614
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

  set rmsg on
  reg y x1 x2
  #> r; t=1.20 12:32:46
  reg y x1 x2, cl(id2)
  #> r; t=11.15 12:32:57
  areg y x1 x2, a(id1)
  #>r; t=15.51 12:33:13
  areg y x1 x2, a(id1) cl(id1)
  #> r; t=53.02 12:34:06
  reghdfe y x1 x2, a(id1 id2) fast keepsingletons
  #> r; t=100.50 12:35:47
  ````






Note: `reg`, `reghdfe` (Stata) and `lfe` (R) roughly use the same repeated demeaning procedure by default. When the demean procedure is slow to converge, `reghdfe` and `lfe` switch to different algorithms. For some "hard" datasets, these commands may therefore be faster ( = `reg` is fast because Julia allows to write faster code, not because the underlying algorithm is better).

