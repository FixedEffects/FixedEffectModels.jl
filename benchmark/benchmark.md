### Simple benchmark 
![benchmark](https://cdn.rawgit.com/matthieugomez/FixedEffectModels.jl/4c7d1db39377f1ee649624c909c9017f92484114/benchmark/result.svg)

Code to reproduce this graph:

  Julia
  ```julia
  using DataFrames, FixedEffectModels
  N = 10000000
  K = 100
  id1 = rand(1:(N/K), N)
  id2 = rand(1:K, N)
  x1 =  randn(N)
  x2 =  randn(N)
  w = cos(id1)
  y= 3 .* x1 .+ 5 .* x2 .+ cos(id1) .+ cos(id2).^2 .+ randn(N)
  df = DataFrame(id1 = pool(id1), id2 = pool(id2), x1 = x1, x2 = x2, w = w, y = y)
  @time reg(y ~ x1 + x2, df)
  # 1.258554 seconds (723 allocations: 1.205 GB, 18.70% gc time)
  @time reg(y ~ x1 + x2, df, VcovCluster(:id2))
  #  1.569679 seconds (843 allocations: 1.293 GB, 25.21% gc time)
  @time reg(y ~ x1 + x2 |> id1, df)
  # 1.476390 seconds (890 allocations: 1.175 GB, 20.15% gc time)
  @time reg(y ~ x1 + x2 |> id1, df, VcovCluster(:id1))
  # 1.974738 seconds (1.04 k allocations: 1.255 GB, 15.85% gc time)
  @time reg(y ~ x1 + x2 |> id1 + id2, df)
  # 4.554836 seconds (1.01 k allocations: 1.188 GB, 9.56% gc time)
  @time reg(y ~ x1 + x2 |> id1, df, weight = :w)
  # 2.000850 seconds (20.00 M allocations: 1.010 GB, 18.82% gc time)
  @time reg(y ~ x1 + x2 |> id1 + id2, df, weight = :w)
  # 4.118974 seconds (20.00 M allocations: 1.018 GB, 9.60% gc time)
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




Note: `reg`, `reghdfe` (Stata) and `lfe` (R) roughly use the same procedures. For some "hard" datasets, accelerations implemented in `reghdfe`may make the command faster ( = `reg` is fast because Julia allows to write faster code, not because the underlying algorithm is better).
