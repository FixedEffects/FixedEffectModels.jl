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
  y= 3 .* x1 .+ 2 .* x2 .+ sin.(id1) .+ cos.(id2).^2 .+ randn(N)
  df = DataFrame(id1 = categorical(id1), id2 = categorical(id2), x1 = x1, x2 = x2, w = w, y = y)
  @time reg(df, @formula(y ~ x1 + x2))
  #0.601445 seconds (1.05 k allocations: 535.311 MiB, 31.95% gc time)
  @time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
  #  1.624446 seconds (1.21 k allocations: 734.353 MiB, 17.27% gc time)
  @time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
  # 3.639817 seconds (1.84 k allocations: 999.675 MiB, 11.25% gc time)
  @time reg(df, @formula(y ~ x1 + x2), Vcov.cluster(:id1))
  # 1.462648 seconds (499.30 k allocations: 690.102 MiB, 15.92% gc time)
  @time reg(df, @formula(y ~ x1 + x2, Vcov.cluster(:id1, :id2)))
  #  7.187382 seconds (7.02 M allocations: 2.753 GiB, 24.19% gc time)
  ````


  R (lfe package)
  ```R
  library(lfe)
  N = 10000000
  K = 100
  df = data.frame(
    id1 =  as.factor(sample(N/K, N, replace = TRUE)),
    id2 =  as.factor(sample(K, N, replace = TRUE)),
    x1 =  runif(N),
    x2 =  runif(N)
  )
  df[, "y"] =  3 * df[, "x1"] + 2 * df[, "x2"] + sin(as.numeric(df[, "id1"])) + cos(as.numeric(df[, "id2"])) + runif(N)

  system.time(felm(y ~ x1 + x2, df))
  #>   user  system elapsed
  #>   1.843   0.476   2.323 
  system.time(felm(y ~ x1 + x2|id1, df))
  #>    user  system elapsed 
  #> 14.831   1.342  15.993 
  system.time(felm(y ~ x1 + x2|id1 + id2, df))
  #>  user  system elapsed 
  #>  10.626   1.358  10.336
  system.time(felm(y ~ x1 + x2|0|0|id1, df))
  #> user  system elapsed 
  #> 9.255   0.843  10.110
  system.time(felm(y ~ x1 + x2|0|0|id1 + id2, df)) 
  #>  user  system elapsed 
  #> 96.958   1.474  99.113 
  ```

  Stata (reghdfe  version 5.2.9 06aug2018)
  ```
  clear all
  local N = 10000000
  local K = 100
  set obs `N'
  gen  id1 =  floor(runiform() * (`N'+1)/`K')
  gen  id2 =  floor(runiform() * (`K'+1))
  gen   x1 =  runiform()
  gen   x2 =  runiform()
  gen   y =  3 * x1 + 2 * x2 + sin(id1) + cos(id2) + runiform()
  timer clear

  set rmsg on
  reg y x1 x2
  #> r; t=1.20
  areg y x1 x2, a(id1)
  #>r; t=15.51
  reghdfe y x1 x2, a(id1 id2)
  #> r; t=49.38
  reg y x1 x2, cl(id1)
  #> r; t=11.15
  ivreg2 y x1 x2, cluster(id1 id2)
  #> r; t=118.67 
  ````
