### Simple benchmark 
![benchmark](https://cdn.rawgit.com/matthieugomez/FixedEffectModels.jl/4c7d1db39377f1ee649624c909c9017f92484114/benchmark/result.svg)

Code to reproduce this graph:

  Julia 1.9 (FixedEffectModels.jl, v1.8.2)
  ```julia
  using DataFrames, CategoricalArrays, FixedEffectModels
  N = 10000000
  K = 100
  id1 = rand(1:(N/K), N)
  id2 = rand(1:K, N)
  x1 =  randn(N)
  x2 =  randn(N)
  y= 3 .* x1 .+ 2 .* x2 .+ sin.(id1) .+ cos.(id2).^2 .+ randn(N)
  df = DataFrame(id1 = categorical(id1), id2 = categorical(id2), x1 = x1, x2 = x2, y = y)
  @time reg(df, @formula(y ~ x1 + x2))
  # 0.357360 seconds (450 allocations: 691.441 MiB, 4.09% gc time)
  @time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
  #  0.463058 seconds (1.00 k allocations: 929.129 MiB, 13.31% gc time)
  @time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
  #  1.006031 seconds (3.22 k allocations: 1.057 GiB, 1.68% gc time)
  @time reg(df, @formula(y ~ x1 + x2), Vcov.cluster(:id1))
  # 0.380562 seconds (580 allocations: 771.606 MiB, 3.07% gc time)
  @time reg(df, @formula(y ~ x1 + x2), Vcov.cluster(:id1, :id2))
  #0.765847 seconds (719 allocations: 1.128 GiB, 2.01% gc time)
  ````


  R 4.2.2 (fixest, 0.8.4)
  ```R
  library(fixest)
  N = 10000000
  K = 100
  df = data.frame(
    id1 =  as.factor(sample(N/K, N, replace = TRUE)),
    id2 =  as.factor(sample(K, N, replace = TRUE)),
    x1 =  runif(N),
    x2 =  runif(N)
  )
  df[, "y"] =  3 * df[, "x1"] + 2 * df[, "x2"] + sin(as.numeric(df[, "id1"])) + cos(as.numeric(df[, "id2"])) + runif(N)

  system.time(feols(y ~ x1 + x2, df))
  #>      user  system elapsed 
  #>     0.280   0.036   0.317 
  system.time(feols(y ~ x1 + x2|id1, df))
  #>    user  system elapsed 
  #> 0.616   0.089   0.704 
  system.time(feols(y ~ x1 + x2|id1 + id2, df))
  #>  user  system elapsed 
  #>   1.181   0.120   1.297 
  system.time(feols(y ~ x1 + x2, cluster = "id1", df))
  #> user  system elapsed 
  #>  0.630   0.071   0.700 
  system.time(feols(y ~ x1 + x2, cluster = c("id1", "id2"), df)) 
  #>  user  system elapsed 
  #> 1.570   0.197   1.803 
  ```


  R 4.2.2 (lfe, 2.8-8)
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
  #>   1.137   0.232   1.596 
  system.time(felm(y ~ x1 + x2|id1, df))
  #>    user  system elapsed 
  #>    7.08    0.41    7.46 
  system.time(felm(y ~ x1 + x2|id1 + id2, df))
  #>  user  system elapsed 
  #>  4.832   0.370   4.615 
  system.time(felm(y ~ x1 + x2|0|0|id1, df))
  #> user  system elapsed 
  #>  3.712   0.287   3.996 
  system.time(felm(y ~ x1 + x2|0|0|id1 + id2, df)) 
  #>  user  system elapsed 
  #> 59.119   0.889  59.946 


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
