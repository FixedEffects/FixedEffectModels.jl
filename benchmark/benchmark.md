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
  w = cos.(id1)
  y= 3 .* x1 .+ 2 .* x2 .+ sin.(id1) .+ cos.(id2).^2 .+ randn(N)
  df = DataFrame(id1 = categorical(id1), id2 = categorical(id2), x1 = x1, x2 = x2, w = w, y = y)
  @time reg(df, @model(y ~ x1 + x2))
  #0.601445 seconds (1.05 k allocations: 535.311 MiB, 31.95% gc time)
  @time reg(df, @model(y ~ x1 + x2, vcov = cluster(id2)))
  #  1.213357 seconds (2.01 k allocations: 878.712 MiB, 16.65% gc time)
  @time reg(df, @model(y ~ x1 + x2, fe = id1))
  # 1.476390 seconds (890 allocations: 1.175 GB, 20.15% gc time)
  @time reg(df, @model(y ~ x1 + x2, fe = id1, vcov = cluster(id1)))
  # 2.448953 seconds (500.21 k allocations: 1.052 GiB, 17.36% gc time)
  @time reg(df, @model(y ~ x1 + x2, fe = id1 + id2))
  # 3.639817 seconds (1.84 k allocations: 999.675 MiB, 11.25% gc time)
  @
  ````

  Additionally, `FixedEffectModels` can use a sparse matrix factorization
  ```julia
  @time reg(y ~ x1 + x2 |> id1 + id2, df, method = :cholfact)
  # 21.603901 seconds (200.51 M allocations: 7.460 GB, 6.16% gc time)
  @time reg(y ~ x1 + x2 |> id1 + id2, df, method = :qrfact)
  # 120.274748 seconds (199.94 M allocations: 23.713 GB, 1.35% gc time)
  ```

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
  #>    3.529   0.597   4.144 
  system.time(felm(y ~ x1 + x2|0|0|id2, df))
  #> user  system elapsed 
  #> 3.529   0.597   4.144 
  system.time(felm(y ~ x1 + x2|id1, df))
  #>    user  system elapsed 
  #> 15.507   0.980  16.462 
  system.time(felm(y ~ x1 + x2|id1|0|id1, df)) 
  #>  user  system elapsed 
  #> 21.197   1.163  22.327 
  system.time(felm(y ~ x1 + x2|(id1 + id2), df))
  #>  user  system elapsed 
  #> 11.144   1.321  11.031
  system.time(getfe(felm(y ~ x1 + x2|(id1 + id2), df)))
  #>  user  system elapsed 
  #>  14.330   1.444  14.202 
  ```

  Stata
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
  reg y x1 x2, cl(id2)
  #> r; t=11.15 12:32:57
  areg y x1 x2, a(id1)
  #>r; t=15.51
  areg y x1 x2, a(id1) cl(id1)
  #> r; t=53.02
  reghdfe y x1 x2, a(id1 id2) fast
  #> r; t=88.98 
  reghdfe y x1 x2, a(fe1 = id1 fe2 = id2)
  #> r; t=108.47 12:35:47
  ````




Note: `reg` is fast mainly because Julia allows to write faster code.  On these examples,  `reg`, `reghdfe` and `lfe` require the same number of iterations to converge.
