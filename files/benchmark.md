# Simple benchmark 
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





## Benchmark with multiple high dimensional fixed effects


- [Somaini and Wolak (2014](http://web.stanford.edu/group/fwolak/cgi-bin/sites/default/files/jem-2014-0008.pdf) compare several Stata programs for the case of two high dimensional fixed effects. Below are the results for `reg` (corresponding to Table 1)

  ```julia
  using DataFrames, Distributions, FixedEffectModels
  N = [100, 1000, 1000, 10_000, 10_000, 100_000]
  T = [100, 100, 1000, 100, 1000, 100]
  for i in 1:length(N)
    df = DataFrame(
      T = pool(repmat([1:T[i]], N[i], 1)[:]),
      N = pool(div([1:N[i]*T[i]], T[i])),
      y =  rand(Normal(), N[i]*T[i]), 
      v1 =  rand(Normal(), N[i]*T[i]), 
      v2 =  rand(Normal(), N[i]*T[i]), 
      v3 =  rand(Normal(), N[i]*T[i]), 
      v4 =  rand(Normal(), N[i]*T[i]), 
      v5 =  rand(Normal(), N[i]*T[i]), 
      v6 =  rand(Normal(), N[i]*T[i]), 
      v7 =  rand(Normal(), N[i]*T[i]), 
      v8 =  rand(Normal(), N[i]*T[i]), 
      v9 =  rand(Normal(), N[i]*T[i]), 
      v10 =  rand(Normal(), N[i]*T[i]), 
      sample = rand(1:10, N[i]*T[i])
    )
    df = df[df[:sample] .< 10, :]
    print("N = $(N[i]), T = $(T[i]), K = 2 :")
    @time reg(y ~ v1 + v2 |> N + T, df)
    print("N = $(N[i]), T = $(T[i]), K = 10 :")
    @time reg(y ~ v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10|> N + T, df)
  end
  #> N = 100, T = 100, K = 2 :elapsed time: 0.00177218 seconds (1593896 bytes allocated)
  #> N = 100, T = 100, K = 10 :elapsed time: 0.016803317 seconds (5487008 bytes allocated)
  #> N = 1000, T = 100, K = 2 :elapsed time: 0.129419728 seconds (18553744 bytes allocated)
  #> N = 1000, T = 100, K = 10 :elapsed time: 0.057043524 seconds (50258432 bytes allocated)
  #> N = 1000, T = 1000, K = 2 :elapsed time: 0.465490222 seconds (150845032 bytes allocated, 59.77% gc time)
  #> N = 1000, T = 1000, K = 10 :elapsed time: 0.781285787 seconds (500965808 bytes allocated, 27.22% gc time)
  #> N = 10000, T = 100, K = 2 :elapsed time: 0.222820456 seconds (150383160 bytes allocated, 19.82% gc time)
  #> N = 10000, T = 100, K = 10 :elapsed time: 0.722421988 seconds (501285120 bytes allocated, 20.00% gc time)
  #> N = 10000, T = 1000, K = 2 :elapsed time: 2.055887877 seconds (1507874760 bytes allocated, 8.26% gc time)
  #> N = 10000, T = 1000, K = 10 :elapsed time: 6.864606832 seconds (5009281936 bytes allocated, 10.81% gc time)
  #> N = 100000, T = 100, K = 2 :elapsed time: 2.038136114 seconds (1520761528 bytes allocated, 8.28% gc time)
  #> N = 100000, T = 100, K = 10 :elapsed time: 6.834395734 seconds (5028378640 bytes allocated, 11.08% gc time)
  ```
-  `reg`, `reghdfe` (Stata) and `lfe`  use the same repeated demeaning procedure by default. However, when the demean procedure is slow to converge, `reghdfe` and `lfe` switch to different algorithms. If you're working with datasets where the demean procedure is extremly slow, this may make them faster ( = `reg` is fast because Julia allows to write fast code, not because it uses a superior algorithm).



