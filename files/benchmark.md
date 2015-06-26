# Simple benchmark 
Julia
```julia
using DataFrames, FixedEffectModels
N = 10000000
K = 100
df = DataFrame(
  v1 =  pool(rand(1:(N/K), N)),
  v2 =  pool(rand(1:K, N)),
  v3 =  randn(N), 
  v4 =  randn(N) 
)
@time reg(v4 ~ v3, df)
# elapsed time: 1.22074119 seconds (1061288240 bytes allocated, 22.01% gc time)
@time reg(v4 ~ v3, df, VcovCluster(:v2))
# elapsed time: 1.289171101 seconds (1235060180 bytes allocated, 24.11% gc time)
@time reg(v4 ~ v3 |> v1, df)
# elapsed time: 2.063922289 seconds (1448598696 bytes allocated, 17.96% gc time)
@time reg(v4 ~ v3 |> v1 + v2, df)
# elapsed time: 2.494780022 seconds (1283607248 bytes allocated, 18.87% gc time)
````

R (lfe package)
```R
library(lfe)
N = 10000000
K = 100
df = data.frame(
  v1 =  as.factor(sample(N/K, N, replace = TRUE)),
  v2 =  as.factor(sample(K, N, replace = TRUE)),
  v3 =  runif(N), 
  v4 =  runif(N) 
)
system.time(lm(v4 ~ v3, df))
#   user  system elapsed 
# 15.712   0.811  16.448 
system.time(felm(v4 ~ v3|0|0|v2, df))
#  user  system elapsed 
# 9.160   1.437  10.543 
system.time(felm(v4 ~ v3|v1, df))
#   user  system elapsed 
# 19.971   1.595  22.112 
system.time(felm(v4 ~ v3|(v1 + v2), df))
#   user  system elapsed 
# 23.980   1.950  24.942 
```



Stata
```
clear all
local N = 10000000
local K = 100
set obs `N'
gen  v1 =  floor(runiform() * (`N'+1)/`K')
gen  v2 =  floor(runiform() * (`K'+1))
gen  v3 =  runiform()
gen  v4 =  runiform()

timer clear

timer on 1
reg v4 v3
timer off 1

timer on 2
reg v4 v3, cl(v2)
timer off 2

timer on 3
areg v4 v3, a(v1)
timer off 3

timer on 4
reghdfe v4 v3, a(v1 v2)
timer off 4

. . timer list
   1:      1.61 /        1 =       1.4270
   3:     10.42 /        1 =      10.4190
   4:     15.97 /        1 =      15.9690
   6:     67.24 /        1 =      67.2430
````










## Benchmark with multiple high dimensional fixed effects


[Somaini and Wolak (2014](http://web.stanford.edu/group/fwolak/cgi-bin/sites/default/files/jem-2014-0008.pdf) compare several Stata programs for the case of two high dimensional fixed effects. Below are the results for `reg` (corresponding to Table 1)

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

#> N = 100, T = 100, K = 2 :elapsed time: 1.13054393 seconds (35212076 bytes allocated)
#> N = 100, T = 100, K = 10 :elapsed time: 0.016806448 seconds (5692840 bytes allocated)
#> N = 1000, T = 100, K = 2 :elapsed time: 0.129198173 seconds (20294564 bytes allocated)
#> N = 1000, T = 100, K = 10 :elapsed time: 0.138118011 seconds (52136232 bytes allocated, 54.16% gc time)
#> N = 1000, T = 1000, K = 2 :elapsed time: 0.2525171 seconds (169222688 bytes allocated)
#> N = 1000, T = 1000, K = 10 :elapsed time: 0.883338913 seconds (520312408 bytes allocated, 22.89% gc time)
#> N = 10000, T = 100, K = 2 :elapsed time: 0.247904513 seconds (167789520 bytes allocated)
#> N = 10000, T = 100, K = 10 :elapsed time: 0.929528625 seconds (519347720 bytes allocated, 27.30% gc time)
#> N = 10000, T = 1000, K = 2 :elapsed time: 2.254222754 seconds (1691770400 bytes allocated)
#> N = 10000, T = 1000, K = 10 :elapsed time: 8.02534309 seconds (5203114776 bytes allocated, 13.37% gc time)
#> N = 100000, T = 100, K = 2 :elapsed time: 2.617323992 seconds (1716464492 bytes allocated, 5.90% gc time)
#> N = 100000, T = 100, K = 10 :elapsed time: 7.964445226 seconds (5230600920 bytes allocated, 11.32% gc time)
```


In the case of multiple high dimensional fixed effects, `reg`, `reghdfe` (Stata) and `lfe` roughly use the same repeated demeaning procedure by default. However, when the demean procedure is slow to converge, `reghdfe` and `lfe` switch to different algorithms. If you're working with datasets where the demean procedure is extremly slow, this will make them faster. 
