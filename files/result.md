
Julia
```julia
using DataFrames, FixedEffectModels
N = 10000000
K = 100
df = DataFrame(
  v1 =  pool(rand(@data([NA, 1:(N/K)], N)),
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

