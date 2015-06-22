

## Comparison

Julia
```julia
using DataArrays, DataFrames, FixedEffectModels
N = 10000000
K = 100
df = DataFrame(
  v1 =  PooledDataArray(rand(1:N/K, N)),
  v2 =  PooledDataArray(rand(1:K, N)),
  v3 =  randn(N), 
  v4 =  randn(N) 
)
@time reg(v4~v3 | v1, df)
elapsed time: 0.485013326 seconds (189237352 bytes allocated, 15.80% gc time)
@time reg(v4~v3 |(v1+v2), df)
# elapsed time: 1.398623511 seconds (194714280 bytes allocated, 8.86% gc time)
````

R (lfe package, C)
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
system.time(felm(v4~v3|v1, df))
#  user  system elapsed 
# 5.025   0.158   5.166 
system.time(felm(v4~v3|v1+v2, df))
#  user  system elapsed 
# 5.817   0.211   5.426 
```

