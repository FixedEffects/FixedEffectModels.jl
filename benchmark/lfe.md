

## Comparison

Julia
```julia
using DataArrays, DataFrames, FixedEffectModels
N = 1000000
K = 10000
df = DataFrame(
  v1 =  PooledDataArray(rand(1:N, N)),
  v2 =  PooledDataArray(rand(1:K, N)),
  v3 =  randn(N), 
  v4 =  randn(N) 
)
@time reg(v4~v3 | v1, df)
# elapsed time: 0.666867215 seconds (269151632 bytes allocated, 23.04% gc time)
@time reg(v4~v3 |(v1+v2), df)
# elapsed time: 1.650180923 seconds (307684448 bytes allocated, 11.86% gc time)
````

R (lfe package, C)
```R
library(lfe)
N = 1000000
K = N/100
df = data_frame(
  v1 =  as.factor(sample(N, N, replace = TRUE)),
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

