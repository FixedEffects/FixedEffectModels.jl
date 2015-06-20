

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
# elapsed time: 0.746867215 seconds (245244128 bytes allocated, 20.77% gc time)
@time reg(v4~v3 |(v1+v2), df)
# elapsed time: 1.909417973 seconds (283775368 bytes allocated, 9.76% gc time)
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

