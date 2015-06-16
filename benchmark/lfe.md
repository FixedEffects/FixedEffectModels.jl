

## Comparison
Julia
```julia
using DataArrays, DataFrames
N = 1000000
K = 10000
df = DataFrame(
  v1 =  PooledDataArray(rand(1:N, N)),
  v2 =  PooledDataArray(rand(1:K, N)),
  v3 =  randn(N), 
  v4 =  randn(N) 
)
@time areg(v4~v3, df, nothing ~ v1)
# elapsed time: 0.401270786 seconds (226271604 bytes allocated, 30.61% gc time)
@time areg(v4~v3, df, nothing ~ v1+v2)
# elapsed time: 1.220437439 seconds (243547784 bytes allocated, 11.34% gc time)
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

