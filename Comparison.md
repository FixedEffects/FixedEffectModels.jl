# Comparison

## Fixed effects
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

@time FixedEffects.demean(df, [:v3,:v4], nothing ~ v1)
# elapsed time: 0.602404481 seconds (169166440 bytes allocated, 24.85% gc time)
@time FixedEffects.demean(df, [:v3,:v4], nothing ~ v1+v2)
# elapsed time: 1.473874682 seconds (192951364 bytes allocated)
```

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
system.time(felm(v3+v4~1|v1, df))
#  user  system elapsed 
# 3.909   0.117   4.009 
system.time(felm(v3+v4~1|v1+v2, df))
#  user  system elapsed 
# 5.009   0.147   4.583 
```




## Factor model

Julia

```julia
using DataFrames
df = readtable("/Users/Matthieu/Dropbox/Github/stata-regife/data/income-deregulation.csv", separator = '\t')
df[:statecode] = PooledDataArray(df[:statecode])
df[:year] = PooledDataArray(df[:year])
@time FixedEffects.demean_factors(p30~0+intra_dummy, df , nothing ~ statecode + year, 1)
```


timer clear
timer on 1
regife p30 intra_dummy, f(state year) d(1) tol(1e-7) fast
timer off 1
timer list