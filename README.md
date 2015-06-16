
The function `demean` is a basic implementation of the function `demean` in the [lfe R package](http://journal.r-project.org/archive/2013-2/gaure.pdf). It allows to estimate models with multiple high dimentional fixed effects.

The function `demean` accepts a dataframe, a set of columns to demean (an array of symbols), and a formula. It returns a new data.frame with the demeaned version of columns.

For instance, the following command returns the residuals of the regression of Sales on State dummies and Year dummies.



## Example
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
@time FixedEffects.demean(df, [:v3,:v4], nothing ~ v1+v1:v3)
```

Factors must be PooledDataArray



## Comparison
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