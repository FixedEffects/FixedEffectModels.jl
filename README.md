The package `fixedeffects` allow to estimate models with high dimensional fixed effects.


## demean
The function `demean` is a basic implementation of the functions `reghdfe` in Stata and `lfe` in R. It allows to estimate models with multiple high dimentional fixed effects.

```julia
using DataArrays, DataFrame, RDataSets, FixedEffects
df = dataset("plm", "Cigar")
```


The function `demean` accepts a dataframe, a set of columns to demean (an array of symbols), and a formula. It returns a new data.frame with the demeaned version of columns.

```
df[:State] = PooledDataArray(df[:State])
df[:Year] = PooledDataArray(df[:Year])
demean(f, [:Sales], nothing ~ State + Year)
```

To construct a fixed effect from set of variable into one factor, use `group`

```
df[:group] = group(df[:State, :Year])
demean(f, [:Sales], nothing ~ :group)
```




Interactions with continuous variable are indicated by `&`

```julia
df = dataset("plm", "Cigar")
df[:State] = PooledDataArray(df[:State])
demean(df, [:Sales], nothing ~ State + State&Year)
```




## areg
The function `areg` estimates a linear model on the demeaned variables. In particular errors are not adjusted for dof etc.

```julia
areg(Sales~NDI, df, nothing ~ State + Year)
```






## Comparison
R (lfe package, C)

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
@time FixedEffects.demean!(df, [:v3,:v4], nothing ~ v1)
# elapsed time:  0.313016191 seconds (169166440 bytes allocated, 24.85% gc time)
@time FixedEffects.demean!(df, [:v3,:v4], nothing ~ v1+v2)
# elapsed time: 1.138125588 seconds (192951364 bytes allocated)
````

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