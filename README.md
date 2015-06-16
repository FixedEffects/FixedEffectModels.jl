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
The function `areg` simply estimates a linear model on the demeaned variables. In particular errors are not adjusted for dof etc.

```julia
areg(Sales~NDI, df, nothing ~ State + Year)
```




