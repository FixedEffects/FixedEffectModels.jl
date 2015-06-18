[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffectModels.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffects.jl?branch=master)
[![Build Status](https://travis-ci.org/matthieugomez/FixedEffectModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffects.jl)

This package estimates models with high dimensional fixed effects. It is a basic and mostly untested implementation of the packages `reghdfe` in Stata and `lfe` in R





## areg
The function `areg`  estimates a linear model after demeaning variables. In particular errors are not adjusted for dof etc.

```julia
df = dataset("plm", "Cigar")
df[:State] =  pool(df[:State]
areg(Sales~NDI, df, nothing ~ State + Year, nothing ~ State)
```
THe first argument is the regression formula, the second is the dataframe, the third is the set of fixed effects, the fourth is the set of clusters.

Both fixed effects and cluster columns must be of type PooledDataArray.


You can interactions with continuous variable using `&`
```julia

demean(df, [:Sales], nothing ~ State + State&Year, nothing ~ State)
```



## demean
The function `demean` demeans columns with respect to fixed effects. 


`demean!` replaces columns in the original dataset

```julia
df[:StateYear] = group(df[:State, :Year])
demean(df, [:Sales], nothing ~ StateYear)
```


##
Construct PooledDataArray from one column using `pool`
Construct PooledDataArray from multiple columns using `group` 

