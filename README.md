[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffectModels.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffects.jl?branch=master)
[![Build Status](https://travis-ci.org/matthieugomez/FixedEffectModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffects.jl)

This package estimates models with high dimensional fixed effects. It is a basic and mostly untested implementation of the packages `reghdfe` in Stata and `lfe` in R


## areg
The function `reg`  estimates a linear model with high dimensional fixed effects.


The first argument is the regression formula, the second is the dataframe, the third is the error method

## Fixed effects

Add fixed effects using `|`

```julia
df = dataset("plm", "Cigar")
df[:State] =  pool(df[:State]
reg(Sales~NDI | State, df)
```

Both fixed effects and cluster columns must be of type PooledDataArray.

You can interactions with continuous variable using `&`
```julia
demean(df, [:Sales], nothing ~ State + State&Year, nothing ~ State)
```

To construct PooledDataArray from one column use `pool`. To construct PooledDataArray from multiple columns, use `group` 


## Errors

Error types can be specified by a third argument

```julia
reg(Sales~NDI | State, df,)
reg(Sales~NDI | State, df, vceRobust())
reg(Sales~NDI | State, df, vceCluster([:State]))
```

The third argument is any instance of a type that inherits from the Abstract type `AbstractVce`. You can write your own type as long as you define a `vcov` model for it.

For now, `vceSimple()`, `vceRobust()` and `vceCluster(cols)` are implemented.

## demean
The function `demean` demeans columns with respect to fixed effects. 

`demean!` replaces columns in the original dataset

```julia
df[:StateYear] = group(df[:State, :Year])
demean(df, [:Sales], nothing ~ StateYear)
```

