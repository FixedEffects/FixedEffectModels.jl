[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffectModels.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffects.jl?branch=master)
[![Build Status](https://travis-ci.org/matthieugomez/FixedEffectModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffects.jl)

The function `reg` estimates linear models with high dimensional categorical variables. 

`reg` also computes robust standard errors (White or clustered). 
It is a basic and mostly untested implementation of the packages `reghdfe` in Stata and `lfe` in R.

## Fixed effects

Fixed effects must be variables of type PooledDataArray. Use the function `pool` to transform one column into a `PooledDataArray` and  `group` to combine multiple columns into a `PooledDataArray`.


```julia
df = dataset("plm", "Cigar")
df[:pState] =  pool(df[:pState])
df[:pState] =  pool(df[:pYear])
```

Add fixed effects with the option `absorb`

```julia
reg(Sales ~ NDI, df, absorb = :pState)
# parenthesis when multiple fixed effects
reg(Sales ~ NDI, df, absorb = [:pState, :pYear]))
```

Add interactions with continuous variable using `&`

```julia
reg(Sales ~ NDI, absorb = [:pState, :pState&Year])
```




## Errors

Compute robust standard errors using elements of type `AbstractVce`. For now, `VceSimple()` (default), `VceWhite()` and `VceCluster(cols)` are implemented.

```julia
reg(Sales ~ NDI, df,)
reg(Sales ~ NDI, df, VceWhite())
reg(Sales ~ NDI, df, VceCluster([:State]))
reg(Sales ~ NDI, df, VceCluster([:State, :Year]))
```


You can define your own type: After declaring it as a child of `AbstractVce`, define a `vcov` methods for it.

For instance,  White errors are implemented with the following code:

```julia
immutable type VceWhite <: AbstractVce 
end

function StatsBase.vcov(x::AbstractVceModel, t::VceWhite) 
	Xu = broadcast(*,  regressors(X), residuals(X))
	S = At_mul_B(Xu, Xu)
	scale!(S, nobs(X)/df_residual(X))
	sandwich(x, S) 
end
```

## Regression Result
`reg` returns a very light object of type RegressionResult. It is only composed of coefficients, covariance matrix, and some scalars like number of observations, degrees of freedoms, etc.






