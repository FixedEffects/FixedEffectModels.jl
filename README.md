[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffectModels.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffects.jl?branch=master)
[![Build Status](https://travis-ci.org/matthieugomez/FixedEffectModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffects.jl)

The function `reg` estimates linear models. Compared to the Julia function `lm`, `reg`
- computes robust standard errors (White or clustered)
- estimates models with high dimensional fixed effects
- returns a very light object (mainly coefficients and covariance matrix). 

It is a basic and mostly untested implementation of the packages `reghdfe` in Stata and `lfe` in R.

## Fixed effects

Add (an arbitrary number of) fixed effects using `|`. Fixed effects must be of type PooledDataArray.

```julia
df = dataset("plm", "Cigar")
df[:State] =  pool(df[:State]
reg(Sales ~ NDI | State, df)
df[:Year] =  pool(df[:Year]
reg(Sales ~ NDI | (State + Year), df)
```

In the `DataFrames` package, the function `pool` transforms one column into a  `PooledDataArray`, the function `group` transforms multiple columns into a `PooledDataArray`.

Add interactions with continuous variable using `&`

```julia
reg(Sales ~ NDI | (State + State&Year))
```



## Errors

Compute robust standard errors using a third argument

```julia
reg(Sales ~ NDI, df,)
reg(Sales ~ NDI, df, VceWhite())
reg(Sales ~ NDI, df, VceCluster([:State]))
reg(Sales ~ NDI, df, VceCluster([:State, :Year]))

```

For now, `VceSimple()` (default), `VceWhite()` and `VceCluster(cols)` are implemented.

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

## demean
The function `demean` demeans columns with respect to fixed effects. 

`demean!` replaces columns in the original dataset

```julia
df[:StateYear] = group(df[:State, :Year])
demean(df, [:Sales], nothing ~ StateYear)
```

