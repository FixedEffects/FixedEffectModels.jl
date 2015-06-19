[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffectModels.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffects.jl?branch=master)
[![Build Status](https://travis-ci.org/matthieugomez/FixedEffectModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffects.jl)

Contrary to the function `lm`, the function `reg`  estimates linear models with (i) robust standard errors (ii) high dimensional fixed effects. Moreover, `reg` returns a very light object (mainly coefficients and covariance matrix). It is a basic and mostly untested implementation of the packages `reghdfe` in Stata and `lfe` in R.

## Fixed effects

Add (an arbitrary number of) fixed effects using `|`. Fixed effects must be of type PooledDataArray.

```julia
df = dataset("plm", "Cigar")
df[:State] =  pool(df[:State]
reg(Sales ~ NDI | State, df)
df[:Year] =  pool(df[:Year]
reg(Sales ~ NDI | (State + Year), df)
```

Construct PooledDataArray from one column using `pool`. Construct PooledDataArray from multiple columns using `group`

Add interactions with continuous variable using `&`

```julia
reg(Sales ~ NDI | (State + State&Year))
```



## Errors

Compute robust standard errors using a third argument

```julia
reg(Sales ~ NDI, df,)
reg(Sales ~ NDI, df, vceWhite())
reg(Sales ~ NDI, df, vceCluster([:State]))
```

For now, `vceSimple()` (default), `vceWhite()` and `vceCluster(cols)` are implemented.

You can define your own type: After declaring it as a child of `AbstractVce`, define a `vcov` methods for it.

For instance,  White errors are implemented with the following code:

```julia
immutable type VceWhite <: AbstractVce 
end

function StatsBase.vcov(x::AbstractVceModel, t::VceWhite, df::AbstractDataFrame) 
	Xu = broadcast(*,  regressors(X), residuals(X))
	S = At_mul_B(Xu, Xu)
	scale!(S, nobs(X)/df_residual(X))
	sandwich(x, S) 
end
```
Note the AbstractDataFrame in the signature of `vcov`. This does not make sense for `VceWhite` but it allows to use symbols when errors require supplementary variables, like `vceCluster([:State])`.

## demean
The function `demean` demeans columns with respect to fixed effects. 

`demean!` replaces columns in the original dataset

```julia
df[:StateYear] = group(df[:State, :Year])
demean(df, [:Sales], nothing ~ StateYear)
```

