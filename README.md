[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffectModels.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffectModels.jl?branch=master)
[![Build Status](https://travis-ci.org/matthieugomez/FixedEffectModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffectModels.jl)

The function `reg` estimates linear models with 
  - high dimensional categorical variable (intercept or interacted with continuous variables)
  - instrumental variables (via 2SLS)
  - robust standard errors (White or clustered) 


reg is fast (code used for this benchmark [here](benchmark/result.md))

![benchmark](https://cdn.rawgit.com/matthieugomez/FixedEffectModels.jl/master/benchmark/result.svg)



`reg` returns a very light object. This allows to estimate multiple models on the same DataFrame without worrying about RAM. It is simply composed of 
 
  - the vector of coefficients, 
  - the covariance matrix, 
  - a set of scalars (number of observations, the degree of freedoms, r2, etc)
  - a boolean vector reporting rows used in the estimation

Methods such as `predict`, `residuals` are still defined but require to specify a dataframe as a second argument.  The huge size of `lm` and `glm` models in R (and for now in Julia) is discussed [here](http://www.r-bloggers.com/trimming-the-fat-from-glm-models-in-r/), [here](https://blogs.oracle.com/R/entry/is_the_size_of_your), [here](http://stackoverflow.com/questions/21896265/how-to-minimize-size-of-object-of-class-lm-without-compromising-it-being-passe) [here](http://stackoverflow.com/questions/15260429/is-there-a-way-to-compress-an-lm-class-for-later-prediction) (and for absurd consequences, [here](http://stackoverflow.com/questions/26010742/using-stargazer-with-memory-greedy-glm-objects) and [there](http://stackoverflow.com/questions/22577161/not-enough-ram-to-run-stargazer-the-normal-way)).


## Syntax

The general syntax is

```julia
reg(depvar ~ exogenousvars + (endogeneousvars = instrumentvars) |> absorbvars, df)
```

```julia
using  RDatasets, DataFrames, FixedEffectModels
df = dataset("plm", "Cigar")
df[:pState] =  pool(df[:State])
reg(Sales ~ NDI |> pState, df)
#>                          Fixed Effect Model                         
#> =====================================================================
#> Dependent variable          Sales   Number of obs                1380
#> Degree of freedom              47   R2                          0.207
#> R2 Adjusted                 0.179   F Statistics:             7.40264
#> =====================================================================
#>         Estimate  Std.Error  t value Pr(>|t|)   Lower 95%   Upper 95%
#> ---------------------------------------------------------------------
#> NDI  -0.00170468 9.13903e-5 -18.6527    0.000 -0.00188396 -0.00152539
#> =====================================================================
```


### Fixed effects


- Specify multiple high dimensional fixed effects.

  ```julia
  df[:pYear] =  pool(df[:Year])
  reg(Sales ~ NDI |> pState + pYear, df)
  ```
- Interact fixed effects with continuous variables using `&`

  ```julia
  reg(Sales ~ NDI |> pState + pState&Year)
  ```

- Categorical variables must be of type PooledDataArray. Use the function `pool` to transform one column into a `PooledDataArray` and  `group` to combine multiple columns into a `PooledDataArray`.


### Weights

 Weights are supported with the option `weight`. They correspond to R weights and analytical weights in Stata.

```julia
reg(Sales ~ NDI |> pState, weight = :Pop)
```

## Errors
Compute robust standard errors with a third argument

For now, `VcovSimple()` (default), `VcovWhite()` and `VcovCluster(cols)` are implemented.

```julia
reg(Sales ~ NDI, df, VcovWhite())
reg(Sales ~ NDI, df, VcovCluster([:State]))
reg(Sales ~ NDI, df, VcovCluster([:State, :Year]))
```


You can easily define your own type: after declaring it as a child of `AbstractVcov`, define a `vcov` methods for it. For instance,  White errors are implemented with the following code:

```julia
immutable type VcovWhite <: AbstractVcov 
end

function StatsBase.vcov(x::AbstractVcovModel, t::VcovWhite) 
	Xu = broadcast(*,  regressors(X), residuals(X))
	S = At_mul_B(Xu, Xu)
	scale!(S, nobs(X)/df_residual(X))
	sandwich(x, S) 
end
```


## Partial out

`partial_out` returns the residuals of a set of variables after regressing them on a set of regressors. Models are estimated on only the rows where *none* of the dependent variables is missing. The result is a dataframe with as many columns as there are dependent variables and as many rows as the original dataframe.
The syntax is similar to `reg` - just with multiple `lhs`. With the option `add_mean = true`, the mean of the initial variable mean is added to the residuals.



```julia
using  RDatasets, DataFrames, FixedEffectModels
df = dataset("plm", "Cigar")
df[:pState] =  pool(df[:State])
df[:pYear] =  pool(df[:Year])
result = partial_out(Sales + Price ~ 1|> pYear + pState, df, add_mean = true)
#> 1380x2 DataFrame
#> | Row  | Sales   | Price   |
#> |------|---------|---------|
#> | 1    | 107.029 | 69.7684 |
#> | 2    | 112.099 | 70.1641 |
#> | 3    | 113.325 | 69.9445 |
#> | 4    | 110.523 | 70.1401 |
#> | 5    | 109.501 | 69.5184 |
#> | 6    | 104.332 | 71.451  |
#> | 7    | 107.266 | 71.3488 |
#> | 8    | 109.769 | 71.2836 |
#> ⋮
#> | 1372 | 117.975 | 64.5648 |
#> | 1373 | 116.216 | 64.8778 |
#> | 1374 | 117.605 | 68.7996 |
#> | 1375 | 106.281 | 67.0257 |
#> | 1376 | 113.707 | 68.3996 |
#> | 1377 | 115.144 | 63.4974 |
#> | 1378 | 105.099 | 61.1083 |
#> | 1379 | 119.936 | 49.9365 |
#> | 1380 | 122.503 | 57.7017 |
```

One can then examine graphically the relation between two variables after removing the variation due to control variables.

```julia
using Gadfly
plot(
   layer(result, x="Price", y="Sales", Stat.binmean(n=10), Geom.point),
   layer(result, x="Price", y="Sales", Geom.smooth(method=:lm))
)

```
![binscatter](https://cdn.rawgit.com/matthieugomez/FixedEffectModels.jl/master/benchmark/binscatter.svg)
(this basically replicates the Stata program [binscatter](https://michaelstepner.com/binscatter/)




