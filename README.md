[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffectModels.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffectModels.jl?branch=master)
[![Build Status](https://travis-ci.org/matthieugomez/FixedEffectModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffectModels.jl)

1. The function `reg` estimates linear models with 
  - instrumental variables (via 2SLS)
  - high dimensional categorical variable (as intercept or interacted with continuous variables)
  - robust standard errors (White or clustered) 


2. `reg` returns a very light object. This allows to estimate multiple models on the same DataFrame without worrying about your RAM. It is simply composed of 
 
  - the vector of coefficients, 
  - the covariance matrix, 
  - a boolean vector reporting rows used in the estimation
  - a set of scalars (number of observations, the degree of freedoms, r2, etc). 

  Methods such as `predict`, `residuals` are still defined but require a dataframe as a second argument.  The huge size of `lm` and `glm` models in R (and for now in Julia) is discussed [here](http://www.r-bloggers.com/trimming-the-fat-from-glm-models-in-r/), [here](https://blogs.oracle.com/R/entry/is_the_size_of_your), [here](http://stackoverflow.com/questions/21896265/how-to-minimize-size-of-object-of-class-lm-without-compromising-it-being-passe) [here](http://stackoverflow.com/questions/15260429/is-there-a-way-to-compress-an-lm-class-for-later-prediction) (and for absurd consequences, [here](http://stackoverflow.com/questions/26010742/using-stargazer-with-memory-greedy-glm-objects) and [there](http://stackoverflow.com/questions/22577161/not-enough-ram-to-run-stargazer-the-normal-way)).



3. reg is fast (simple benchmark [here](benchmark/result.md))

![benchmark](https://cdn.rawgit.com/matthieugomez/FixedEffectModels.jl/master/benchmark/result.svg)

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
The syntax is similar to `reg` - just with multiple `lhs`. 

```julia
using  RDatasets, DataFrames, FixedEffectModels
df = dataset("plm", "Cigar")
df[:pState] =  pool(df[:State])
partial_out(Sales + Price ~ 1 |> pState, df)
#> 1380x2 DataFrame
#> | Row  | Sales    | Price    |
#> |------|----------|----------|
#> | 1    | -13.5767 | -40.5467 |
#> | 2    | -12.0767 | -39.3467 |
#> | 3    | -8.97667 | -39.3467 |
#> | 4    | -11.0767 | -37.6467 |
#> | 5    | -11.9767 | -37.5467 |
#> | 6    | -19.0767 | -33.5467 |
#> | 7    | -17.3767 | -32.5467 |
#> | 8    | -17.6767 | -29.5467 |
#> ⋮
#> | 1372 | -7.59667 | 19.8367  |
#> | 1373 | -10.7967 | 25.5367  |
#> | 1374 | -11.6967 | 35.9367  |
#> | 1375 | -26.0967 | 40.8367  |
#> | 1376 | -22.1967 | 51.0367  |
#> | 1377 | -25.0967 | 56.7367  |
#> | 1378 | -39.5967 | 67.6367  |
#> | 1379 | -27.3967 | 65.1367  |
#> | 1380 | -25.6967 | 93.2367  |
partial_out(Sales + Price ~ CPI |> pState, df)
#> 1380x2 DataFrame
#> | Row  | Sales    | Price    |
#> |------|----------|----------|
#> | 1    | -21.2454 | 6.35952  |
#> | 2    | -19.6741 | 7.12315  |
#> | 3    | -16.4849 | 6.57769  |
#> | 4    | -18.4244 | 7.29585  |
#> | 5    | -19.146  | 6.30493  |
#> | 6    | -25.9963 | 8.77763  |
#> | 7    | -23.9575 | 7.70487  |
#> | 8    | -23.8829 | 8.41392  |
#> ⋮
#> | 1372 | -2.19184 | -13.222  |
#> | 1373 | -4.73191 | -11.5585 |
#> | 1374 | -5.2752  | -3.34031 |
#> | 1375 | -18.9618 | -2.80401 |
#> | 1376 | -14.2235 | 2.26863  |
#> | 1377 | -16.1069 | 1.75036  |
#> | 1378 | -29.4119 | 5.34115  |
#> | 1379 | -16.2309 | -3.15894 |
#> | 1380 | -13.7996 | 20.4683  |
```

With the option `add_mean = TRUE`, the initial variable mean is added to the residuals.




