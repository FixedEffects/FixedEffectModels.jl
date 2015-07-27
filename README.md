[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffectModels.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffectModels.jl?branch=master)
[![FixedEffectModels](http://pkg.julialang.org/badges/FixedEffectModels_release.svg)](http://pkg.julialang.org/?pkg=FixedEffectModels&ver=release)
[![FixedEffectModels](http://pkg.julialang.org/badges/FixedEffectModels_nightly.svg)](http://pkg.julialang.org/?pkg=FixedEffectModels&ver=nightly)




The function `reg` estimates linear models with 
  - high dimensional categorical variable (intercept or interacted with continuous variables). 
  - instrumental variables (via 2SLS)
  - robust standard errors (White or clustered) 


`reg` is fast (benchmark [here](https://github.com/matthieugomez/FixedEffectModels.jl/blob/master/benchmark/benchmark.md))
![benchmark](https://cdn.rawgit.com/matthieugomez/FixedEffectModels.jl/4c7d1db39377f1ee649624c909c9017f92484114/benchmark/result.svg)


`reg` returns a very light object. This allows to estimate multiple models on the same DataFrame without ever worrying about RAM. It is simply composed of 
 
  - the vector of coefficients & the covariance matrix
  - a boolean vector reporting rows used in the estimation
  - a set of scalars (number of observations, the degree of freedoms, r2, etc)
  - only with the option `save = true`, a dataframe aligned with the initial dataframe storing residuals and high dimensional fixed effects 

Methods such as `predict`, `residuals` are still defined but require to specify a dataframe as a second argument.  The huge size of `lm` and `glm` models in R (and for now in Julia) is discussed [here](http://www.r-bloggers.com/trimming-the-fat-from-glm-models-in-r/), [here](https://blogs.oracle.com/R/entry/is_the_size_of_your), [here](http://stackoverflow.com/questions/21896265/how-to-minimize-size-of-object-of-class-lm-without-compromising-it-being-passe) [here](http://stackoverflow.com/questions/15260429/is-there-a-way-to-compress-an-lm-class-for-later-prediction) (and for absurd consequences, [here](http://stackoverflow.com/questions/26010742/using-stargazer-with-memory-greedy-glm-objects) and [there](http://stackoverflow.com/questions/22577161/not-enough-ram-to-run-stargazer-the-normal-way)).




# installation

To install the package, 

```julia
Pkg.add("FixedEffectModels")
```


## reg

The general syntax is
```julia
reg(f::Formula, 
    df::AbstractDataFrame, 
    vcov_method::AbstractVcovMethod = VcovSimple(); 
    weight::Union(Symbol, Nothing) = nothing, 
    subset::Union(AbstractVector{Bool}, Nothing) = nothing, 
    maxiter::Int64 = 10000, tol::Float64 = 1e-8
    )
```


#### Formula

A typical formula is composed of one dependent variable, exogeneous variables, endogeneous variables, instruments, and high dimensional fixed effects

```
depvar ~ exogeneousvars + (endogeneousvars = instrumentvars) |> absorbvars
```

##### Fixed effects


- Estimate models with an arbitrary number of high dimensional fixed effects.

  ```julia
  using RDatasets, DataFrames, FixedEffectModels
  df = dataset("plm", "Cigar")
  df[:pState] =  pool(df[:State])
  df[:pYear] =  pool(df[:Year])
  reg(Sales ~ Price |> pState + pYear, df)
  # ===============================================================
  # Number of obs             1380   Degree of freedom           77
  # R2                       0.137   R2 Adjusted              0.085
  # F Stat                 205.989   p-val                    0.000
  # Iterations                   2   Converged:                true
  # ===============================================================
  #        Estimate Std.Error  t value Pr(>|t|) Lower 95% Upper 95%
  # ---------------------------------------------------------------
  # Price  -1.08471 0.0755775 -14.3523    0.000  -1.23298 -0.936445
  # ===============================================================
  ```
- Interact fixed effects with continuous variables using `&`

  ```julia
  reg(Sales ~ NDI |> pState + pState&Year, df)
  # =====================================================================
  # Number of obs                1380   Degree of freedom              93
  # R2                          0.245   R2 Adjusted                 0.190
  # F Stat                    417.342   p-val                       0.000
  # Iterations                      2   Converged:                   true
  # =====================================================================
  #         Estimate   Std.Error t value Pr(>|t|)   Lower 95%   Upper 95%
  # ---------------------------------------------------------------------
  # NDI  -0.00568607 0.000278334 -20.429    0.000 -0.00623211 -0.00514003
  # =====================================================================
  ```

- Categorical variables must be of type PooledDataArray. Use the function `pool` to transform one column into a `PooledDataArray` and  `group` to combine multiple columns into a `PooledDataArray`.

##### Instrumental variables

- Models with instruments variables are estimated using 2SLS.
- `reg` tests for weak instruments by computing the Kleibergen-Paap rk Wald F statistic, a generalization of the Cragg-Donald Wald F statistic for non i.i.d. errors. The statistic is similar to the one returned by the Stata command `ivreg2`.

  ```julia
  reg(Sales ~ (Price = Pimin), df)
  #                                IV Model                               
  # ======================================================================
  # Number of obs                 1380  Degree of freedom                2
  # R2                           0.096  R2 Adjusted                  0.095
  # F Statistic                117.173  Prob > F                     0.000
  # First Stage F-stat (KP)    52248.8  First State p-val (KP):      0.000
  # ======================================================================
  #               Estimate Std.Error  t value Pr(>|t|) Lower 95% Upper 95%
  # ----------------------------------------------------------------------
  # Price        -0.207335  0.019154 -10.8247    0.000  -0.24491 -0.169761
  # (Intercept)    138.195   1.53661  89.9347    0.000    135.18   141.209
  # ======================================================================
  ```

#### Weights

 Weights are supported with the option `weight`. They correspond to analytical weights in Stata.

```julia
reg(Sales ~ Price |> pState, df, weight = :Pop)
```

#### Subset

Estimate a model on a subset of your data with the option `subset` 

```julia
reg(Sales ~ NDI |> pState, weight = :Pop, subset = df[:pState] .< 30)
```

#### Errors
Compute robust standard errors by constructing an object of type `AbstractVcovMethod`. For now, `VcovSimple()` (default), `VcovWhite()` and `VcovCluster(cols)` are implemented.

```julia
reg(Sales ~ NDI, df, VcovWhite())
reg(Sales ~ NDI, df, VcovCluster([:State]))
reg(Sales ~ NDI, df, VcovCluster([:State, :Year]))
```

#### Save
The option `save = true` saves a new dataframe which stores residuals and the eventual fixed effects. The new dataframe is aligned with the initial dataframe: rows non used in the estimation are simply filled with NA.

Fixed effects are obtained by the Kaczmarz algorithm, according to the following normalization: for each factor except the first one, the sum of fixed effects within each connected component is normalized to zero.

## Partial out

`partial_out` returns the residuals of a set of variables after regressing them on a set of regressors. The syntax is similar to `reg` - just with multiple `lhs`. It returns  a dataframe with as many columns as there are dependent variables and as many rows as the original dataframe.
The regression model is estimated on only the rows where *none* of the dependent variables is missing. With the option `add_mean = true`, the mean of the initial variable is added to the residuals.



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

This allows to examine graphically the relation between two variables after partialing out the variation due to control variables. For instance, the relationship between SepalLength seems to be decreasing in SepalWidth in the `iris` dataset
```julia
using  RDatasets, DataFrames, Gadfly, FixedEffectModels
df = dataset("datasets", "iris")
plot(
   layer(df, x="SepalWidth", y="SepalLength", Stat.binmean(n=10), Geom.point),
   layer(df, x="SepalWidth", y="SepalLength", Geom.smooth(method=:lm))
)
```
![binscatter](http://cdn.rawgit.com/matthieugomez/FixedEffectModels.jl/master/benchmark/first.svg)

But the relationship is actually increasing within species.
```julia
plot(
   layer(df, x="SepalWidth", y="SepalLength", color = "Species", Stat.binmean(n=10), Geom.point),
   layer(df, x="SepalWidth", y="SepalLength", color = "Species", Geom.smooth(method=:lm))
)
```
![binscatter](https://cdn.rawgit.com/matthieugomez/FixedEffectModels.jl/9a12681d81f9d713cec3b88b1abf362cdddb9a14/benchmark/second.svg)


If there is large number of groups, a better way to visualize this fact is to plot the variables after partialing them out:
```julia
result = partial_out(SepalWidth + SepalLength ~ 1|> Species, df, add_mean = true)
using Gadfly
plot(
   layer(result, x="SepalWidth", y="SepalLength", Stat.binmean(n=10), Geom.point),
   layer(result, x="SepalWidth", y="SepalLength", Geom.smooth(method=:lm))
)
```
![binscatter](https://cdn.rawgit.com/matthieugomez/FixedEffectModels.jl/9a12681d81f9d713cec3b88b1abf362cdddb9a14/benchmark/third.svg)

The combination of `partial_out` and Gadfly `Stat.binmean` is very similar to the the Stata program [binscatter](https://michaelstepner.com/binscatter/).





# References
Models with high dimensional fixed effects:
 - Simen Gaure. 2013. *OLS with multiple high dimensional category variables*. Computational Statistics & Data Analysis http://www.sciencedirect.com/science/article/pii/S0167947313001266
 -  Sergio Correia, 2015. *reghdfe: Stata module for linear and instrumental-variable/GMM regression absorbing multiple levels of fixed effects.*  https://ideas.repec.org/c/boc/bocode/s457874.html


First Stage F-Stat for IV models:
- Frank Kleibergen, Mark E Schaffer. 2014 *ranktest: Stata module to test the rank of a matrix using the Kleibergen-Paap rk statistic* https://ideas.repec.org/c/boc/bocode/s456865.html
- Christopher F Baum, Mark E Schaffer, Steven Stillman, 2015. *ivreg2: Stata module for extended instrumental variables/2SLS and GMM estimation* https://ideas.repec.org/c/boc/bocode/s457874.html



