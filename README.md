[![Build Status](https://travis-ci.org/matthieugomez/FixedEffectModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffectModels.jl)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffectModels.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffectModels.jl?branch=master)
[![FixedEffectModels](http://pkg.julialang.org/badges/FixedEffectModels_0.4.svg)](http://pkg.julialang.org/?pkg=FixedEffectModels)

This package estimates linear models with high dimensional categorical variables and/or instrumental variables. 

Its objective is similar to the Stata command `reghdfe` and the R command `felm`.

The package is [fast](https://github.com/matthieugomez/FixedEffectModels.jl/blob/master/benchmark/benchmark.md):
![benchmark](https://cdn.rawgit.com/matthieugomez/FixedEffectModels.jl/4c7d1db39377f1ee649624c909c9017f92484114/benchmark/result.svg)

To install the package, 

```julia
Pkg.add("FixedEffectModels")
```


## Formula

A typical formula is composed of one dependent variable, exogeneous variables, endogeneous variables, instruments, and high dimensional fixed effects

```
depvar ~ exogeneousvars + (endogeneousvars = instrumentvars) |> absorbvars
```
High dimensional categorical variables should enter after the `|>` symbol


## result
`reg` returns a light object. This allows to estimate multiple models without worrying about memory space. It is simply composed of 
 
  - the vector of coefficients & the covariance matrix
  - a boolean vector reporting rows used in the estimation
  - a set of scalars (number of observations, the degree of freedoms, r2, etc)
  - with the option `save = true`, a dataframe aligned with the initial dataframe with residuals and, if the model contains high dimensional fixed effects, fixed effects estimates.

Methods such as `predict`, `residuals` are still defined but require to specify a dataframe as a second argument.  The problematic size of `lm` and `glm` models in R or Julia is discussed [here](http://www.r-bloggers.com/trimming-the-fat-from-glm-models-in-r/), [here](https://blogs.oracle.com/R/entry/is_the_size_of_your), [here](http://stackoverflow.com/questions/21896265/how-to-minimize-size-of-object-of-class-lm-without-compromising-it-being-passe) [here](http://stackoverflow.com/questions/15260429/is-there-a-way-to-compress-an-lm-class-for-later-prediction) (and for absurd consequences, [here](http://stackoverflow.com/questions/26010742/using-stargazer-with-memory-greedy-glm-objects) and [there](http://stackoverflow.com/questions/22577161/not-enough-ram-to-run-stargazer-the-normal-way)).


## Syntax

Similarly to the `DataFrames` documentation

-  Categorical variable must be of type PooledDataArray.

  ```julia
  using DataFrames, RDatasets, FixedEffectModels
  df = dataset("plm", "Cigar")
  df[:StatePooled] =  pool(df[:State])
  df[:YearPooled] =  pool(df[:Year])
  reg(Sales ~ Price |> StatePooled + YearPooled, df)
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

-  Combine multiple categorical variables with the operator `&` 

  ```julia
  using DataFrames, RDatasets, FixedEffectModels
  df = dataset("plm", "Cigar")
  df[:StatePooled] =  pool(df[:State])
  df[:DecPooled] =  pool(div(df[:Year], 10))
  reg(Sales ~ NDI |> StatePooled&DecPooled, df)
  # =====================================================================
  # Number of obs:               1380   Degree of freedom:            185
  # R2:                         0.923   R2 within:                  0.101
  # F-Statistic:              133.916   p-value:                    0.000
  # Iterations:                     1   Converged:                   true
  # =====================================================================
  #        Estimate   Std.Error  t value Pr(>|t|)   Lower 95%   Upper 95%
  # ---------------------------------------------------------------------
  # NDI  -0.0020522 0.000177339 -11.5722    0.000 -0.00240014 -0.00170427
  # =====================================================================
  ```



-  In addition to specifying main effects, specify interactions with continuous variables using the `&` operator:

  ```julia
  using DataFrames, RDatasets, FixedEffectModels
  df = dataset("plm", "Cigar")
  df[:StatePooled] =  pool(df[:State])
  reg(Sales ~ NDI |> StatePooled + StatePooled&Year, df)
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

-  Specify both main effects and an interaction term at once using the `*` operator:
  
  ```julia
  using DataFrames, RDatasets, FixedEffectModels
  df = dataset("plm", "Cigar")
  df[:StatePooled] =  pool(df[:State])
  reg(Sales ~ NDI |> StatePooled*Year, df)
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

## Errors

Compute robust standard errors by constructing an object of type `AbstractVcovMethod`. For now, `VcovSimple()` (default), `VcovWhite()` and `VcovCluster(cols)` are implemented.

```julia
reg(Sales ~ NDI, df, VcovWhite())
reg(Sales ~ NDI, df, VcovCluster([:State]))
reg(Sales ~ NDI, df, VcovCluster([:State, :Year]))
```

`reg` also supports `weights`, `subset`. Type `?reg` to learn about these options.

## Solution Method
Denote the model `y = X β + D θ + e` where X is a matrix with few columns and D is the design matrix from categorical variables. Estimates for `β`, along with their standard errors, are obtained in two steps:

1. `y, X`  are regressed on `D` by one of these methods
  - [MINRES on the normal equation](http://web.stanford.edu/group/SOL/software/lsmr/) with `method = :lsmr` (with a diagonal preconditioner).
  - sparse factorization with `method = :cholesky` or `method = :qr` (using the SuiteSparse library)

  The default method, `:lsmr`, is faster in most cases. Now, when the design matrix is poorly conditioned, `method = :cholesky` may be faster.

2.  Estimates for `β`, along with their standard errors, are obtained by regressing the projected `y` on the projected `X` (an application of the Frisch Waugh-Lovell Theorem)

3. With the option `save = true`, estimates for the high dimensional fixed effects are obtained after regressing the residuals of the full model minus the residuals of the partialed out models on `D`



## Partial out

`partial_out` returns the residuals of a set of variables after regressing them on a set of regressors. The syntax is similar to `reg` - but it accepts multiple dependent variables. It returns a dataframe with as many columns as there are dependent variables and as many rows as the original dataframe.
The regression model is estimated on only the rows where *none* of the dependent variables is missing. With the option `add_mean = true`, the mean of the initial variable is added to the residuals.


```julia
using  RDatasets, DataFrames, FixedEffectModels
df = dataset("plm", "Cigar")
df[:StatePooled] =  pool(df[:State])
df[:YearPooled] =  pool(df[:Year])
result = partial_out(Sales + Price ~ 1|> YearPooled + StatePooled, df, add_mean = true)
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

The combination of `partial_out` and Gadfly `Stat.binmean` is similar to the the Stata program [binscatter](https://michaelstepner.com/binscatter/).





# References

Baum, C. and Schaffer, M. (2013) *AVAR: Stata module to perform asymptotic covariance estimation for iid and non-iid data robust to heteroskedasticity, autocorrelation, 1- and 2-way clustering, and common cross-panel autocorrelated disturbances*. Statistical Software Components, Boston College Department of Economics.

Correia, S. (2014) *REGHDFE: Stata module to perform linear or instrumental-variable regression absorbing any number of high-dimensional fixed effects*. Statistical Software Components, Boston College Department of Economics.

Fong, DC. and Saunders, M. (2011) *LSMR: An Iterative Algorithm for Sparse Least-Squares Problems*.  SIAM Journal on Scientific Computing

Gaure, S. (2013) *OLS with Multiple High Dimensional Category Variables*. Computational Statistics and Data Analysis

Kleibergen, F, and Paap, R. (2006) *Generalized reduced rank tests using the singular value decomposition.* Journal of econometrics 

Kleibergen, F. and Schaffer, M.  (2007) *RANKTEST: Stata module to test the rank of a matrix using the Kleibergen-Paap rk statistic*. Statistical Software Components, Boston College Department of Economics.




