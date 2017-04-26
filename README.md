[![FixedEffectModels](http://pkg.julialang.org/badges/FixedEffectModels_0.5.svg)](http://pkg.julialang.org/?pkg=FixedEffectModels)
[![Build Status](https://travis-ci.org/matthieugomez/FixedEffectModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffectModels.jl)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffectModels.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffectModels.jl?branch=master)

This package estimates linear models with high dimensional categorical variables and/or instrumental variables. 

Its objective is similar to the Stata command [`reghdfe`](https://github.com/sergiocorreia/reghdfe) and the R function [`felm`](https://cran.r-project.org/web/packages/lfe/lfe.pdf). The package is usually much faster than these two options. The package implements a novel algorithm, which combines projection methods with the conjugate gradient descent.

![benchmark](https://cdn.rawgit.com/matthieugomez/FixedEffectModels.jl/4c7d1db39377f1ee649624c909c9017f92484114/benchmark/result.svg)

To install the package, 

```julia
Pkg.add("FixedEffectModels")
```

## Syntax
To estimate a linear model, one needs to specify a formula with `@formula`, and, eventually, a set of fixed effects with `@fe`, a way to compute standard errors with `@vcov`, or a weight variable with `@weight`.

#### `@formula`
A typical formula is composed of one dependent variable, exogeneous variables, endogeneous variables, and instrumental variables.
```
@formula(dependent variable ~ exogenous variables + (endogenous variables ~ instrumental variables))
```


```julia
using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
@formula(Sales ~ Pop + (Price ~ Pimin))
```
#### `@fe`

Fixed effect variable are indicated with the macro `@fe`. Fixed effect variables must be of type PooledDataArray (use `pool` to convert a variable to a `PooledDataArray`).

```julia
df[:StatePooled] =  pool(df[:State])
# one high dimensional fixed effect
@fe(StatePooled)
```
You can add an arbitrary number of high dimensional fixed effects, separated with `+`
```
df[:YearPooled] =  pool(df[:Year])
@fe(StatePooled + YearPooled)
```
Interact multiple categorical variables using `&` 
```julia
@fe(StatePooled&DecPooled)
```
Interact a categorical variable with a continuous variable using `&`
```julia
@fe(StatePooled + StatePooled&Year)
```
Instead of adding a categorical variable and its interaction with a continuous variable, you can directly use `*`
```julia
@fe(StatePooled*Year)
# equivalent to @fe(StatePooled StatePooled&year)
```

#### `@vcov`

Standard errors are indicated with the macro `@vcovrobust()` or `@vcovcluster()`
```julia
@vcovrobust()
@vcovcluster(StatePooled)
@vcovcluster(StatePooled + YearPooled)
```

#### `@weight`
weights are indicated with the macro `@weight`
```julia
@weight(Pop)
```

####  Putting everything together
```julia
using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
df[:StatePooled] =  pool(df[:State])
df[:YearPooled] =  pool(df[:Year])
reg(df, @formula(Sales ~ NDI), @fe(StatePooled + YearPooled), @weight(Pop))
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

## Result
`reg` returns a light object. It is composed of 
 
  - the vector of coefficients & the covariance matrix
  - a boolean vector reporting rows used in the estimation
  - a set of scalars (number of observations, the degree of freedoms, r2, etc)
  - with the option `save = true`, a dataframe aligned with the initial dataframe with residuals and, if the model contains high dimensional fixed effects, fixed effects estimates.




Methods such as `predict`, `residuals` are still defined but require to specify a dataframe as a second argument.  The problematic size of `lm` and `glm` models in R or Julia is discussed [here](http://www.r-bloggers.com/trimming-the-fat-from-glm-models-in-r/), [here](https://blogs.oracle.com/R/entry/is_the_size_of_your), [here](http://stackoverflow.com/questions/21896265/how-to-minimize-size-of-object-of-class-lm-without-compromising-it-being-passe) [here](http://stackoverflow.com/questions/15260429/is-there-a-way-to-compress-an-lm-class-for-later-prediction) (and for absurd consequences, [here](http://stackoverflow.com/questions/26010742/using-stargazer-with-memory-greedy-glm-objects) and [there](http://stackoverflow.com/questions/22577161/not-enough-ram-to-run-stargazer-the-normal-way)).



## Solution Method
Denote the model `y = X β + D θ + e` where X is a matrix with few columns and D is the design matrix from categorical variables. Estimates for `β`, along with their standard errors, are obtained in two steps:

1. `y, X`  are regressed on `D` by one of these methods
  - [MINRES on the normal equation](http://web.stanford.edu/group/SOL/software/lsmr/) with `method = :lsmr` (with a diagonal preconditioner).
  - sparse factorization with `method = :cholesky` or `method = :qr` (using the SuiteSparse library)

  The default method`:lsmr`, should be the fastest in most cases. If the method does not converge, frist please get in touch, I'd be interested to hear about your problem.  Second use the `method = :cholesky`, which should do the trick.

2.  Estimates for `β`, along with their standard errors, are obtained by regressing the projected `y` on the projected `X` (an application of the Frisch Waugh-Lovell Theorem)

3. With the option `save = true`, estimates for the high dimensional fixed effects are obtained after regressing the residuals of the full model minus the residuals of the partialed out models on `D`



## Partial out

`partial_out` returns the residuals of a set of variables after regressing them on a set of regressors. The syntax is similar to `reg` - but it accepts multiple dependent variables. It returns a dataframe with as many columns as there are dependent variables and as many rows as the original dataframe. With the option `add_mean = true`, the mean of the initial variable is added to the residuals.


```julia
using  RDatasets, DataFrames, FixedEffectModels
df = dataset("plm", "Cigar")
df[:StatePooled] =  pool(df[:State])
df[:YearPooled] =  pool(df[:Year])
result = partial_out(df, @formula(Sales + Price ~ 1), @fe(YearPooled + StatePooled), add_mean = true)
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

`partial_out` allows to examine graphically the relation between two variables after partialing out the variation due to control variables. For instance, the relationship between SepalLength seems to be decreasing in SepalWidth in the `iris` dataset
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
result = partial_out(df, @formula(SepalWidth + SepalLength ~ 1), @fe(Species), add_mean = true)
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




