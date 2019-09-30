[![Build Status](https://travis-ci.org/matthieugomez/FixedEffectModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffectModels.jl)
[![pipeline status](https://gitlab.com/JuliaGPU/FixedEffectModels.jl/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/FixedEffectModels.jl/commits/master)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffectModels.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/matthieugomez/FixedEffectModels.jl?branch=master)

This package estimates linear models with high dimensional categorical variables and/or instrumental variables. 

Its objective is similar to the Stata command [`reghdfe`](https://github.com/sergiocorreia/reghdfe) and the R function [`felm`](https://cran.r-project.org/web/packages/lfe/lfe.pdf). The package tends to be much faster than these two options.

![benchmark](http://www.matthieugomez.com/files/fixedeffectmodels_benchmark.png)

## Estimate a model
To estimate a `@model`, specify  a formula with a way to compute standard errors with the argument `vcov`.

```julia
using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
reg(df, @model(Sales ~ NDI + fe(State) + fe(Year), vcov = cluster(State)), weights = :Pop)
# =====================================================================
# Number of obs:               1380   Degrees of freedom:            31
# R2:                         0.804   R2 within:                  0.139
# F-Statistic:              13.3481   p-value:                    0.000
# Iterations:                     6   Converged:                   true
# =====================================================================
#         Estimate  Std.Error  t value Pr(>|t|)   Lower 95%   Upper 95%
# ---------------------------------------------------------------------
# NDI  -0.00526264 0.00144043 -3.65351    0.000 -0.00808837 -0.00243691
# =====================================================================
```
- A typical formula is composed of one dependent variable, exogeneous variables, endogeneous variables, instrumental variables, and a set of high-dimensional fixed effects.
	
	```julia
	dependent variable ~ exogenous variables + (endogenous variables ~ instrumental variables) + fe(fixedeffect variable)
	```

	High-dimensional fixed effect variables are indicated with the function `fe`.  You can add an arbitrary number of high dimensional fixed effects, separated with `+`. Moreover, you can interact a fixed effect with a continuous variable (e.g. `fe(State)&Year`) or with another fixed effect (e.g. `fe(State)&fe(Year)`).

- Standard errors are indicated with the keyword argument `vcov`.
	```julia
	vcov = robust()
	vcov = cluster(State)
	vcov = cluster(State + Year)
	```

## Output
`reg` returns a light object. It is composed of 
 
  - the vector of coefficients & the covariance matrix
  - a boolean vector reporting rows used in the estimation
  - a set of scalars (number of observations, the degree of freedoms, r2, etc)
  - with the option `save = true`, a dataframe aligned with the initial dataframe with residuals and, if the model contains high dimensional fixed effects, fixed effects estimates.




Methods such as `predict`, `residuals` are still defined but require to specify a dataframe as a second argument.  The problematic size of `lm` and `glm` models in R or Julia is discussed [here](http://www.r-bloggers.com/trimming-the-fat-from-glm-models-in-r/), [here](https://blogs.oracle.com/R/entry/is_the_size_of_your), [here](http://stackoverflow.com/questions/21896265/how-to-minimize-size-of-object-of-class-lm-without-compromising-it-being-passe) [here](http://stackoverflow.com/questions/15260429/is-there-a-way-to-compress-an-lm-class-for-later-prediction) (and for absurd consequences, [here](http://stackoverflow.com/questions/26010742/using-stargazer-with-memory-greedy-glm-objects) and [there](http://stackoverflow.com/questions/22577161/not-enough-ram-to-run-stargazer-the-normal-way)).


You may use [RegressionTables.jl](https://github.com/jmboehm/RegressionTables.jl) to get publication-quality regression tables.

## Construct Model Programatically
You can use
```julia
using StatsModels, DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
reg(df, Term(:Sales) ~ Term(:NDI) + FixedEffectTerm(:State) + FixedEffectTerm(:Year); vcov = :(cluster(State)))
```

## Performances
#### GPU
The package has support for GPUs (Nvidia) (thanks to Paul Schrimpf). This makes the package an order of magnitude faster for complicated problems.

First make sure that `using CuArrays` works without issue.
```julia
using FixedEffectModels
df = dataset("plm", "Cigar")
reg(df, @model(Sales ~ NDI + fe(State) + fe(Year)), method = :lsmr_gpu)
```

It is also encouraged to set the floating point precision to float32 when working on the GPU as that is usually much faster (using the option `double_precision = false`).


#### Parallel Computing
The package has support for [multi-threading](https://docs.julialang.org/en/v1.2/manual/parallel-computing/#man-multithreading-1 and [multi-cores](https://docs.julialang.org/en/v1.2/manual/parallel-computing/#Multi-Core-or-Distributed-Processing-1). In this case, each regressor is demeaned in a different thread. It only allows for a modest speedup (between 10% and 60%) since the demeaning operation is typically memory bound.

```julia
# Multi-threading
Threads.nthreads()
using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
reg(df, @model(Sales ~ NDI + fe(State) + fe(Year)), method = :lsmr_threads)

# Multi-cores 
using Distributed
addprocs(4)
@everywhere using DataFrames,  RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
reg(df, @model(Sales ~ NDI + fe(State) + fe(Year)), method = :lsmr_cores)
```


## Solution Method
Denote the model `y = X β + D θ + e` where X is a matrix with few columns and D is the design matrix from categorical variables. Estimates for `β`, along with their standard errors, are obtained in two steps:

1. `y, X`  are regressed on `D` using the package [FixedEffects.jl](https://github.com/matthieugomez/FixedEffects.jl)
2.  Estimates for `β`, along with their standard errors, are obtained by regressing the projected `y` on the projected `X` (an application of the Frisch Waugh-Lovell Theorem)
3. With the option `save = true`, estimates for the high dimensional fixed effects are obtained after regressing the residuals of the full model minus the residuals of the partialed out models on `D` using the package [FixedEffects.jl](https://github.com/matthieugomez/FixedEffects.jl)

# References

Baum, C. and Schaffer, M. (2013) *AVAR: Stata module to perform asymptotic covariance estimation for iid and non-iid data robust to heteroskedasticity, autocorrelation, 1- and 2-way clustering, and common cross-panel autocorrelated disturbances*. Statistical Software Components, Boston College Department of Economics.

Correia, S. (2014) *REGHDFE: Stata module to perform linear or instrumental-variable regression absorbing any number of high-dimensional fixed effects*. Statistical Software Components, Boston College Department of Economics.

Fong, DC. and Saunders, M. (2011) *LSMR: An Iterative Algorithm for Sparse Least-Squares Problems*.  SIAM Journal on Scientific Computing

Gaure, S. (2013) *OLS with Multiple High Dimensional Category Variables*. Computational Statistics and Data Analysis

Kleibergen, F, and Paap, R. (2006) *Generalized reduced rank tests using the singular value decomposition.* Journal of econometrics 

Kleibergen, F. and Schaffer, M.  (2007) *RANKTEST: Stata module to test the rank of a matrix using the Kleibergen-Paap rk statistic*. Statistical Software Components, Boston College Department of Economics.




