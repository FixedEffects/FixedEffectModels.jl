[![FixedEffectModels](http://pkg.julialang.org/badges/FixedEffectModels_0.6.svg)](http://pkg.julialang.org/?pkg=FixedEffectModels)
[![Build Status](https://travis-ci.org/matthieugomez/FixedEffectModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffectModels.jl)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffectModels.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffectModels.jl?branch=master)

This package estimates linear models with high dimensional categorical variables and/or instrumental variables. 

Its objective is similar to the Stata command [`reghdfe`](https://github.com/sergiocorreia/reghdfe) and the R function [`felm`](https://cran.r-project.org/web/packages/lfe/lfe.pdf). The package is usually much faster than these two options. The package implements a novel algorithm, which combines projection methods with the conjugate gradient descent.

![benchmark](https://cdn.rawgit.com/matthieugomez/FixedEffectModels.jl/4c7d1db39377f1ee649624c909c9017f92484114/benchmark/result.svg)

To install the package, 

```julia
Pkg.add("FixedEffectModels")
```

## Estimate a model
To estimate a `@model`, specify  a formula with, eventually, a set of fixed effects with the argument `fe`, a way to compute standard errors with the argument `vcov`, and a weight variable with `weights`.

```julia
using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
df[:StatePooled] =  categorical(df[:State])
df[:YearPooled] =  categorical(df[:Year])
reg(df, @model(Sales ~ NDI, fe = StatePooled + YearPooled, weights = Pop, vcov = cluster(StatePooled)))
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
- A typical formula is composed of one dependent variable, exogeneous variables, endogeneous variables, and instrumental variables.
	```julia
	dependent variable ~ exogenous variables + (endogenous variables ~ instrumental variables)
	```

- Fixed effect variables are indicated with the keyword argument `fe`. They must be of type PooledDataArray (use `pool` to convert a variable to a `PooledDataArray`).

	```julia
	df[:StatePooled] =  categorical(df[:State])
	# one high dimensional fixed effect
	fe = StatePooled
	```
	You can add an arbitrary number of high dimensional fixed effects, separated with `+`
	```julia
	df[:YearPooled] =  categorical(df[:Year])
	fe = StatePooled + YearPooled
	```
	Interact multiple categorical variables using `&` 
	```julia
	fe = StatePooled&DecPooled
	```
	Interact a categorical variable with a continuous variable using `&`
	```julia
	fe = StatePooled + StatePooled&Year
	```
	Alternative, use `*` to add a categorical variable and its interaction with a continuous variable
	```julia
	fe = StatePooled*Year
	# equivalent to fe = StatePooled + StatePooled&year
	```

- Standard errors are indicated with the keyword argument `vcov`.
	```julia
	vcov = robust
	vcov = cluster(StatePooled)
	vcov = cluster(StatePooled + YearPooled)
	```

- weights are indicated with the keyword argument `weights`
	```julia
	weights = Pop
	```

Arguments of `@model` are captured and transformed into expressions. If you want to program with `@model`, use expression interpolations:
```julia
using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
w = :Pop
reg(df, @model(Sales ~ NDI, weights = $(w)))
```

## Output
`reg` returns a light object. It is composed of 
 
  - the vector of coefficients & the covariance matrix
  - a boolean vector reporting rows used in the estimation
  - a set of scalars (number of observations, the degree of freedoms, r2, etc)
  - with the option `save = true`, a dataframe aligned with the initial dataframe with residuals and, if the model contains high dimensional fixed effects, fixed effects estimates.




Methods such as `predict`, `residuals` are still defined but require to specify a dataframe as a second argument.  The problematic size of `lm` and `glm` models in R or Julia is discussed [here](http://www.r-bloggers.com/trimming-the-fat-from-glm-models-in-r/), [here](https://blogs.oracle.com/R/entry/is_the_size_of_your), [here](http://stackoverflow.com/questions/21896265/how-to-minimize-size-of-object-of-class-lm-without-compromising-it-being-passe) [here](http://stackoverflow.com/questions/15260429/is-there-a-way-to-compress-an-lm-class-for-later-prediction) (and for absurd consequences, [here](http://stackoverflow.com/questions/26010742/using-stargazer-with-memory-greedy-glm-objects) and [there](http://stackoverflow.com/questions/22577161/not-enough-ram-to-run-stargazer-the-normal-way)).


You may use [RegressionTables.jl](https://github.com/jmboehm/RegressionTables.jl) to get publication-quality regression tables.


## Solution Method
Denote the model `y = X β + D θ + e` where X is a matrix with few columns and D is the design matrix from categorical variables. Estimates for `β`, along with their standard errors, are obtained in two steps:

1. `y, X`  are regressed on `D` by one of these methods
  - [MINRES on the normal equation](http://web.stanford.edu/group/SOL/software/lsmr/) with `method = lsmr` (with a diagonal preconditioner).
  - sparse factorization with `method = cholesky` or `method = qr` (using the SuiteSparse library)

  The default method`lsmr`, should be the fastest in most cases. If the method does not converge, frist please get in touch, I'd be interested to hear about your problem.  Second use the `method = cholesky`, which should do the trick.

2.  Estimates for `β`, along with their standard errors, are obtained by regressing the projected `y` on the projected `X` (an application of the Frisch Waugh-Lovell Theorem)

3. With the option `save = true`, estimates for the high dimensional fixed effects are obtained after regressing the residuals of the full model minus the residuals of the partialed out models on `D`



# References

Baum, C. and Schaffer, M. (2013) *AVAR: Stata module to perform asymptotic covariance estimation for iid and non-iid data robust to heteroskedasticity, autocorrelation, 1- and 2-way clustering, and common cross-panel autocorrelated disturbances*. Statistical Software Components, Boston College Department of Economics.

Correia, S. (2014) *REGHDFE: Stata module to perform linear or instrumental-variable regression absorbing any number of high-dimensional fixed effects*. Statistical Software Components, Boston College Department of Economics.

Fong, DC. and Saunders, M. (2011) *LSMR: An Iterative Algorithm for Sparse Least-Squares Problems*.  SIAM Journal on Scientific Computing

Gaure, S. (2013) *OLS with Multiple High Dimensional Category Variables*. Computational Statistics and Data Analysis

Kleibergen, F, and Paap, R. (2006) *Generalized reduced rank tests using the singular value decomposition.* Journal of econometrics 

Kleibergen, F. and Schaffer, M.  (2007) *RANKTEST: Stata module to test the rank of a matrix using the Kleibergen-Paap rk statistic*. Statistical Software Components, Boston College Department of Economics.




