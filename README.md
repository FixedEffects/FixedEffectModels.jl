[![Coverage Status](https://coveralls.io/repos/matthieugomez/FixedEffectModels.jl/badge.svg?branch=master)](https://coveralls.io/r/matthieugomez/FixedEffects.jl?branch=master)
[![Build Status](https://travis-ci.org/matthieugomez/FixedEffectModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/FixedEffects.jl)

The function `reg` estimates linear models with 
- instrumental variables (via 2SLS)
- high dimensional categorical variable (multiple intercept and slope fixed effects)
- robust standard errors (White or clustered) 

Its functionality corresponds roughly to the commands `reghdfe` in Stata and `lfe` in R.

## Formula Syntax

The general syntax is

```julia
reg(depvar ~ exogenousvars + (endogeneousvars = instrumentvars) |> absorbvars, df)
```


```julia
using  RDatasets, DataFrames, FixedEffectModels
df = dataset("plm", "Cigar")
df[:pState] =  pool(df[:State])
df[:pYear] =  pool(df[:Year])
reg(Sales ~ NDI |> pState, df)
#                         Fixed Effect Model                         
#=====================================================================
#Dependent variable          Sales   Number of obs                1380
#Degree of freedom              47   R2                          0.207
#R2 Adjusted                 0.179   F Statistics:             7.40264
#=====================================================================
#        Estimate  Std.Error  t value Pr(>|t|)   Lower 95%   Upper 95%
#---------------------------------------------------------------------
#NDI  -0.00170468 9.13903e-5 -18.6527    0.000 -0.00188396 -0.00152539
#=====================================================================
reg(Sales ~ NDI |> pState + pYear, df)
reg(Sales ~ NDI |> pState + pState&Year)
```


- Fixed effects must be variables of type PooledDataArray. Use the function `pool` to transform one column into a `PooledDataArray` and  `group` to combine multiple columns into a `PooledDataArray`.
- Interactions with a continuous variable can be specified with `&`





## Errors
Compute robust standart errors with a third argument

For now, `VcovSimple()` (default), `VcovWhite()` and `VcovCluster(cols)` are implemented.

```julia
reg(Sales ~ NDI, df, VcovWhite())
reg(Sales ~ NDI, df, VcovCluster([:State]))
reg(Sales ~ NDI, df, VcovCluster([:State, :Year]))
```


You can define your own type: after declaring it as a child of `AbstractVcov`, define a `vcov` methods for it. For instance,  White errors are implemented with the following code:

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





## Regression Result

`reg` returns a light object of type RegressionResult. It is simply composed of coefficients, covariance matrix, and some scalars like number of observations, degrees of freedoms, r2, etc. Usual methods `coef`, `vcov`, `nobs`, `predict`, `residuals` are defined.


## Comparison


Julia
```julia
using DataArrays, DataFrames, FixedEffectModels
N = 10000000
K = 100
df = DataFrame(
  v1 =  pool(rand(1:N/K, N)),
  v2 =  pool(rand(1:K, N)),
  v3 =  randn(N), 
  v4 =  randn(N),
  w =  abs(randn(N)) 
)
@time reg(v4 ~ v3, df)
# elapsed time: 1.22074119 seconds (1061288240 bytes allocated, 22.01% gc time)
@time reg(v4 ~ v3, df, weight = :w)
# elapsed time: 1.56727235 seconds (1240040272 bytes allocated, 15.59% gc time)
@time reg(v4 ~ v3 |> v1, df)
# elapsed time: 1.563452151 seconds (1269846952 bytes allocated, 17.99% gc time)
@time reg(v4 ~ v3 |> v1, df, weight = :w)
# elapsed time: 2.063922289 seconds (1448598696 bytes allocated, 17.96% gc time)
@time reg(v4 ~ v3 |> v1 + v2, df)
# elapsed time: 2.494780022 seconds (1283607248 bytes allocated, 18.87% gc time)
````

R (lfe package)
```R
library(lfe)
N = 10000000
K = 100
df = data.frame(
  v1 =  as.factor(sample(N/K, N, replace = TRUE)),
  v2 =  as.factor(sample(K, N, replace = TRUE)),
  v3 =  runif(N), 
  v4 =  runif(N), 
  w = abs(runif(N))
)
system.time(lm(v4 ~ v3, df))
#   user  system elapsed 
# 15.712   0.811  16.448 
system.time(lm(v4 ~ v3, df, w = w))
#   user  system elapsed 
# 10.416   0.995  11.474 
system.time(felm(v4 ~ v3|v1, df))
#   user  system elapsed 
# 19.971   1.595  22.112 
system.time(felm(v4 ~ v3|v1, df))
#   user  system elapsed 
# 19.971   1.595  22.112 
system.time(felm(v4 ~ v3|v1, df, w = w))
#   user  system elapsed 
# 19.971   1.595  22.112 
system.time(felm(v4 ~ v3|(v1 + v2), df))
#   user  system elapsed 
# 23.980   1.950  24.942 
```



Stata
```
clear all
local N = 10000000
local K = 100
set obs `N'
gen  v1 =  floor(runiform() * (`_N'+1)/`K')
gen  v2 =  floor(runiform() * (`K'+1))
gen  v3 =  runiform()
gen  v4 =  runiform()
gen  w =  abs(runiform()) 

timer clear

timer on 1
reg v4 v3
timer off 1

timer on 2
reg v4 v3 [w = w]
timer off 2

timer on 3
areg v4 v3, a(v1)
timer off 3

timer on 4
areg v4 v3 [w = w], a(v1)
timer off 4

timer on 5
reghdfe v4 v3, a(v1 v2)
timer off 5

. . timer list
   1:      1.61 /        1 =       1.4270
   2:      1.95 /        2 =       1.9510
   2:      5.04 /        1 =       5.0380
   3:      5.45 /        1 =       5.4530
   4:     67.24 /        1 =      67.2430
````



