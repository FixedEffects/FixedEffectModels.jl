
using FixedEffectModels, CSV, DataFrames, LinearAlgebra, Test
df = CSV.read(joinpath(dirname(@__FILE__), "..", "dataset", "Cigar.csv"))
df[:id1] = df[:State]
df[:id2] = df[:Year]
df[:pid1] = categorical(df[:id1])
df[:ppid1] = categorical(div.(df[:id1], 10))
df[:pid2] = categorical(df[:id2])
df[:y] = df[:Sales]
df[:x1] = df[:Price]
df[:z1] = df[:Pimin]
df[:x2] = df[:NDI]
df[:w] = df[:Pop]

##############################################################################
##
## coefficients
##
##############################################################################

# simple
m = @model y ~ x1
x = reg(df, m)
@test coef(x) ≈ [139.73446,-0.22974] atol = 1e-4
m = @model y ~ x1 weights = w
x = reg(df, m)
@test coef(x) ≈ [137.72495428982756,-0.23738] atol = 1e-4

df[:SalesInt] = round.(Int64, df[:Sales])
m = @model SalesInt ~ Price
x = reg(df, m)
@test coef(x) ≈ [139.72674,-0.2296683205] atol = 1e-4



# absorb
m = @model y ~ x1 fe = pid1
x = reg(df, m)
@test coef(x) ≈ [-0.20984] atol = 1e-4
@test x.iterations == 1

m = @model y ~ x1 fe = pid1 + pid2
x = reg(df, m)
@test coef(x)  ≈ [-1.08471] atol = 1e-4
m = @model y ~ x1 fe = pid1 + pid1&id2
x = reg(df, m)
@test coef(x) ≈   [-0.53470] atol = 1e-4
m = @model y ~ x1 fe = pid1*id2
x = reg(df, m)
@test coef(x) ≈   [-0.53470] atol = 1e-4
#@test isempty(coef(reg(df, @formula(y ~ 0), @fe(pid1*x1))))
m = @model y ~ x1 fe = ppid1&pid2
x = reg(df, m)
@test coef(x)  ≈  [-1.44255] atol = 1e-4

m = @model y ~ x1 fe = pid1&id2
x = reg(df, m)
@test coef(x) ≈ [13.993028174622104,-0.5804357763515606] atol = 1e-4
m = @model y ~ x1 fe = id2&pid1
x = reg(df, m)
@test coef(x) ≈ [13.993028174622104,-0.5804357763515606] atol = 1e-4
m = @model y ~ 1 fe = id2&pid1
x = reg(df, m)
@test coef(x) ≈ [174.4084407796102] atol = 1e-4

m = @model y ~ x1 fe = pid1&id2 + pid2&id1
x = reg(df, m)
@test coef(x) ≈ [51.2359,- 0.5797] atol = 1e-4
m = @model y ~ x1 + x2 fe = pid1&id2 + pid2&id1
x = reg(df, m)
@test coef(x) ≈ [-46.4464,-0.2546, -0.005563] atol = 1e-4
m = @model y ~ 0 + x1 + x2 fe = pid1&id2 + pid2&id1
x = reg(df, m)
@test coef(x) ≈  [-0.21226562244177932,-0.004775616634862829] atol = 1e-4


# recheck these two below
m = @model y ~ z1 fe = x1&x2&pid1
x = reg(df, m)
@test coef(x)  ≈  [122.98713, 0.30933] atol = 1e-4
m = @model y ~ z1 fe = (x1&x2)*pid1
x = reg(df, m)
@test coef(x) ≈   [ 0.421406] atol = 1e-4

# only one intercept
m = @model y ~ 1 fe = pid1 + pid2
x = reg(df, m)

# absorb + weights
m = @model y ~ x1 fe = pid1 weights = w
x = reg(df, m)
@test coef(x) ≈ [- 0.21741] atol = 1e-4
m = @model y ~ x1 fe = pid1 + pid2 weights = w
x = reg(df, m)
@test coef(x) ≈ [- 0.88794] atol = 1e-3
m = @model y ~ x1 fe = pid1 + pid1&id2 weights = w
x = reg(df, m)
@test coef(x) ≈ [- 0.461085492] atol = 1e-4

# iv
m = @model y ~ (x1 ~ z1)
x = reg(df, m)
@test coef(x) ≈ [138.19479,- 0.20733] atol = 1e-4
m = @model y ~ x2 + (x1 ~ z1)
x = reg(df, m)
@test coef(x) ≈  [137.45096,0.00516,- 0.76276] atol = 1e-4
m = @model y ~ x2 + (x1 ~ z1 + w)
x = reg(df, m)
@test coef(x) ≈ [137.57335,0.00534,- 0.78365] atol = 1e-4
## multiple endogeneous variables
m = @model y ~ (x1 + x2 ~ z1 + w)
x = reg(df, m)
@test coef(x) ≈ [139.544, .8001, -.00937] atol = 1e-4
m = @model y ~ 1 + (x1 + x2 ~ z1 + w)
x = reg(df, m)
@test coef(x) ≈ [139.544, .8001, -.00937] atol = 1e-4
result = [196.576, 0.00490989, -2.94019, -3.00686, -2.94903, -2.80183, -2.74789, -2.66682, -2.63855, -2.52394, -2.34751, -2.19241, -2.18707, -2.09244, -1.9691, -1.80463, -1.81865, -1.70428, -1.72925, -1.68501, -1.66007, -1.56102, -1.43582, -1.36812, -1.33677, -1.30426, -1.28094, -1.25175, -1.21438, -1.16668, -1.13033, -1.03782]
m = @model y ~ x2 + (x1&pid2 ~ z1&pid2)
x = reg(df, m)
@test coef(x) ≈ result atol = 1e-4

# iv + weight
m = @model y ~ (x1 ~ z1) weights = w
x = reg(df, m)
@test coef(x) ≈ [137.03637,- 0.22802] atol = 1e-4

# iv + weight + absorb
m = @model y ~ (x1 ~ z1) fe = pid1
x = reg(df, m)
@test coef(x) ≈ [-0.20284] atol = 1e-4
m = @model y ~ (x1 ~ z1) fe = pid1 weights = w
x = reg(df, m)
@test coef(x) ≈ [-0.20995] atol = 1e-4
m = @model y ~ x2 + (x1 ~ z1)
x = reg(df, m)
@test coef(x) ≈  [137.45096580480387,0.005169677634275297,-0.7627670265757879] atol = 1e-4
m = @model y ~ x2 + (x1 ~ z1) fe = pid1
x = reg(df, m)
@test coef(x)  ≈  [0.0011021722526916768,-0.3216374943695231] atol = 1e-4

# non high dimensional factors
m = @model y ~ x1 + pid2
x = reg(df, m)
m = @model y ~  pid2 fe = pid1
x = reg(df, m)
m = @model y ~ x1 + pid2 fe = pid1
x = reg(df, m)
@test coef(x)[1] ≈ -1.08471 atol = 1e-4
m = @model y ~ x1 + pid2 fe = pid1 weights = w
x = reg(df, m)
@test coef(x)[1] ≈ -0.88794 atol = 1e-4
m = @model y ~ x2 + (x1 ~ z1) + pid2 fe = pid1
x = reg(df, m)
@test coef(x)[1] ≈ -0.00525 atol = 1e-4


# * and & are supported
#m = @model y ~ x1*pid2
#x = reg(df, m)






##############################################################################
##
## collinearity
## add more tests
##
##############################################################################
# ols
df[:x12] = df[:x1]
m = @model y ~ x1 + x12
x = reg(df, m)
@test coef(x) ≈ [139.7344639806166,-0.22974688593485126,0.0] atol = 1e-4

# iv
df[:x22] = df[:x2]
m = @model y ~ x22 + x2 + (x1 ~ z1)
x = reg(df, m)
@test coef(x)[2] == 0 || coef(x)[3] == 0

df[:zz1] = df[:z1]
m = @model y ~ zz1 + (x1 ~ x2 + z1)
x = reg(df, m)
@test coef(x)[2] != 0.0

# catch when IV underidentified : re-try when 0.5
#@test_throws ErrorException reg(df, @formula(y ~ x1 + (x2 + w = x2)))

# catch continuous variables in fixed effects
@test_throws ErrorException reg(df, @model(y ~ x1, fe = x2))
@test_throws ErrorException reg(df, @model(y ~ x1, fe = x2 + pid1))



# Make sure all coefficients are estimated
using Random
Random.seed!(0)
df_r = DataFrame(x1 = randn(10000) * 100)
df_r[:x2] = df_r[:x1].^4
result = reg(df_r, @model(x1 ~ x2 ))
@test sum(abs.(coef(result)) .> 0)  == 2

##############################################################################
##
## std errors
##
##############################################################################

# Simple - matches stata
m = @model y ~ x1
x = reg(df, m)
@test stderror(x) ≈ [1.521269, 0.0188963] atol = 1e-6
# Stata ivreg - ivreg2 discrepancy - matches with df_add=-2
m = @model y ~ (x1 ~ z1)
x = reg(df, m)
@test stderror(x) ≈ [1.53661, 0.01915] atol = 1e-4
# Stata areg
m = @model y ~ x1 fe = pid1
x = reg(df, m)
@test stderror(x) ≈  [0.0098003] atol = 1e-7

# White
# Stata reg - matches stata
m = @model y ~ x1 vcov = robust
x = reg(df, m)
@test stderror(x) ≈ [1.686791, 0.0167042] atol = 1e-6
# Stata ivreg - ivreg2 discrepancy - matches with df_add=-2
m = @model y ~ (x1 ~ z1) vcov = robust
x = reg(df, m)
@test stderror(x) ≈ [1.63305, 0.01674] atol = 1e-4
# Stata areg - matches areg
m = @model y ~ x1 fe = pid1 vcov = robust
x = reg(df, m)
@test stderror(x) ≈ [0.0110005] atol = 1e-7

# Clustering models
# cluster - matches stata
m = @model y ~ x1 vcov = cluster(pid1)
x = reg(df, m)
@test stderror(x)[2] ≈ 0.0379228 atol = 1e-7
# cluster with fe - matches areg & reghdfe
m = @model y ~ x1 fe = pid1 vcov = cluster(pid2)
x = reg(df, m)
@test stderror(x) ≈ [0.0220563] atol = 1e-7
# stata reghxe - matches reghdfe (not areg)
m = @model y ~ x1 fe = pid1 vcov = cluster(pid1)
x = reg(df, m)
@test stderror(x) ≈ [0.0357498] atol = 1e-7
# iv + fe + cluster - matches ivreghdfe
m = @model y ~ x2 + (x1 ~z1) fe = pid1 vcov = cluster(pid1)
x = reg(df, m)
@test stderror(x) ≈ [0.0019704, 0.1893396] atol = 1e-7
# iv + fe + cluster + weights - matches ivreghdfe
m = @model y ~ x2 + (x1 ~z1) fe = pid1 vcov = cluster(pid1) weights = w
x = reg(df, m)
@test stderror(x) ≈ [0.000759, 0.070836] atol = 1e-6
# iv + fe + cluster + weights - matches ivreghdfe
m = @model y ~ (x1 ~z1) fe = pid1 vcov = cluster(pid1) weights = w
x = reg(df, m)
@test stderror(x) ≈ [0.0337439] atol = 1e-7
# multiway clustering - matches reghdfe
m = @model y ~ x1 vcov = cluster(pid1 + pid2)
x = reg(df, m)
@test stderror(x) ≈ [6.196362, 0.0403469] atol = 1e-6
# fe + multiway clustering - matches reghdfe
m = @model y ~ x1 fe = pid1 vcov = cluster(pid1 + pid2)
x = reg(df, m)
@test stderror(x) ≈ [0.0405335] atol = 1e-7
# fe + clustering on interactions - matches reghdfe
m = @model y ~ x1 fe = pid1 vcov = cluster(pid1&pid2)
x = reg(df, m)
@test stderror(x) ≈ [0.0110005] atol = 1e-7
# fe partially nested in interaction clusters - matches reghdfe
m=@model y ~ x1 fe = pid1 vcov=cluster(pid1&pid2)
x = reg(df, m)
@test stderror(x) ≈ [0.0110005] atol = 1e-7
# regressor partially nested in interaction clusters - matches reghdfe
m=@model y ~ x1 + pid1 vcov=cluster(pid1&pid2)
x = reg(df, m)
@test stderror(x)[1:2] ≈ [3.032187, 0.0110005] atol=1e-5

# catch continuous variable in cluster
@test_throws ErrorException reg(df, @model(y ~ x1, vcov = cluster(State)))

##############################################################################
##
## subset
##
##############################################################################
m = @model y ~ x1 + pid1
x0 = reg(df[df[:State] .<= 30, :], m)

# categorical variable as
m = @model y ~ x1 + pid1 subset = (State .<= 30)
x1 = reg(df, m)
@test length(x1.esample) == size(df, 1)
@test coef(x0) ≈ coef(x1) atol = 1e-4
@test vcov(x0) ≈ vcov(x1) atol = 1e-4


df[:id1_missing] = ifelse.(df[:id1] .<= 30, df[:id1], missing)
df[:pid1_missing] = categorical(df[:id1_missing])
m = @model y ~ x1 + pid1_missing
x2 = reg(df, m)
@test length(x2.esample) == size(df, 1)
@test coef(x0) ≈ coef(x2) atol = 1e-4
@test vcov(x0) ≈ vcov(x2) atol = 1e-2



m = @model y ~ x1 fe = pid1 subset = (State .<= 30)
x3 = reg(df, m)
@test length(x3.esample) == size(df, 1)
@test coef(x0)[2] ≈ coef(x3)[1] atol = 1e-4

m = @model y ~ x1 fe = pid1_missing
x4 = reg(df, m)
@test coef(x0)[2] ≈ coef(x4)[1] atol = 1e-4



# categorical variable as fixed effects
m = @model y ~ x1 fe = pid1 subset = (State .>= 30)
x5 = reg(df, m)


#Error reported by Erik
m = @model y ~ z1 + CPI vcov = cluster(pid1) subset = (State .>= 30)
x = reg(df, m)
@test diag(x.vcov) ≈ [130.7464887, 0.0257875, 0.0383939] atol = 1e-4



##############################################################################
##
## R2
##
##############################################################################
m = @model y ~ x1
x = reg(df, m)
@test r2(x) ≈ 0.0969 atol = 1e-4
@test adjr2(x) ≈ 0.09622618 atol = 1e-4

##############################################################################
##
## F Stat
##
##############################################################################
m = @model y ~ x1
x = reg(df, m)
@test x.F  ≈ 147.82425 atol = 1e-4
m = @model y ~ x1 fe = pid1
x = reg(df, m)
@test x.F  ≈ 458.45825 atol = 1e-4
m = @model y ~ (x1 ~ z1)
x = reg(df, m)
@test x.F  ≈ 117.17329 atol = 1e-4

m = @model y ~ x1 vcov = cluster(pid1)
x = reg(df, m)
@test x.F    ≈ 36.70275 atol = 1e-4
m = @model y ~ (x1 ~ z1) vcov = cluster(pid1)
x = reg(df, m)
@test x.F  ≈ 39.67227 atol = 1e-4
#  xtivreg2
m = @model y ~ (x1 ~ z1) fe = pid1
x = reg(df, m)
@test x.F  ≈ 422.46444 atol = 1e-4

##############################################################################
##
## F_kp r_kp statistics for IV. Difference degrees of freedom.
##
##############################################################################
m = @model y ~ (x1 ~ z1)
x = reg(df, m)
@test x.F_kp ≈ 52248.79247 atol = 1e-4
m = @model y ~ x2 + (x1 ~ z1)
x = reg(df, m)
@test x.F_kp ≈ 5159.812208193612 atol = 1e-4
m = @model y ~ (x1 ~ z1 + CPI)
x = reg(df, m)
@test x.F_kp   ≈ 27011.44080 atol = 1e-4
m = @model y ~ (x1 ~ z1) fe = pid1
x = reg(df, m)
@test x.F_kp ≈ 100927.75710 atol = 1e-4

# exactly same with ivreg2
m = @model y ~ (x1 ~ z1) vcov = robust
x = reg(df, m)
@test x.F_kp ≈ 23160.06350 atol = 1e-4
m = @model y ~ (x1 ~ z1) fe = pid1 vcov = robust
x = reg(df, m)
@test x.F_kp  ≈ 37662.82808 atol = 1e-4
m = @model y ~ CPI + (x1 ~ z1)  vcov = robust
x = reg(df, m)
@test x.F_kp ≈ 2093.46609 atol = 1e-4
m = @model y ~ (x1 ~ z1 + CPI) vcov = robust
x = reg(df, m)
@test x.F_kp  ≈ 16418.21196 atol = 1e-4



# like in ivreg2 but += 5 difference. combination iv difference, degrees of freedom difference?
# iv + cluster - F_kp varies from ivreg2 about 7e-4 (SEs off by more than 2 df)
m = @model y ~ (x1 ~ z1) vcov = cluster(pid1)
x = reg(df, m)
@test x.F_kp     ≈ 7249.88606 atol = 1e-4
# iv + cluster - F_kp varies from ivreg2 about 5e-6 (SEs off by more than 2 df)
m = @model y ~ CPI + (x1 ~ z1) vcov = cluster(pid1)
x = reg(df, m)
@test x.F_kp   ≈ 538.40393 atol = 1e-4
# Modified test values below after multiway clustering update
# iv + 2way clustering - F_kp matches ivreg2 (SEs match with df_add=-2)
m = @model y ~ CPI + (x1 ~ z1) vcov = cluster(pid1 + pid2)
x = reg(df, m)
@test x.F_kp   ≈ 421.9651 atol = 1e-4
# multivariate iv + clustering - F_kp varies from ivreg2 about 3 (SEs off by more than 2 df)
m = @model y ~ (x1 ~ z1 + CPI) vcov = cluster(pid1)
x = reg(df, m)
@test x.F_kp      ≈ 4080.66081 atol = 1e-4
# multivariate iv + multiway clustering - F_kp varies from ivreg2 about 2 (SEs off by more than 2 df)
m = @model y ~ (x1 ~ z1 + CPI) vcov = cluster(pid1 + pid2)
x = reg(df, m)
@test x.F_kp  ≈ 2873.1405 atol = 1e-4



##############################################################################
##
## Test singleton
##
##
##############################################################################
df[:n] = max.(1:size(df, 1), 60)
df[:pn] = categorical(df[:n])
m = @model y ~ x1 fe = pn  vcov = cluster(pid1)
x = reg(df, m)
@test x.nobs == 60


m = @model y ~ x1 fe = pn  vcov = cluster(pid1)
x = reg(df, m, drop_singletons = false)
@test x.nobs == 1380

##############################################################################
##
## Test unbalanced panel
##
## corresponds to abdata in Stata, for instance reghxe wage emp [w=indoutpt], a(id year)
##
##############################################################################
df = CSV.read(joinpath(dirname(@__FILE__), "..", "dataset", "EmplUK.csv"))
df[:id1] = df[:Firm]
df[:id2] = df[:Year]
df[:pid1] = categorical(df[:id1])
df[:pid2] = categorical(df[:id2])
df[:y] = df[:Wage]
df[:x1] = df[:Emp]
df[:w] = df[:Output]

if Base.USE_GPL_LIBS
	method_s = [:cholesky, :qr, :lsmr, :lsmr_parallel, :lsmr_threads]
else
	method_s = [:lsmr, :lsmr_parallel, :lsmr_threads]
end

for method in method_s
	# absorb
	m = @model y ~ x1 fe = pid1 method = $(method)
	x = reg(df, m)
	@test coef(x) ≈ [- 0.11981270017206136] atol = 1e-4
	m = @model y ~ x1 fe = pid1&id2 method = $(method)
	x = reg(df, m)
	@test coef(x)  ≈ [-315.0000747500431,- 0.07633636891202833] atol = 1e-4
	m = @model y ~ x1 fe = id2&pid1 method = $(method)
	x = reg(df, m)
	@test coef(x) ≈  [-315.0000747500431,- 0.07633636891202833] atol = 1e-4
	m = @model y ~ 1 fe = id2&pid1 method = $(method)
	x = reg(df, m)
	@test coef(x) ≈  [- 356.40430526316396] atol = 1e-4
	m = @model y ~ x1 fe = pid1 weights = w method = $(method)
	x = reg(df, m)
	@test coef(x) ≈ [- 0.11514363590574725] atol = 1e-4

	# absorb + weights
	m = @model y ~ x1 fe = pid1 + pid2 method = $(method)
	x = reg(df, m)
	@test coef(x)  ≈  [- 0.04683333721137311] atol = 1e-4
	m = @model y ~ x1 fe = pid1 + pid2 weights = w method = $(method)
	x = reg(df, m)
	@test coef(x) ≈  [- 0.043475472188120416] atol = 1e-3

	## the last two ones test an ill conditioned model matrix
	m = @model y ~ x1 fe = pid1 + pid1&id2 method = $(method)
	x = reg(df, m)
	@test coef(x)  ≈   [- 0.122354] atol = 1e-4
	@test x.iterations <= 30

	m = @model y ~ x1 fe = pid1 + pid1&id2 weights = w method = $(method)
	x = reg(df, m)
	@test coef(x) ≈ [- 0.11752306001586807] atol = 1e-4
	@test x.iterations <= 50
end


# add tests with missing interaction. add tests with missing fixed effects
