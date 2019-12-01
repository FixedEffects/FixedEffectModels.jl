
using FixedEffectModels, CSV, DataFrames, LinearAlgebra, Test
df = CSV.read(joinpath(dirname(pathof(FixedEffectModels)), "../dataset/Cigar.csv"))
df.id1 = df.State
df.id2 = df.Year
df.pid1 = categorical(df.id1)
df.pid2 = categorical(df.id2)

df.mid1 = div.(df.id1, Ref(10))
df.y = df.Sales
df.x1 = df.Price
df.z1 = df.Pimin
df.x2 = df.NDI
df.w = df.Pop


##############################################################################
##
## coefficients
##
##############################################################################

# simple
m = @formula y ~ x1
x = reg(df, m)
@test coef(x) ≈ [139.73446,-0.22974] atol = 1e-4
m = @formula y ~ x1
x = reg(df, m, weights = :w)
@test coef(x) ≈ [137.72495428982756,-0.23738] atol = 1e-4

df.SalesInt = round.(Int64, df.Sales)
m = @formula SalesInt ~ Price
x = reg(df, m)
@test coef(x) ≈ [139.72674,-0.2296683205] atol = 1e-4



# absorb
m = @formula y ~ x1 + fe(id1)
x = reg(df, m)
@test coef(x) ≈ [-0.20984] atol = 1e-4
@test x.iterations == 1

m = @formula y ~ x1 + fe(id1) + fe(id2)
x = reg(df, m)
@test coef(x)  ≈ [-1.08471] atol = 1e-4
m = @formula y ~ x1 + fe(id1) + fe(id1)*id2
x = reg(df, m)
@test coef(x) ≈  [-0.53470, 0.0] atol = 1e-4
m = @formula y ~ x1 + fe(id1)*id2
x = reg(df, m)
@test coef(x) ≈  [-0.53470, 0.0] atol = 1e-4
#@test isempty(coef(reg(df, @formula(y ~ 0), @fe(id1*x1))))
m = @formula y ~ x1 + fe(mid1)&fe(id2)
x = reg(df, m)
@test coef(x)  ≈  [-1.44255] atol = 1e-4

m = @formula y ~ x1 + fe(id1)&id2
x = reg(df, m)
@test coef(x) ≈ [13.993028174622104,-0.5804357763515606] atol = 1e-4
m = @formula y ~ x1 + id2&fe(id1)
x = reg(df, m)
@test coef(x) ≈ [13.993028174622104,-0.5804357763515606] atol = 1e-4
m = @formula y ~ 1 + id2&fe(id1)
x = reg(df, m)
@test coef(x) ≈ [174.4084407796102] atol = 1e-4

m = @formula y ~ x1 + fe(id1)&id2 + fe(id2)&id1
x = reg(df, m)
@test coef(x) ≈ [51.2359,- 0.5797] atol = 1e-4
m = @formula y ~ x1 + x2 + fe(id1)&id2 + fe(id2)&id1
x = reg(df, m)
@test coef(x) ≈ [-46.4464,-0.2546, -0.005563] atol = 1e-4
m = @formula y ~ 0 + x1 + x2 + fe(id1)&id2 + fe(id2)&id1
x = reg(df, m)
@test coef(x) ≈  [-0.21226562244177932,-0.004775616634862829] atol = 1e-4


# recheck these two below
m = @formula y ~ z1 + x1&x2&fe(id1)
x = reg(df, m)
@test coef(x)  ≈  [122.98713, 0.30933] atol = 1e-4
m = @formula y ~ z1 + (x1&x2)*fe(id1)
x = reg(df, m)
@test coef(x) ≈   [0.421406, 0.0] atol = 1e-4

# only one intercept
m = @formula y ~ 1 + fe(id1) + fe(id2)
x = reg(df, m)





# TO DO: REPORT INTERCEPT IN CASE OF FIXED EFFFECTS, LIKE STATA
df.id3  = categorical(mod.(1:size(df, 1), Ref(3)))
df.id4  = categorical(div.(1:size(df, 1), Ref(10)))

m = @formula y ~ id3
x = reg(df, m)
@test length(coef(x)) == 3

m = @formula y ~ 0 + id3
x = reg(df, m)
@test length(coef(x)) == 3


# with fixed effects it's like implicit intercept
m = @formula y ~  id3 + fe(id4)
x = reg(df, m)
@test length(coef(x)) == 2


m = @formula y ~ id2 + fe(id1)
x = reg(df, m)


m = @formula y ~ id3&x1
x = reg(df, m)
@test length(coef(x)) == 4
m = @formula y ~ id3&x1 + x1
x = reg(df, m)
@test length(coef(x)) == 4





m = @formula y ~ id2 + fe(id1)
x = reg(df, m)



m = @formula y ~ id2&x1 + fe(id1)
x = reg(df, m)




# absorb + weights
m = @formula y ~ x1 + fe(id1)
x = reg(df, m, weights = :w)
@test coef(x) ≈ [- 0.21741] atol = 1e-4
m = @formula y ~ x1 + fe(id1) + fe(id2)
x = reg(df, m, weights = :w)
@test coef(x) ≈ [- 0.88794] atol = 1e-3
m = @formula y ~ x1 + fe(id1) + fe(id1)&id2
x = reg(df, m, weights = :w)
@test coef(x) ≈ [- 0.461085492] atol = 1e-4

# iv
m = @formula y ~ (x1 ~ z1)
x = reg(df, m)
@test coef(x) ≈ [138.19479,- 0.20733] atol = 1e-4
m = @formula y ~ x2 + (x1 ~ z1)
x = reg(df, m)
@test coef(x) ≈  [137.45096,0.00516,- 0.76276] atol = 1e-4
m = @formula y ~ x2 + (x1 ~ z1 + w)
x = reg(df, m)
@test coef(x) ≈ [137.57335,0.00534,- 0.78365] atol = 1e-4
## multiple endogeneous variables
m = @formula y ~ (x1 + x2 ~ z1 + w)
x = reg(df, m)
@test coef(x) ≈ [139.544, .8001, -.00937] atol = 1e-4
m = @formula y ~ 1 + (x1 + x2 ~ z1 + w)
x = reg(df, m)
@test coef(x) ≈ [139.544, .8001, -.00937] atol = 1e-4
result = [196.576, 0.00490989, -2.94019, -3.00686, -2.94903, -2.80183, -2.74789, -2.66682, -2.63855, -2.52394, -2.34751, -2.19241, -2.18707, -2.09244, -1.9691, -1.80463, -1.81865, -1.70428, -1.72925, -1.68501, -1.66007, -1.56102, -1.43582, -1.36812, -1.33677, -1.30426, -1.28094, -1.25175, -1.21438, -1.16668, -1.13033, -1.03782]

m = @formula y ~ x2 + (x1&pid2 ~ z1&pid2)
x = reg(df, m)
@test coef(x) ≈ result atol = 1e-4

# iv + weight
m = @formula y ~ (x1 ~ z1)
x = reg(df, m,  weights = :w)
@test coef(x) ≈ [137.03637,- 0.22802] atol = 1e-4

# iv + weight + absorb
m = @formula y ~ (x1 ~ z1) + fe(id1)
x = reg(df, m)
@test coef(x) ≈ [-0.20284] atol = 1e-4
m = @formula y ~ (x1 ~ z1) + fe(id1)
x = reg(df, m,  weights = :w)
@test coef(x) ≈ [-0.20995] atol = 1e-4
m = @formula y ~ x2 + (x1 ~ z1)
x = reg(df, m)
@test coef(x) ≈  [137.45096580480387,0.005169677634275297,-0.7627670265757879] atol = 1e-4
m = @formula y ~ x2 + (x1 ~ z1) + fe(id1)
x = reg(df, m)
@test coef(x)  ≈  [0.0011021722526916768,-0.3216374943695231] atol = 1e-4

# non high dimensional factors
m = @formula y ~ x1 + pid2
x = reg(df, m)
m = @formula y ~ pid2 + fe(id1)
x = reg(df, m)
m = @formula y ~ x1 + pid2 + fe(id1)
x = reg(df, m)
@test coef(x)[1] ≈ -1.08471 atol = 1e-4
m = @formula y ~ x1 + pid2 + fe(id1)
x = reg(df, m,  weights = :w)
@test coef(x)[1] ≈ -0.88794 atol = 1e-4
m = @formula y ~ x2 + (x1 ~ z1) + pid2 + fe(id1)
x = reg(df, m)
@test coef(x)[1] ≈ -0.00525 atol = 1e-4




##############################################################################
##
## Programming
##
##############################################################################
using StatsModels
reg(df, Term(:Sales) ~ Term(:NDI) + fe(Term(:State)) + fe(Term(:Year)), Vcov.cluster(:State))

# old syntax
m = @model y ~ x1 fe = pid1
x = reg(df, m)

m = @model y ~ x1 vcov = cluster(State + Year)
x = reg(df, m)

##############################################################################
##
## Functions
##
##############################################################################

# function
m = @formula y ~ log(x1)
x = reg(df, m)
@test coef(x)[1] ≈184.98520688 atol = 1e-4

# function defined in user space
mylog(x) = log(x)
m = @formula y ~ mylog(x1)
x = reg(df, m)

# Function returning Inf
df.x1_zero = copy(df.x1)
df.x1_zero[1] = 0.0
m = @formula y ~ log(x1_zero)
@test_throws  "Some observations for the regressor are infinite" x = reg(df, m)
##############################################################################
##
## collinearity
## add more tests
##
##############################################################################
# ols
df.x12 = df.x1
m = @formula y ~ x1 + x12
x = reg(df, m)
@test coef(x) ≈ [139.7344639806166,-0.22974688593485126,0.0] atol = 1e-4

# iv
df.x22 = df.x2
m = @formula y ~ x22 + x2 + (x1 ~ z1)
x = reg(df, m)
@test coef(x)[2] == 0 || coef(x)[3] == 0

df.zz1 = df.z1
m = @formula y ~ zz1 + (x1 ~ x2 + z1)
x = reg(df, m)
@test coef(x)[2] != 0.0

# catch when IV underidentified 
@test_throws "Model not identified. There must be at least as many ivs as endogeneneous variables" reg(df, @formula(y ~ x1 + (x2 + w ~ x2)))




# Make sure all coefficients are estimated
using Random
Random.seed!(0)
df_r = DataFrame(x1 = randn(10000) * 100)
df_r.x2 = df_r.x1.^4
result = reg(df_r, @formula(x1 ~ x2 ))
@test sum(abs.(coef(result)) .> 0)  == 2

##############################################################################
##
## std errors
##
##############################################################################

# Simple - matches stata
m = @formula y ~ x1
x = reg(df, m)
@test stderror(x) ≈ [1.521269, 0.0188963] atol = 1e-6
# Stata ivreg - ivreg2 discrepancy - matches with df_add=-2
m = @formula y ~ (x1 ~ z1)
x = reg(df, m)
@test stderror(x) ≈ [1.53661, 0.01915] atol = 1e-4
# Stata areg
m = @formula y ~ x1 + fe(id1)
x = reg(df, m)
@test stderror(x) ≈  [0.0098003] atol = 1e-7

# White
# Stata reg - matches stata
m = @formula y ~ x1
x = reg(df, m, Vcov.robust())
@test stderror(x) ≈ [1.686791, 0.0167042] atol = 1e-6
# Stata ivreg - ivreg2 discrepancy - matches with df_add=-2
m = @formula y ~ (x1 ~ z1)
x = reg(df, m, Vcov.robust())
@test stderror(x) ≈ [1.63305, 0.01674] atol = 1e-4
# Stata areg - matches areg
m = @formula y ~ x1 + fe(id1)
x = reg(df, m, Vcov.robust())
@test stderror(x) ≈ [0.0110005] atol = 1e-7

# Clustering models
# cluster - matches stata
m = @formula y ~ x1
x = reg(df, m, Vcov.cluster(:id1))
@test stderror(x)[2] ≈ 0.0379228 atol = 1e-7
# cluster with fe - matches areg & reghdfe
m = @formula y ~ x1 + fe(id1)
x = reg(df, m, Vcov.cluster(:id2))
@test stderror(x) ≈ [0.0220563] atol = 1e-7
# stata reghxe - matches reghdfe (not areg)
m = @formula y ~ x1 + fe(id1)
x = reg(df, m, Vcov.cluster(:id1))
@test stderror(x) ≈ [0.0357498] atol = 1e-7
# iv + fe + cluster - matches ivreghdfe
m = @formula y ~ x2 + (x1 ~z1) + fe(id1)
x = reg(df, m, Vcov.cluster(:id1))
@test stderror(x) ≈ [0.0019704, 0.1893396] atol = 1e-7
# iv + fe + cluster + weights - matches ivreghdfe
m = @formula y ~ x2 + (x1 ~ z1) + fe(id1)
x = reg(df, m,  Vcov.cluster(:id1), weights = :w)
@test stderror(x) ≈ [0.000759, 0.070836] atol = 1e-6
# iv + fe + cluster + weights - matches ivreghdfe
m = @formula y ~ (x1 ~ z1) + fe(id1)
x = reg(df, m,  Vcov.cluster(:id1), weights = :w)
@test stderror(x) ≈ [0.0337439] atol = 1e-7
# multiway clustering - matches reghdfe
m = @formula y ~ x1
x = reg(df, m, Vcov.cluster(:id1, :id2))
@test stderror(x) ≈ [6.196362, 0.0403469] atol = 1e-6
# multiway clustering - matches reghdfe
m = @formula y ~ x1
x = reg(df, m, Vcov.cluster(:id1, :id2))
@test stderror(x) ≈ [6.196362, 0.0403469] atol = 1e-6
# fe + multiway clustering - matches reghdfe
m = @formula y ~ x1 + fe(id1)
x = reg(df, m, Vcov.cluster(:id1, :id2))
@test stderror(x) ≈ [0.0405335] atol = 1e-7
m = @formula y ~ x1 + fe(id1)
x = reg(df, m, Vcov.cluster(:id1, :id2))
@test stderror(x) ≈ [0.0405335] atol = 1e-7
# fe + clustering on interactions - matches reghdfe
#m = @formula y ~ x1 + fe(id1) vcov = cluster(id1&id2)
#x = reg(df, m)
#@test stderror(x) ≈ [0.0110005] atol = 1e-7
# fe partially nested in interaction clusters - matches reghdfe
#m=@formula y ~ x1 + fe(id1) vcov=cluster(id1&id2)
#x = reg(df, m)
#@test stderror(x) ≈ [0.0110005] atol = 1e-7
# regressor partially nested in interaction clusters - matches reghdfe
#m=@formula y ~ x1 + pid1 vcov=cluster(id1&id2)
#x = reg(df, m)
#@test stderror(x)[1:2] ≈ [3.032187, 0.0110005] atol=1e-5

#check palue printed is correct 
#m = @formula y ~ x1 + fe(id1)
#x = reg(df, m, Vcov.cluster(:id1))
#@test coeftable(x).mat[1, 4] ≈ 4.872723900371927e-7 atol = 1e-7


##############################################################################
##
## subset
##
##############################################################################
m = @formula y ~ x1 + pid1
x0 = reg(df[df.State .<= 30, :], m)

m = @formula y ~ x1 + pid1
x1 = reg(df, m, subset = df.State .<= 30)
@test length(x1.esample) == size(df, 1)
@test coef(x0) ≈ coef(x1) atol = 1e-4
@test vcov(x0) ≈ vcov(x1) atol = 1e-4


df.id1_missing = ifelse.(df.id1 .<= 30, df.id1, missing)
df.pid1_missing = categorical(df.id1_missing)
m = @formula y ~ x1 + pid1_missing
x2 = reg(df, m)
@test length(x2.esample) == size(df, 1)
@test coef(x0) ≈ coef(x2) atol = 1e-4
@test vcov(x0) ≈ vcov(x2) atol = 1e-2



m = @formula y ~ x1 + fe(id1)
x3 = reg(df, m, subset = df.State .<= 30)
@test length(x3.esample) == size(df, 1)
@test coef(x0)[2] ≈ coef(x3)[1] atol = 1e-4

m = @formula y ~ x1 + fe(id1_missing)
x4 = reg(df, m)
@test coef(x0)[2] ≈ coef(x4)[1] atol = 1e-4



# categorical variable as fixed effects
m = @formula y ~ x1 + fe(id1)
x5 = reg(df, m, subset = df.State .>= 30)


#Error reported by Erik
m = @formula y ~ z1 + CPI
x = reg(df, m, Vcov.cluster(:id1), subset = df.State .>= 30)
@test diag(x.vcov) ≈ [130.7464887, 0.0257875, 0.0383939] atol = 1e-4



##############################################################################
##
## R2
##
##############################################################################
m = @formula y ~ x1
x = reg(df, m)
@test r2(x) ≈ 0.0969 atol = 1e-4
@test adjr2(x) ≈ 0.09622618 atol = 1e-4

##############################################################################
##
## F Stat
##
##############################################################################
m = @formula y ~ x1
x = reg(df, m)
@test x.F  ≈ 147.82425 atol = 1e-4
m = @formula y ~ x1 + fe(id1)
x = reg(df, m)
@test x.F  ≈ 458.45825 atol = 1e-4
m = @formula y ~ (x1 ~ z1)
x = reg(df, m)
@test x.F  ≈ 117.17329 atol = 1e-4
m = @formula y ~ x1 + w
x = reg(df, m)
@test x.F ≈ 79.091576 atol = 1e-4
m = @formula y ~ x1
x = reg(df, m, Vcov.cluster(:id1))
@test x.F    ≈ 36.70275 atol = 1e-4
m = @formula y ~ (x1 ~ z1)
x = reg(df, m, Vcov.cluster(:id1))
@test x.F  ≈ 39.67227 atol = 1e-4
#  xtivreg2
m = @formula y ~ (x1 ~ z1) + fe(id1)
x = reg(df, m)
@test x.F  ≈ 422.46444 atol = 1e-4

# p value
m = @formula w ~ z1
x = reg(df, m)
@test x.p ≈ 0.003998718283554043 atol = 1e-4
m = @formula w ~ z1 + x1
x = reg(df, m)
@test x.p ≈ 3.369423076033613e-14 atol = 1e-4
##############################################################################
##
## F_kp r_kp statistics for IV. Difference degrees of freedom.
##
##############################################################################
m = @formula y ~ (x1 ~ z1)
x = reg(df, m)
@test x.F_kp ≈ 52248.79247 atol = 1e-4
m = @formula y ~ x2 + (x1 ~ z1)
x = reg(df, m)
@test x.F_kp ≈ 5159.812208193612 atol = 1e-4
m = @formula y ~ (x1 ~ z1 + CPI)
x = reg(df, m)
@test x.F_kp   ≈ 27011.44080 atol = 1e-4
m = @formula y ~ (x1 ~ z1) + fe(id1)
x = reg(df, m)
@test x.F_kp ≈ 100927.75710 atol = 1e-4

# exactly same with ivreg2
m = @formula y ~ (x1 ~ z1)
x = reg(df, m, Vcov.robust())
@test x.F_kp ≈ 23160.06350 atol = 1e-4
m = @formula y ~ (x1 ~ z1) + fe(id1)
x = reg(df, m, Vcov.robust())
@test x.F_kp  ≈ 37662.82808 atol = 1e-4
m = @formula y ~ CPI + (x1 ~ z1) 
x = reg(df, m, Vcov.robust())
@test x.F_kp ≈ 2093.46609 atol = 1e-4
m = @formula y ~ (x1 ~ z1 + CPI)
x = reg(df, m, Vcov.robust())
@test x.F_kp  ≈ 16418.21196 atol = 1e-4



# like in ivreg2 but += 5 difference. combination iv difference, degrees of freedom difference?
# iv + cluster - F_kp varies from ivreg2 about 7e-4 (SEs off by more than 2 df)
m = @formula y ~ (x1 ~ z1)
x = reg(df, m, Vcov.cluster(:id1))
@test x.F_kp     ≈ 7249.88606 atol = 1e-4
# iv + cluster - F_kp varies from ivreg2 about 5e-6 (SEs off by more than 2 df)
m = @formula y ~ CPI + (x1 ~ z1)
x = reg(df, m, Vcov.cluster(:id1))
@test x.F_kp   ≈ 538.40393 atol = 1e-4
# Modified test values below after multiway clustering update
# iv + 2way clustering - F_kp matches ivreg2 (SEs match with df_add=-2)
m = @formula y ~ CPI + (x1 ~ z1)
x = reg(df, m, Vcov.cluster(:id1, :id2))
@test x.F_kp   ≈ 421.9651 atol = 1e-4
# multivariate iv + clustering - F_kp varies from ivreg2 about 3 (SEs off by more than 2 df)
m = @formula y ~ (x1 ~ z1 + CPI)
x = reg(df, m, Vcov.cluster(:id1))
@test x.F_kp      ≈ 4080.66081 atol = 1e-4
# multivariate iv + multiway clustering - F_kp varies from ivreg2 about 2 (SEs off by more than 2 df)
m = @formula y ~ (x1 ~ z1 + CPI)
x = reg(df, m, Vcov.cluster(:id1, :id2))
@test x.F_kp  ≈ 2873.1405 atol = 1e-4



##############################################################################
##
## Test singleton
##
##
##############################################################################
df.n = max.(1:size(df, 1), 60)
df.pn = categorical(df.n)
m = @formula y ~ x1 + fe(pn)
x = reg(df, m, Vcov.cluster(:id1))
@test x.nobs == 60


m = @formula y ~ x1 + fe(pn)
x = reg(df, m, Vcov.cluster(:id1), drop_singletons = false)
@test x.nobs == 1380

##############################################################################
##
## Test unbalanced panel
##
## corresponds to abdata in Stata, for instance reghxe wage emp [w=indoutpt], a(id year)
##
##############################################################################
df = CSV.read(joinpath(dirname(pathof(FixedEffectModels)), "../dataset/EmplUK.csv"))
df.id1 = df.Firm
df.id2 = df.Year
df.y = df.Wage
df.x1 = df.Emp
df.w = df.Output


methods_vec = [:lsmr, :lsmr_threads, :lsmr_cores]


for method in methods_vec
	@show method
	m = @formula y ~ x1 + fe(id1)
	x = reg(df, m, method = method)
	@test coef(x) ≈ [- 0.11981270017206136] atol = 1e-4
	m = @formula y ~ x1 + fe(id1)&id2
	x = reg(df, m, method = method)
	@test coef(x)  ≈ [-315.0000747500431,- 0.07633636891202833] atol = 1e-4
	m = @formula y ~ x1 + id2&fe(id1)
	x = reg(df, m, method = method)
	@test coef(x) ≈  [-315.0000747500431,- 0.07633636891202833] atol = 1e-4
	m = @formula y ~ 1 + id2&fe(id1)
	x = reg(df, m, method = method)
	@test coef(x) ≈  [- 356.40430526316396] atol = 1e-4
	m = @formula y ~ x1 + fe(id1)
	x = reg(df, m, weights = :w,  method = method)
	@test coef(x) ≈ [- 0.11514363590574725] atol = 1e-4

	# absorb + weights
	m = @formula y ~ x1 + fe(id1) + fe(id2)
	x = reg(df, m, method = method)
	@test coef(x)  ≈  [- 0.04683333721137311] atol = 1e-4
	m = @formula y ~ x1 + fe(id1) + fe(id2) 
	x = reg(df, m, weights = :w, method = method)
	@test coef(x) ≈  [- 0.043475472188120416] atol = 1e-3

	## the last two ones test an ill conditioned model matrix
	m = @formula y ~ x1 + fe(id1) + fe(id1)&id2
	x = reg(df, m, method = method)
	@test coef(x)  ≈   [- 0.122354] atol = 1e-4
	@test x.iterations <= 30

	m = @formula y ~ x1 + fe(id1) + fe(id1)&id2
	x = reg(df, m, weights = :w, method = method)
	@test coef(x) ≈ [- 0.11752306001586807] atol = 1e-4
	@test x.iterations <= 50
end
if isdefined(FixedEffectModels.FixedEffects, :FixedEffectSolverLSMRGPU)
	push!(methods_vec, :lsmr_gpu)
end
for method in methods_vec
	# same thing with float32 precision
	m = @formula y ~ x1 + fe(id1)
	x = reg(df, m, method = method, double_precision = false)
	@test coef(x) ≈ [- 0.11981270017206136] rtol = 1e-4
	m = @formula y ~ x1 + fe(id1)&id2
	x = reg(df, m, method = method, double_precision = false)
	@test coef(x)  ≈ [-315.0000747500431,- 0.07633636891202833] rtol = 1e-4
	m = @formula y ~ x1 + id2&fe(id1)
	x = reg(df, m, method = method, double_precision = false)
	@test coef(x) ≈  [-315.0000747500431,- 0.07633636891202833] rtol = 1e-4
	m = @formula y ~ 1 + id2&fe(id1)
	x = reg(df, m, method = method, double_precision = false)
	@test coef(x) ≈  [- 356.40430526316396] rtol = 1e-4
	m = @formula y ~ x1 + fe(id1)
	x = reg(df, m, weights = :w, method = method, double_precision = false)
	@test coef(x) ≈ [- 0.11514363590574725] rtol = 1e-4
	m = @formula y ~ x1 + fe(id1) + fe(id2)
	x = reg(df, m, method = method, double_precision = false)
	@test coef(x)  ≈  [- 0.04683333721137311] rtol = 1e-4
	m = @formula y ~ x1 + fe(id1) + fe(id2)
	x = reg(df, m, weights = :w, method = method, double_precision = false)
	@test coef(x) ≈  [- 0.043475472188120416] atol = 1e-3
end


# add tests with missing fixed effects
df.id1_missing = ifelse.(df.id1 .<= 30, missing, df.id1)


## test with missing fixed effects
m = @formula y ~ x1 + fe(id1_missing)
x = reg(df, m)
@test coef(x) ≈ [-.1093657] atol = 1e-4
@test stderror(x) ≈  [.032949 ] atol = 1e-4
@test r2(x) ≈ 0.8703 atol = 1e-2
@test adjr2(x) ≈ 0.8502 atol = 1e-2
@test x.nobs == 821

## test with missing interaction
df.id3 = df.id2 .>= 1980
m = @formula y ~ x1 + fe(id1_missing) & fe(id3)
x = reg(df, m)
@test coef(x) ≈ [-0.100863] atol = 1e-4
@test stderror(x) ≈  [0.04149] atol = 1e-4
@test x.nobs == 821


##############################################################################
##
## Missing dependent / independent variables
##
##############################################################################


# Missing
df1 = DataFrame(a=[1.0, 2.0, 3.0, 4.0], b=[5.0, 7.0, 11.0, 13.0])
df2 = DataFrame(a=[1.0, missing, 3.0, 4.0], b=[5.0, 7.0, 11.0, 13.0])
x = reg(df1,@formula(a ~ b))
@test coef(x) ≈ [ -0.6500000000000004, 0.35000000000000003] atol = 1e-4
x = reg(df1,@formula(b ~ a))
@test coef(x) ≈ [2.0, 2.8] atol = 1e-4
x = reg(df2,@formula(a ~ b))
@test coef(x) ≈ [ -0.8653846153846163, 0.3653846153846155] atol = 1e-4
x = reg(df2,@formula(b ~ a))
@test coef(x) ≈ [ 2.4285714285714253, 2.7142857142857157] atol = 1e-4

# Works with Integers
df1 = DataFrame(a=[1, 2, 3, 4], b=[5, 7, 11, 13], c = categorical([1, 1, 2, 2]))
x = reg(df1,@formula(a ~ b))
@test coef(x) ≈ [-0.65, 0.35] atol = 1e-4
x = reg(df1,@formula(a ~ b + fe(c)))
@test coef(x) ≈ [0.5] atol = 1e-4


