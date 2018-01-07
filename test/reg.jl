using DataFrames, FixedEffectModels, Base.Test
#df = readtable("/Users/Matthieu/Dropbox/Github/FixedEffectModels.jl/dataset/Cigar.csv.gz")
df = readtable(joinpath(dirname(@__FILE__), "..", "dataset", "Cigar.csv.gz"))
df[:id1] = df[:State]
df[:id2] = df[:Year]
df[:pid1] = pool(df[:id1])
df[:ppid1] = pool(div(df[:id1], 10))
df[:pid2] = pool(df[:id2])
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
m = @model y ~ x1 + pid2 fe = pid1
x = reg(df, m)
@test coef(x)[1] ≈ -1.08471 atol = 1e-4
m = @model y ~ x1 + pid2 fe = pid1 weights = w
x = reg(df, m)
@test coef(x)[1] ≈ -0.88794 atol = 1e-4
m = @model y ~ x2 + (x1 ~ z1) + pid2 fe = pid1
x = reg(df, m)
@test coef(x)[1] ≈ -0.00525 atol = 1e-4

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



#colinearity with fixed effect
df[:dec] = pool(div(df[:id2], 10))
m = @model y ~ dec fe = pid2
x = reg(df, m)
@test coef(x)[1] ≈ 0.0
##############################################################################
##
## std errors
##
##############################################################################

# Simple
m = @model y ~ x1
x = reg(df, m)
@test stderr(x) ≈ [1.52126, 0.01889] atol = 1e-4
# Stata ivreg
m = @model y ~ (x1 ~ z1)
x = reg(df, m)
@test stderr(x) ≈ [1.53661, 0.01915] atol = 1e-4
# Stata areg
m = @model y ~ x1 fe = pid1
x = reg(df, m)
@test stderr(x) ≈  [0.00980] atol = 1e-4

# White
# Stata reg
m = @model y ~ x1 vcov = robust
x = reg(df, m)
@test stderr(x) ≈ [1.68679, 0.01670] atol = 1e-4
# Stata ivreg
m = @model y ~ (x1 ~ z1) vcov = robust
x = reg(df, m)
@test stderr(x) ≈ [1.63305, 0.01674] atol = 1e-4
# Stata areg
m = @model y ~ x1 fe = pid1 vcov = robust
x = reg(df, m)
@test stderr(x) ≈ [0.01100] atol = 1e-4

# cluster
m = @model y ~ x1 vcov = cluster(pid1)
x = reg(df, m)
@test stderr(x)[2] ≈ 0.03792 atol = 1e-4
m = @model y ~ x1 fe = pid1 vcov = cluster(pid2)
x = reg(df, m)
@test stderr(x) ≈ [0.02205] atol = 1e-4
# stata reghxe
m = @model y ~ x1 fe = pid1 vcov = cluster(pid1)
x = reg(df, m)
@test stderr(x) ≈ [0.03573] atol = 1e-4

# no reference
m = @model y ~ x1 vcov = cluster(pid1 + pid2)
x = reg(df, m)
@test stderr(x)[1] ≈ 6.17025 atol = 1e-4
# no reference
m = @model y ~ x1 fe = pid1 vcov = cluster(pid1 + pid2)
x = reg(df, m)
@test stderr(x)[1] ≈ 0.04037 atol = 1e-4

# TO CHECK WITH STATA
m = @model y ~ x1 fe = pid1 vcov = cluster(pid1&pid2)
x = reg(df, m)
# 0.0110005 <- from "reghdfe sales price, absorb(state) vce(cluster state#year)"
@test stderr(x) ≈ [0.0110005] atol = 1e-4


@test_throws ErrorException reg(df, @model(y ~ x1, vcov = cluster(State)))

##############################################################################
##
## subset
##
##############################################################################
m = @model y ~ x1 + pid1 subset = (State .<= 30)
x = reg(df, m)
@test length(x.esample) == size(df, 1)
m = @model y ~ x1 + pid1
x2 = reg(df[df[:State] .<= 30, :], m)
@test coef(x) ≈ coef(x2) atol = 1e-4
@test vcov(x) ≈ vcov(x2) atol = 1e-4

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



# like in ivreg2 but += 5 difference. there is a degrees of freedom difference. not sure where.
m = @model y ~ (x1 ~ z1) vcov = cluster(pid1)
x = reg(df, m)
@test x.F_kp     ≈ 7249.88606 atol = 1e-4
m = @model y ~ CPI + (x1 ~ z1) vcov = cluster(pid1)
x = reg(df, m)
@test x.F_kp   ≈ 538.40393 atol = 1e-4
m = @model y ~ CPI + (x1 ~ z1) vcov = cluster(pid1 + pid2)
x = reg(df, m)
@test x.F_kp   ≈ 423.00342 atol = 1e-4
m = @model y ~ (x1 ~ z1 + CPI) vcov = cluster(pid1)
x = reg(df, m)
@test x.F_kp      ≈ 4080.66081 atol = 1e-4
m = @model y ~ (x1 ~ z1 + CPI) vcov = cluster(pid1 + pid2)
x = reg(df, m)
@test x.F_kp     ≈ 2877.94477 atol = 1e-4


##############################################################################
##
## Test unbalanced panel
##
## corresponds to abdata in Stata, for instance reghxe wage emp [w=indoutpt], a(id year)
##
##############################################################################
df = readtable(joinpath(dirname(@__FILE__), "..", "dataset", "EmplUK.csv.gz"))
df[:id1] = df[:Firm]
df[:id2] = df[:Year]
df[:pid1] = pool(df[:id1])
df[:pid2] = pool(df[:id2])
df[:y] = df[:Wage]
df[:x1] = df[:Emp]
df[:w] = df[:Output]


for method in [:cholesky, :qr, :lsmr]
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

##############################################################################
##
## NLS model
##
##############################################################################
df = readtable(joinpath(dirname(@__FILE__), "..", "dataset", "nls.csv"))
pool!(df, [:idcode, :year, :race])
x = reg(df, @model(ln_wage ~ hours + race, fe = idcode))
@test coef(x)[2] ≈ 0.0
x = reg(df, @model(ln_wage ~ hours + race, fe = idcode + year))
@test coef(x)[2] ≈ 0.0
