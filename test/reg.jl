
using DataFrames, FixedEffectModels, Base.Test
#x = readtable("/Users/Matthieu/Dropbox/Github/FixedEffectModels.jl/dataset/Cigar.csv.gz")
x = readtable(joinpath(dirname(@__FILE__), "..", "dataset", "Cigar.csv.gz"))
x[:id1] = x[:State]
x[:id2] = x[:Year]
x[:pid1] = pool(x[:id1])
x[:ppid1] = pool(div(x[:id1], 10))
x[:pid2] = pool(x[:id2])
x[:y] = x[:Sales]
x[:x1] = x[:Price]
x[:z1] = x[:Pimin]
x[:x2] = x[:NDI]
x[:w] = x[:Pop]

##############################################################################
##
## coefficients 
##
##############################################################################

# simple
@test coef(reg(x, @formula(y ~ x1))) ≈ [139.73446,-0.22974] atol = 1e-4
@test coef(reg(x, @formula(y ~ x1), @weight(w))) ≈ [137.72495428982756,-0.23738] atol = 1e-4

x[:SalesInt] = round.(Int64, x[:Sales])
@test coef(reg(x, @formula(SalesInt ~ Price))) ≈ [139.72674,-0.2296683205] atol = 1e-4

# absorb
@test coef(reg(x, @formula(y ~ x1), @fe(pid1))) ≈ [-0.20984] atol = 1e-4
@test reg(x, @formula(y ~ x1), @fe(pid1)).iterations == 1

@test coef(reg(x, @formula(y ~ x1), @fe(pid1 + pid2)))  ≈ [-1.08471] atol = 1e-4
@test coef(reg(x, @formula(y ~ x1), @fe(pid1 + pid1&id2))) ≈   [-0.53470] atol = 1e-4
@test coef(reg(x, @formula(y ~ x1), @fe(pid1*id2))) ≈   [-0.53470] atol = 1e-4
#@test isempty(coef(reg(x, @formula(y ~ 0), @fe(pid1*x1))))
@test coef(reg(x, @formula(y ~ x1), @fe(ppid1&pid2)))  ≈  [-1.44255] atol = 1e-4

@test  coef(reg(x, @formula(y ~ x1), @fe(pid1&id2))) ≈ [13.993028174622104,-0.5804357763515606] atol = 1e-4
@test  coef(reg(x, @formula(y ~ x1), @fe(id2&pid1))) ≈ [13.993028174622104,-0.5804357763515606] atol = 1e-4
@test  coef(reg(x, @formula(y ~ 1), @fe(id2&pid1))) ≈ [174.4084407796102] atol = 1e-4
@test  coef(reg(x, @formula(y ~ x1), @fe(pid1&id2 + pid2&id1))) ≈ [51.2359,- 0.5797] atol = 1e-4
@test  coef(reg(x, @formula(y ~ x1 + x2), @fe(pid1&id2 + pid2&id1))) ≈ [-46.4464,-0.2546, -0.005563] atol = 1e-4
@test  coef(reg(x, @formula(y ~ 0 + x1 + x2), @fe(pid1&id2 + pid2&id1))) ≈  [-0.21226562244177932,-0.004775616634862829] atol = 1e-4


# recheck these two below
@test coef(reg(x, @formula(y ~ z1), @fe(x1&x2&pid1)))  ≈  [122.98713, 0.30933] atol = 1e-4
@test coef(reg(x, @formula(y ~ z1), @fe((x1&x2)*pid1))) ≈   [ 0.421406] atol = 1e-4


# absorb + weights
@test coef(reg(x, @formula(y ~ x1), @fe(pid1), @weight(w))) ≈ [- 0.21741] atol = 1e-4
@test coef(reg(x, @formula(y ~ x1), @fe(pid1 + pid2),  @weight(w))) ≈ [- 0.88794] atol = 1e-3
@test coef(reg(x, @formula(y ~ x1), @fe(pid1 + pid1&id2), @weight(w))) ≈ [- 0.461085492] atol = 1e-4

# iv
@test coef(reg(x, @formula(y ~ (x1 ~ z1)))) ≈ [138.19479,- 0.20733] atol = 1e-4
@test coef(reg(x, @formula(y ~ x2 + (x1 ~ z1)))) ≈  [137.45096,0.00516,- 0.76276] atol = 1e-4
@test coef(reg(x, @formula(y ~ x2 + (x1 ~ z1 + w)))) ≈ [137.57335,0.00534,- 0.78365] atol = 1e-4
## multiple endogeneous variables
@test coef(reg(x, @formula(y ~ (x1 + x2 ~ z1 + w)))) ≈ [139.544, .8001, -.00937] atol = 1e-4
@test coef(reg(x, @formula(y ~ 1 + (x1 + x2 ~ z1 + w)))) ≈ [139.544, .8001, -.00937] atol = 1e-4
result = [196.576, 0.00490989, -2.94019, -3.00686, -2.94903, -2.80183, -2.74789, -2.66682, -2.63855, -2.52394, -2.34751, -2.19241, -2.18707, -2.09244, -1.9691, -1.80463, -1.81865, -1.70428, -1.72925, -1.68501, -1.66007, -1.56102, -1.43582, -1.36812, -1.33677, -1.30426, -1.28094, -1.25175, -1.21438, -1.16668, -1.13033, -1.03782]
@test coef(reg(x, @formula(y ~ x2 + (x1&pid2 ~ z1&pid2)))) ≈ result atol = 1e-4


# iv + weight
@test coef(reg(x, @formula(y ~ (x1 ~ z1)), @weight(w))) ≈ [137.03637,- 0.22802] atol = 1e-4


# iv + weight + absorb
@test coef(reg(x, @formula(y ~ (x1 ~ z1)), @fe(pid1))) ≈ [-0.20284] atol = 1e-4
@test coef(reg(x, @formula(y ~ (x1 ~ z1)), @fe(pid1), @weight(w))) ≈ [-0.20995] atol = 1e-4
@test coef(reg(x, @formula(y ~ x2 + (x1 ~ z1)))) ≈  [137.45096580480387,0.005169677634275297,-0.7627670265757879] atol = 1e-4
@test coef(reg(x, @formula(y ~ x2 + (x1 ~ z1)), @fe(pid1)))  ≈  [0.0011021722526916768,-0.3216374943695231] atol = 1e-4

# non high dimensional factors
@test coef(reg(x, @formula(y ~ x1 + pid2), @fe(pid1)))[1] ≈ -1.08471 atol = 1e-4
@test coef(reg(x, @formula(y ~ x1 + pid2), @fe(pid1),  @weight(w)))[1] ≈ -0.88794 atol = 1e-4
@test coef(reg(x, @formula(y ~ x2 + (x1 ~ z1) + pid2), @fe(pid1)))[1] ≈ -0.00525 atol = 1e-4

##############################################################################
##
## collinearity
## add more tests
## 
##############################################################################
# ols
x[:x12] = x[:x1]
model = reg(x, @formula(y ~ x1 + x12))
@test coef(model) ≈ [139.7344639806166,-0.22974688593485126,0.0] atol = 1e-4

# iv
x[:x22] = x[:x2]
model = reg(x, @formula(y ~  x22 + x2 + (x1 ~ z1)))
@test  coef(model)[2] == 0 || coef(model)[3] == 0

x[:zz1] = x[:z1]
model = reg(x, @formula(y ~  zz1 + (x1 ~ x2 + z1)))
@test coef(model)[2] != 0.0 

# catch when IV underidentified : re-try when 0.5
#@test_throws ErrorException reg(x, @formula(y ~ x1 + (x2 + w = x2)))

# catch continuous variables in fixed effects
@test_throws ErrorException reg(x, @formula(y ~ x1), @fe(x2))
@test_throws ErrorException reg(x, @formula(y ~ x1), @fe(x2 + pid1))


##############################################################################
##
## std errors 
##
##############################################################################

# Simple
@test stderr(reg(x, @formula(y ~ x1))) ≈ [1.52126, 0.01889] atol = 1e-4
# Stata ivreg
@test stderr(reg(x, @formula(y ~ (x1 ~ z1)))) ≈ [1.53661, 0.01915] atol = 1e-4
# Stata areg
@test stderr(reg(x, @formula(y ~ x1), @fe(pid1))) ≈  [0.00980] atol = 1e-4

# White
# Stata reg
@test stderr(reg(x, @formula(y ~ x1), @vcov(robust))) ≈ [1.68679, 0.01670] atol = 1e-4
# Stata ivreg
@test stderr(reg(x, @formula(y ~ (x1 ~ z1)), @vcov(robust))) ≈ [1.63305, 0.01674] atol = 1e-4
# Stata areg
@test stderr(reg(x, @formula(y ~ x1), @fe(pid1), @vcov(robust))) ≈ [0.01100] atol = 1e-4

# cluster
@test stderr(reg(x, @formula(y ~ x1), @vcov(cluster(pid1))))[2] ≈ 0.03792 atol = 1e-4
@test stderr(reg(x, @formula(y ~ x1), @fe(pid1), @vcov(cluster(pid2)))) ≈ [0.02205] atol = 1e-4
# stata reghxe
@test stderr(reg(x, @formula(y ~ x1), @fe(pid1), @vcov(cluster(pid1)))) ≈ [0.03573] atol = 1e-4
# no reference
@test stderr(reg(x, @formula(y ~ x1), @vcov(cluster(pid1 + pid2))))[1] ≈ 6.17025 atol = 1e-4
# no reference
@test stderr(reg(x, @formula(y ~ x1), @fe(pid1), @vcov(cluster(pid1 + pid2))))[1] ≈ 0.04037 atol = 1e-4

@test_throws ErrorException reg(x, @formula(y ~ x1), @vcov(cluster(State)))

##############################################################################
##
## subset
##
##############################################################################

result = reg(x, @formula(y ~ x1 + pid1), subset = x[:State] .<= 30)
@test length(result.esample) == size(x, 1)
@test coef(result) ≈ coef(reg(x[x[:State] .<= 30, :], @formula(y ~ x1 + pid1))) atol = 1e-4
@test vcov(result) ≈ vcov(reg(x[x[:State] .<= 30, :], @formula(y ~ x1 + pid1))) atol = 1e-4
@test_throws ErrorException reg(x, @formula(y ~ x1), subset = x[:pid2] .<= 30)


##############################################################################
##
## F Stat
##
##############################################################################

@test reg(x, @formula(y ~ x1)).F  ≈ 147.82425 atol = 1e-4
@test reg(x, @formula(y ~ x1), @fe(pid1)).F  ≈ 458.45825 atol = 1e-4
@test reg(x, @formula(y ~ (x1 ~ z1))).F  ≈ 117.17329 atol = 1e-4

@test reg(x, @formula(y ~ x1), @vcov(cluster(pid1))).F    ≈ 36.70275 atol = 1e-4
@test reg(x, @formula(y ~ (x1 ~ z1)), @vcov(cluster(pid1))).F  ≈ 39.67227 atol = 1e-4

#  xtivreg2 
@test reg(x, @formula(y ~ (x1 ~ z1)), @fe(pid1)).F  ≈ 422.46444 atol = 1e-4

##############################################################################
##
## F_kp r_kp statistics for IV. Difference degrees of freedom.
## 
##############################################################################

@test reg(x, @formula(y ~ (x1 ~ z1))).F_kp    ≈ 52248.79247 atol = 1e-4
@test reg(x, @formula(y ~ x2 + (x1 ~ z1))).F_kp    ≈ 5159.812208193612 atol = 1e-4
@test reg(x, @formula(y ~ (x1 ~ z1 + CPI))).F_kp   ≈ 27011.44080 atol = 1e-4
@test reg(x, @formula(y ~ (x1 ~ z1)), @fe(pid1)).F_kp ≈ 100927.75710 atol = 1e-4

# exactly same with ivreg2
@test reg(x, @formula(y ~ (x1 ~ z1)), @vcov(robust)).F_kp    ≈ 23160.06350 atol = 1e-4
@test reg(x, @formula(y ~ (x1 ~ z1)), @fe(pid1), @vcov(robust)).F_kp  ≈ 37662.82808 atol = 1e-4
@test reg(x, @formula(y ~ CPI + (x1 ~ z1)), @vcov(robust)).F_kp    ≈ 2093.46609 atol = 1e-4
@test reg(x, @formula(y ~ (x1 ~ z1 + CPI)), @vcov(robust)).F_kp  ≈ 16418.21196 atol = 1e-4



# like in ivreg2 but += 5 difference. there is a degrees of freedom difference. not sure where.
@test reg(x, @formula(y ~ (x1 ~ z1)), @vcov(cluster(pid1))).F_kp     ≈ 7249.88606 atol = 1e-4
@test reg(x, @formula(y ~ CPI + (x1 ~ z1)), @vcov(cluster(pid1))).F_kp   ≈ 538.40393 atol = 1e-4
@test reg(x, @formula(y ~ CPI + (x1 ~ z1)), @vcov(cluster(pid1 + pid2))).F_kp   ≈ 423.00342 atol = 1e-4
@test reg(x, @formula(y ~ (x1 ~ z1 + CPI)), @vcov(cluster(pid1))).F_kp      ≈ 4080.66081 atol = 1e-4
@test reg(x, @formula(y ~  (x1 ~ z1 + CPI)), @vcov(cluster(pid1 + pid2))).F_kp     ≈ 2877.94477 atol = 1e-4


##############################################################################
##
## Test unbalanced panel
## 
## corresponds to abdata in Stata, for instance reghxe wage emp [w=indoutpt], a(id year)
##
##############################################################################
x = readtable(joinpath(dirname(@__FILE__), "..", "dataset", "EmplUK.csv.gz"))
x[:id1] = x[:Firm]
x[:id2] = x[:Year]
x[:pid1] = pool(x[:id1])
x[:pid2] = pool(x[:id2])
x[:y] = x[:Wage]
x[:x1] = x[:Emp]
x[:w] = x[:Output]


for method in [:cholesky, :qr, :lsmr]
	# absorb
	@test coef(reg(x, @formula(y ~ x1), @fe(pid1), method = method)) ≈ [- 0.11981270017206136] atol = 1e-4
	@test  coef(reg(x, @formula(y ~ x1), @fe(pid1&id2), method = method))  ≈ [-315.0000747500431,- 0.07633636891202833] atol = 1e-4
	#
	@test  coef(reg(x, @formula(y ~ x1), @fe(id2&pid1), method = method)) ≈  [-315.0000747500431,- 0.07633636891202833] atol = 1e-4

	@test  coef(reg(x, @formula(y ~ 1), @fe(id2&pid1), method = method)) ≈  [- 356.40430526316396] atol = 1e-4
	@test coef(reg(x, @formula(y ~ x1), @fe(pid1), method = method, @weight(w))) ≈ [- 0.11514363590574725] atol = 1e-4

	# absorb + weights
	@test coef(reg(x, @formula(y ~ x1), @fe(pid1 + pid2), method = method))  ≈  [- 0.04683333721137311] atol = 1e-4
	@test coef(reg(x, @formula(y ~ x1), @fe(pid1 + pid2), method = method,  @weight(w))) ≈  [- 0.043475472188120416] atol = 1e-3

	## the last two ones test an ill conditioned model matrix
	@test coef(reg(x, @formula(y ~ x1), @fe(pid1 + pid1&id2), method = method))  ≈   [- 0.122354] atol = 1e-4
	@test reg(x, @formula(y ~ x1), @fe(pid1 + pid1&id2), method = method).iterations <= 30

	@test coef(reg(x, @formula(y ~ x1), @fe(pid1 + pid1&id2), method = method, @weight(w))) ≈ [- 0.11752306001586807] atol = 1e-4
	@test reg(x, @formula(y ~ x1), @fe(pid1 + pid1&id2), method = method, @weight(w)).iterations <= 50
end




