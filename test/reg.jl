
using RDatasets, DataFrames, FixedEffectModels, Base.Test



x = dataset("plm", "Cigar")
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
@test_approx_eq_eps coef(reg(y ~ x1, x))  [139.73446,-0.22974] 1e-4
@test_approx_eq_eps coef(reg(y ~ x1, x, weight = :w))  [137.72495428982756,-0.23738] 1e-4

x[:SalesInt] = round(Int64, x[:Sales])
@test_approx_eq_eps coef(reg(SalesInt ~ Price, x)) [139.72674,-0.2296683205] 1e-4

# absorb
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1, x))  [-0.20984] 1e-4
@test reg(y ~ x1 |> pid1, x).iterations == 1

@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid2, x))   [-1.08471] 1e-4
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid1&id2 , x))    [-0.53470] 1e-4
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1*id2 , x))    [-0.53470] 1e-4
#@test isempty(coef(reg(y ~ 0 |> pid1*x1, x)))
@test_approx_eq_eps coef(reg(y ~ x1 |> ppid1&pid2 , x))    [-1.44255] 1e-4

@test_approx_eq_eps  coef(reg(y ~ x1 |> pid1&id2, x))  [13.993028174622104,-0.5804357763515606] 1e-4
@test_approx_eq_eps  coef(reg(y ~ x1 |> id2&pid1 , x))  [13.993028174622104,-0.5804357763515606] 1e-4
@test_approx_eq_eps  coef(reg(y ~ 1 |> id2&pid1 , x))  [174.4084407796102] 1e-4
@test_approx_eq_eps  coef(reg(y ~ x1 |> pid1&id2 + pid2&id1, x))  [51.2358,-0.5797] 1e-4
@test_approx_eq_eps  coef(reg(y ~ x1 + x2 |> pid1&id2 + pid2&id1, x))  [-46.4464,-0.2546, -0.005563] 1e-4
@test_approx_eq_eps  coef(reg(y ~ 0 + x1 + x2 |> pid1&id2 + pid2&id1, x))   [-0.21226562244177932,-0.004775616634862829] 1e-4


# recheck these two below
@test_approx_eq_eps coef(reg(y ~ z1 |> x1&x2&pid1 , x))    [122.98713,0.30933] 1e-4
@test_approx_eq_eps coef(reg(y ~ z1 |> (x1&x2)*pid1 , x))    [0.421406] 1e-4


# absorb + weights
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1, x, weight = :w)) [-0.21741] 1e-4
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid2, x,  weight = :w))  [-0.88794] 1e-3
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid1&id2, x, weight = :w))  [-0.461085492] 1e-4

# iv
@test_approx_eq_eps coef(reg(y ~ (x1 = z1), x))  [138.19479,-0.20733] 1e-4
@test_approx_eq_eps coef(reg(y ~ x2 + (x1 = z1), x))   [137.45096,0.00516,-0.76276] 1e-4
@test_approx_eq_eps coef(reg(y ~ x2 + (x1 = z1 + w), x))  [137.57335,0.00534,-0.78365] 1e-4
## multiple endogeneous variables
@test_approx_eq_eps coef(reg(y ~ (x1 + x2 = z1 + w), x))  [139.544, .8001, -.00937] 1e-4
@test_approx_eq_eps coef(reg(y ~ 1 + (x1 + x2 = z1 + w), x))  [139.544, .8001, -.00937] 1e-4
result = [196.576, 0.00490989, -2.94019, -3.00686, -2.94903, -2.80183, -2.74789, -2.66682, -2.63855, -2.52394, -2.34751, -2.19241, -2.18707, -2.09244, -1.9691, -1.80463, -1.81865, -1.70428, -1.72925, -1.68501, -1.66007, -1.56102, -1.43582, -1.36812, -1.33677, -1.30426, -1.28094, -1.25175, -1.21438, -1.16668, -1.13033, -1.03782]
@test_approx_eq_eps coef(reg(y ~ x2 + (x1&pid2 = z1&pid2), x)) result 1e-4


# iv + weight
@test_approx_eq_eps coef(reg(y ~ (x1 = z1), x, weight = :w))  [137.03637,-0.22802] 1e-4


# iv + weight + absorb
@test_approx_eq_eps coef(reg(y ~ (x1 = z1) |> pid1, x)) [-0.20284] 1e-4
@test_approx_eq_eps coef(reg(y ~ (x1 = z1) |> pid1, x, weight = :w)) [-0.20995] 1e-4
@test_approx_eq_eps coef(reg(y ~ x2 + (x1 = z1), x))   [137.45096580480387,0.005169677634275297,-0.7627670265757879] 1e-4
@test_approx_eq_eps coef(reg(y ~ x2 + (x1 = z1) |> pid1, x))    [0.0011021722526916768,-0.3216374943695231] 1e-4

# non high dimensional factors
@test_approx_eq_eps coef(reg(y ~ x1 + pid2 |> pid1, x))[1]   -1.08471 1e-4
@test_approx_eq_eps coef(reg(y ~ x1 + pid2 |> pid1, x,  weight = :w))[1]   -0.88794 1e-4
@test_approx_eq_eps coef(reg(y ~ x2 + (x1 = z1) + pid2 |> pid1, x))[1]   -0.00525 1e-4

##############################################################################
##
## collinearity
## add more tests
## 
##############################################################################
# ols
x[:x12] = x[:x1]
model = reg(y ~ x1 + x12, x)
@test_approx_eq_eps coef(model) [139.7344639806166,-0.22974688593485126,0.0] 1e-4

# iv
x[:x22] = x[:x2]
model = reg(y ~  x22 + x2 + (x1 = z1), x)
@test  coef(model)[2] == 0 || coef(model)[3] == 0

x[:zz1] = x[:z1]
model = reg(y ~  zz1 + (x1 = x2 + z1), x)
@test coef(model)[2] != 0.0 

# catch when IV underidentified : re-try when 0.5
#@test_throws ErrorException reg(y ~ x1 + (x2 + w = x2), x)

# catch continuous variables in fixed effects
@test_throws ErrorException reg(y ~ x1 |> x2, x)
@test_throws ErrorException reg(y ~ x1 |> x2 + pid1, x)


##############################################################################
##
## std errors 
##
##############################################################################

# Simple
@test_approx_eq_eps stderr(reg(y ~ x1 , x)) [1.52126,0.01889] 1e-4
# Stata ivreg
@test_approx_eq_eps stderr(reg(y ~ (x1 = z1) , x)) [1.53661,0.01915] 1e-4
# Stata areg
@test_approx_eq_eps stderr(reg(y ~ x1 |> pid1, x))  [0.00980] 1e-4

# White
# Stata reg
@test_approx_eq_eps stderr(reg(y ~ x1 , x, VcovWhite())) [1.68679,0.01670] 1e-4
# Stata ivreg
@test_approx_eq_eps stderr(reg(y ~ (x1 = z1) , x, VcovWhite()))  [1.63305,0.01674] 1e-4
# Stata areg
@test_approx_eq_eps stderr(reg(y ~ x1 |> pid1, x, VcovWhite())) [0.01100] 1e-4

# cluster
@test_approx_eq_eps stderr(reg(y ~ x1 , x, VcovCluster(:pid1)))[2]  0.03792 1e-4
@test_approx_eq_eps stderr(reg(y ~ x1 |> pid1, x, VcovCluster(:pid2)))  [0.02205] 1e-4
# stata reghxe
@test_approx_eq_eps stderr(reg(y ~ x1 |> pid1, x, VcovCluster(:pid1)))  [0.03573] 1e-4
# no reference
@test_approx_eq_eps stderr(reg(y ~ x1, x, VcovCluster([:pid1, :pid2])))[1]  6.17025 1e-4
# no reference
@test_approx_eq_eps stderr(reg(y ~ x1 |> pid1, x, VcovCluster([:pid1, :pid2])))[1]   0.04037 1e-4

@test_throws ErrorException reg(y ~ x1 , x, VcovCluster(:State))

##############################################################################
##
## subset
##
##############################################################################

result = reg(y ~ x1 + pid1, x, subset = x[:State] .<= 30)
@test length(result.esample) == size(x, 1)
@test_approx_eq_eps coef(result)  coef(reg(y ~ x1 + pid1, x[x[:State] .<= 30, :])) 1e-4
@test_approx_eq_eps vcov(result)  vcov(reg(y ~ x1 + pid1, x[x[:State] .<= 30, :])) 1e-4
@test_throws ErrorException reg(y ~ x1, x, subset = x[:pid2] .<= 30)


##############################################################################
##
## F Stat
##
##############################################################################

@test_approx_eq_eps reg(y ~ x1, x).F  147.82425 1e-4
@test_approx_eq_eps reg(y ~ x1 |> pid1, x).F  458.45825 1e-4
@test_approx_eq_eps reg(y ~ (x1 = z1), x).F  117.17329 1e-4

@test_approx_eq_eps reg(y ~ x1, x, VcovCluster(:pid1)).F    36.70275 1e-4
@test_approx_eq_eps reg(y ~ (x1 = z1), x, VcovCluster(:pid1)).F  39.67227 1e-4

#  xtivreg2 
@test_approx_eq_eps reg(y ~ (x1 = z1) |> pid1, x).F  422.46444 1e-4

##############################################################################
##
## F_kp r_kp statistics for IV. Difference degrees of freedom.
## 
##############################################################################

@test_approx_eq_eps reg(y ~ (x1 = z1), x).F_kp    52248.79247 1e-4
@test_approx_eq_eps reg(y ~ x2 + (x1 = z1), x).F_kp    5159.812208193612 1e-4
@test_approx_eq_eps reg(y ~ (x1 = z1 + CPI), x).F_kp   27011.44080 1e-4
@test_approx_eq_eps reg(y ~ (x1 = z1) |> pid1, x).F_kp 100927.75710 1e-4

# exactly same with ivreg2
@test_approx_eq_eps reg(y ~ (x1 = z1), x, VcovWhite()).F_kp    23160.06350 1e-4
@test_approx_eq_eps reg(y ~ (x1 = z1) |> pid1, x, VcovWhite()).F_kp  37662.82808 1e-4
@test_approx_eq_eps reg(y ~ CPI + (x1 = z1), x, VcovWhite()).F_kp    2093.46609 1e-4
@test_approx_eq_eps reg(y ~ (x1 = z1 + CPI), x, VcovWhite()).F_kp  16418.21196 1e-4



# like in ivreg2 but += 5 difference. there is a degrees of freedom difference. not sure where.
@test_approx_eq_eps reg(y ~ (x1 = z1), x, VcovCluster(:pid1)).F_kp     7249.88606 1e-4
@test_approx_eq_eps reg(y ~ CPI + (x1 = z1), x, VcovCluster(:pid1)).F_kp   538.40393 1e-4
@test_approx_eq_eps reg(y ~ CPI + (x1 = z1), x, VcovCluster([:pid1, :pid2])).F_kp   423.00342 1e-4
@test_approx_eq_eps reg(y ~ (x1 = z1 + CPI), x, VcovCluster(:pid1)).F_kp      4080.66081 1e-4
@test_approx_eq_eps reg(y ~  (x1 = z1 + CPI), x, VcovCluster([:pid1, :pid2])).F_kp     2877.94477 1e-4


##############################################################################
##
## Test unbalanced panel
## 
## corresponds to abdata in Stata, for instance reghxe wage emp [w=indoutpt], a(id year)
##
##############################################################################

x = dataset("plm", "EmplUK")
x[:id1] = x[:Firm]
x[:id2] = x[:Year]
x[:pid1] = pool(x[:id1])
x[:pid2] = pool(x[:id2])
x[:y] = x[:Wage]
x[:x1] = x[:Emp]
x[:w] = x[:Output]


for method in [:cholesky, :qr, :lsmr]
	# absorb
	@test_approx_eq_eps coef(reg(y ~ x1 |> pid1, x, method = method))  [-0.11981270017206136] 1e-4
	@test_approx_eq_eps  coef(reg(y ~ x1 |> pid1&id2, x, method = method))   [-315.0000747500431,-0.07633636891202833] 1e-4
	#
	@test_approx_eq_eps  coef(reg(y ~ x1 |> id2&pid1 , x, method = method))   [-315.0000747500431,-0.07633636891202833] 1e-4

	@test_approx_eq_eps  coef(reg(y ~ 1 |> id2&pid1 , x, method = method))   [-356.40430526316396] 1e-4
	@test_approx_eq_eps coef(reg(y ~ x1 |> pid1, x, method = method, weight = :w))  [-0.11514363590574725] 1e-4

	# absorb + weights
	@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid2, x, method = method))    [-0.04683333721137311] 1e-4
	@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid2, x, method = method,  weight = :w))   [-0.043475472188120416] 1e-3

	## the last two ones test an ill conditioned model matrix
	@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid1&id2 , x, method = method))     [-0.122354] 1e-4
	@test reg(y ~ x1 |> pid1 + pid1&id2 , x, method = method).iterations <= 30

	@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid1&id2, x, method = method, weight = :w))  [-0.11752306001586807] 1e-4
	@test reg(y ~ x1 |> pid1 + pid1&id2, x, method = method, weight = :w).iterations <= 50
end




