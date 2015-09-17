
using RDatasets, DataFrames, FixedEffectModels, Base.Test

df = dataset("plm", "Cigar")
df[:id1] = df[:State]
df[:id2] = df[:Year]
df[:pid1] = pool(df[:id1])
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
@test_approx_eq_eps coef(reg(y ~ x1, df))  [139.73446,-0.22974] 1e-4
@test_approx_eq_eps coef(reg(y ~ x1, df, weight = :w))  [137.72495428982756,-0.23738] 1e-4

# absorb
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1, df))  [-0.20984] 1e-4
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid2, df))   [-1.08471] 1e-4
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid1&id2 , df))    [-0.53470] 1e-4
@test_approx_eq_eps  coef(reg(y ~ x1 |> pid1&id2, df))  [13.993028174622104,-0.5804357763515606] 1e-4

@test_approx_eq_eps  coef(reg(y ~ x1 |> id2&pid1 , df))  [13.993028174622104,-0.5804357763515606] 1e-4

@test_approx_eq_eps  coef(reg(y ~ 1 |> id2&pid1 , df))  [174.4084407796102] 1e-4
@test_approx_eq_eps  coef(reg(y ~ x1 |> pid1&id2 + pid2&id1, df))  [51.2358,-0.5797] 1e-4

@test_approx_eq_eps  coef(reg(y ~ x1 + x2 |> pid1&id2 + pid2&id1, df))  [-46.4464,-0.2546, -0.005563] 1e-4

@test_approx_eq_eps  coef(reg(y ~ 0 + x1 + x2 |> pid1&id2 + pid2&id1, df))   [-0.21226562244177932,-0.004775616634862829] 1e-4


# absorb + weights
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1, df, weight = :w)) [-0.21741] 1e-4
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid2, df,  weight = :w))  [-0.88794] 1e-3
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid1&id2, df, weight = :w))  [-0.46108] 1e-4

# iv
@test_approx_eq_eps coef(reg(y ~ (x1 = z1), df))  [138.19479,-0.20733] 1e-4
@test_approx_eq_eps coef(reg(y ~ x2 + (x1 = z1), df))   [137.45096,0.00516,-0.76276] 1e-4
@test_approx_eq_eps coef(reg(y ~ x2 + (x1 = z1 + w), df))  [137.57335,0.00534,-0.78365] 1e-4
## multiple endogeneous variables
result =  [125.26251,0.00426,-0.40085,-0.36012,-0.34378,-0.34446,-0.41338,-0.45733,-0.52369,-0.44583,-0.35888,-0.36710,-0.31850,-0.30442,-0.25224,-0.29850,-0.32437,-0.39829,-0.41551,-0.44669,-0.46166,-0.48471,-0.52652,-0.53770,-0.55054,-0.56913,-0.58820,-0.60052,-0.60470,-0.59949,-0.56440] 
@test_approx_eq_eps coef(reg(y ~ x2 + (x1&pid2 = z1&pid2), df)) result 1e-4


# iv + weight
@test_approx_eq_eps coef(reg(y ~ (x1 = z1), df, weight = :w))  [137.03637,-0.22802] 1e-4


# iv + weight + absorb
@test_approx_eq_eps coef(reg(y ~ (x1 = z1) |> pid1, df)) [-0.20284] 1e-4
@test_approx_eq_eps coef(reg(y ~ (x1 = z1) |> pid1, df, weight = :w)) [-0.20995] 1e-4
@test_approx_eq_eps coef(reg(y ~ x2 + (x1 = z1), df))   [137.45096580480387,0.005169677634275297,-0.7627670265757879] 1e-4
@test_approx_eq_eps coef(reg(y ~ x2 + (x1 = z1) |> pid1, df))    [0.0011021722526916768,-0.3216374943695231] 1e-4

# non high dimensional factors
@test_approx_eq_eps coef(reg(y ~ x1 + pid2 |> pid1, df))[1]   -1.08471 1e-4
@test_approx_eq_eps coef(reg(y ~ x1 + pid2 |> pid1, df,  weight = :w))[1]   -0.88794 1e-4
@test_approx_eq_eps coef(reg(y ~ x2 + (x1 = z1) + pid2 |> pid1, df))[1]   -0.00525 1e-4



##############################################################################
##
## collinearity
## add more tests
## 
##############################################################################
# ols
df[:x12] = df[:x1]
model = reg(y ~ x1 + x12, df)
@test_approx_eq_eps coef(model) [139.7344639806166,-0.22974688593485126,0.0] 1e-4

# iv
df[:x22] = df[:x2]
model = reg(y ~  x22 + x2 + (x1 = z1), df)
@test_approx_eq_eps coef(model) [137.45096580480387,0.005169677634275297,0.0,-0.7627670265757879] 1e-4

# catch when IV underidentified 
@test_throws ErrorException reg(y ~ x1 + (x2 + w = x2), df)


##############################################################################
##
## std errors 
##
##############################################################################

# Simple
@test_approx_eq_eps stderr(reg(y ~ x1 , df)) [1.52126,0.01889] 1e-4
# Stata ivreg
@test_approx_eq_eps stderr(reg(y ~ (x1 = z1) , df)) [1.53661,0.01915] 1e-4
# Stata areg
@test_approx_eq_eps stderr(reg(y ~ x1 |> pid1, df))  [0.00980] 1e-4

# White
# Stata reg
@test_approx_eq_eps stderr(reg(y ~ x1 , df, VcovWhite())) [1.68679,0.01670] 1e-4
# Stata ivreg
@test_approx_eq_eps stderr(reg(y ~ (x1 = z1) , df, VcovWhite()))  [1.63305,0.01674] 1e-4
# Stata areg
@test_approx_eq_eps stderr(reg(y ~ x1 |> pid1, df, VcovWhite())) [0.01100] 1e-4

# cluster
@test_approx_eq_eps stderr(reg(y ~ x1 , df, VcovCluster(:pid1)))[2]  0.03792 1e-4
@test_approx_eq_eps stderr(reg(y ~ x1 |> pid1, df, VcovCluster(:pid2)))  [0.02205] 1e-4
# stata reghdfe
@test_approx_eq_eps stderr(reg(y ~ x1 |> pid1, df, VcovCluster(:pid1)))  [0.03573] 1e-4
# no reference
@test_approx_eq_eps stderr(reg(y ~ x1, df, VcovCluster([:pid1, :pid2])))[1]  6.17025 1e-4
# no reference
@test_approx_eq_eps stderr(reg(y ~ x1 |> pid1, df, VcovCluster([:pid1, :pid2])))[1]   0.04037 1e-4

@test_throws ErrorException reg(y ~ x1 , df, VcovCluster(:State))

##############################################################################
##
## subset
##
##############################################################################

result = reg(y ~ x1 + pid1, df, subset = df[:State] .<= 30)
@test length(result.esample) == size(df, 1)
@test_approx_eq_eps coef(result)  coef(reg(y ~ x1 + pid1, df[df[:State] .<= 30, :])) 1e-4
@test_approx_eq_eps vcov(result)  vcov(reg(y ~ x1 + pid1, df[df[:State] .<= 30, :])) 1e-4
@test_throws ErrorException reg(y ~ x1, df, subset = df[:pid2] .<= 30)


##############################################################################
##
## F Stat
##
##############################################################################

@test_approx_eq_eps reg(y ~ x1, df).F  147.82425 1e-4
@test_approx_eq_eps reg(y ~ x1 |> pid1, df).F  458.45825 1e-4
@test_approx_eq_eps reg(y ~ (x1 = z1), df).F  117.17329 1e-4

@test_approx_eq_eps reg(y ~ x1, df, VcovCluster(:pid1)).F    36.70275 1e-4
@test_approx_eq_eps reg(y ~ (x1 = z1), df, VcovCluster(:pid1)).F  39.67227 1e-4

#  xtivreg2 
@test_approx_eq_eps reg(y ~ (x1 = z1) |> pid1, df).F  422.46444 1e-4

##############################################################################
##
## F_kp r_kp statistics for IV. Difference degree of freedom.
## 
##############################################################################

@test_approx_eq_eps reg(y ~ (x1 = z1), df).F_kp    52248.79247 1e-4
@test_approx_eq_eps reg(y ~ x2 + (x1 = z1), df).F_kp    5159.812208193612 1e-4
@test_approx_eq_eps reg(y ~ (x1 = z1 + CPI), df).F_kp   27011.44080 1e-4
@test_approx_eq_eps reg(y ~ (x1 = z1) |> pid1, df).F_kp 100927.75710 1e-4

# exactly same with ivreg2
@test_approx_eq_eps reg(y ~ (x1 = z1), df, VcovWhite()).F_kp    23160.06350 1e-4
@test_approx_eq_eps reg(y ~ (x1 = z1) |> pid1, df, VcovWhite()).F_kp  37662.82808 1e-4
@test_approx_eq_eps reg(y ~ CPI + (x1 = z1), df, VcovWhite()).F_kp    2093.46609 1e-4
@test_approx_eq_eps reg(y ~ (x1 = z1 + CPI), df, VcovWhite()).F_kp  16418.21196 1e-4



# like in ivreg2 but += 5 difference. there is a degree of freedom difference. not sure where.
@test_approx_eq_eps reg(y ~ (x1 = z1), df, VcovCluster(:pid1)).F_kp     7249.88606 1e-4
@test_approx_eq_eps reg(y ~ CPI + (x1 = z1), df, VcovCluster(:pid1)).F_kp   538.40393 1e-4
@test_approx_eq_eps reg(y ~ CPI + (x1 = z1), df, VcovCluster([:pid1, :pid2])).F_kp   423.00342 1e-4
@test_approx_eq_eps reg(y ~ (x1 = z1 + CPI), df, VcovCluster(:pid1)).F_kp      4080.66081 1e-4
@test_approx_eq_eps reg(y ~  (x1 = z1 + CPI), df, VcovCluster([:pid1, :pid2])).F_kp     2877.94477 1e-4


##############################################################################
##
## Test unbalanced panel
## 
##############################################################################
df = dataset("plm", "EmplUK")
# corresponds to abdata in Stata
df[:id1] = df[:Firm]
df[:id2] = df[:Year]
df[:pid1] = pool(df[:id1])
df[:pid2] = pool(df[:id2])
df[:y] = df[:Wage]
df[:x1] = df[:Emp]
df[:w] = df[:Output]



# absorb
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1, df))  [-0.11981270017206136] 1e-4
@test_approx_eq_eps  coef(reg(y ~ x1 |> pid1&id2, df))   [-315.0000747500431,-0.07633636891202833] 1e-4
#
@test_approx_eq_eps  coef(reg(y ~ x1 |> id2&pid1 , df))   [-315.0000747500431,-0.07633636891202833] 1e-4

@test_approx_eq_eps  coef(reg(y ~ 1 |> id2&pid1 , df))   [-356.40430526316396] 1e-4
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1, df, weight = :w))  [-0.11514363590574725] 1e-4

# absorb + weights
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid2, df))    [-0.04683333721137311] 1e-4
# 10 vs 6
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid2, df,  weight = :w))   [-0.043475472188120416] 1e-3
# 12 vs 7 reghdfe wage emp [w=indoutpt], a(id year)
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid1&id2 , df))     [-0.122354] 1e-4
# 11 vs 555
@test_approx_eq_eps coef(reg(y ~ x1 |> pid1 + pid1&id2, df, weight = :w))  [-0.11752306001586807] 1e-4
# 17 vs 123