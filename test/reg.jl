
using RDatasets, DataFrames, FixedEffectModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])

##############################################################################
##
## coefficients 
##
##############################################################################

# simple
@test_approx_eq_eps coef(reg(Sales ~ Price, df))  [139.73446,-0.22974] 1e-4
@test_approx_eq_eps coef(reg(Sales ~ Price, df, weight = :Pop))  [137.72495428982756,-0.23738] 1e-4

# absorb
@test_approx_eq_eps coef(reg(Sales ~ Price |> pState, df))  [-0.20984] 1e-4
@test_approx_eq_eps coef(reg(Sales ~ Price |> pState + pYear, df))   [-1.08471] 1e-4
@test_approx_eq_eps coef(reg(Sales ~ Price |> pState + pState&Year , df))    [-0.53470] 1e-4
@test_approx_eq_eps  coef(reg(Sales ~ Price |> pState&Year, df))  [13.993028174622104,-0.5804357763515606] 1e-4
@test_approx_eq_eps  coef(reg(Sales ~ Price |> Year&pState , df))  [13.993028174622104,-0.5804357763515606] 1e-4

# absorb + weights
@test_approx_eq_eps coef(reg(Sales ~ Price |> pState, df, weight = :Pop)) [-0.21741] 1e-4
@test_approx_eq_eps coef(reg(Sales ~ Price |> pState + pYear, df,  weight = :Pop))  [-0.88794] 1e-4
@test_approx_eq_eps coef(reg(Sales ~ Price |> pState + pState&Year, df, weight = :Pop))  [-0.46108] 1e-4

# iv
@test_approx_eq_eps coef(reg(Sales ~ (Price = Pimin), df))  [138.19479,-0.20733] 1e-4
@test_approx_eq_eps coef(reg(Sales ~ NDI + (Price = Pimin), df))   [137.45096,0.00516,-0.76276] 1e-4
@test_approx_eq_eps coef(reg(Sales ~ NDI + (Price = Pimin + Pop), df))  [137.57335,0.00534,-0.78365] 1e-4
result =  [125.26251,0.00426,-0.40085,-0.36012,-0.34378,-0.34446,-0.41338,-0.45733,-0.52369,-0.44583,-0.35888,-0.36710,-0.31850,-0.30442,-0.25224,-0.29850,-0.32437,-0.39829,-0.41551,-0.44669,-0.46166,-0.48471,-0.52652,-0.53770,-0.55054,-0.56913,-0.58820,-0.60052,-0.60470,-0.59949,-0.56440] 
@test_approx_eq_eps coef(reg(Sales ~ NDI + (Price&pYear = Pimin&pYear), df)) result 1e-4


# iv + weight
@test_approx_eq_eps coef(reg(Sales ~ (Price = Pimin), df, weight = :Pop))  [137.03637,-0.22802] 1e-4


# iv + weight + absorb
@test_approx_eq_eps coef(reg(Sales ~ (Price = Pimin) |> pState, df)) [-0.20284] 1e-4
@test_approx_eq_eps coef(reg(Sales ~ (Price = Pimin) |> pState, df, weight = :Pop)) [-0.20995] 1e-4
@test_approx_eq_eps coef(reg(Sales ~ CPI + (Price = Pimin), df))   [129.12222,0.57052,-0.68646] 1e-4
@test_approx_eq_eps coef(reg(Sales ~ CPI + (Price = Pimin) |> pState, df))    [0.56265,-0.67924] 1e-4

# non high dimensional factors
@test_approx_eq_eps coef(reg(Sales ~ Price + pYear |> pState, df))[1]   -1.08471 1e-4
@test_approx_eq_eps coef(reg(Sales ~ Price + pYear |> pState, df,  weight = :Pop))[1]   -0.88794 1e-4
@test_approx_eq_eps coef(reg(Sales ~ NDI + (Price = Pimin) + pYear |> pState, df))[1]   -0.00525 1e-4

@test_throws ErrorException reg(Sales ~ Price + (NDI + Pop = CPI), df)

##############################################################################
##
## std errors 
##
##############################################################################

# Simple
@test_approx_eq_eps stderr(reg(Sales ~ Price , df)) [1.52126,0.01889] 1e-4
# Stata ivreg
@test_approx_eq_eps stderr(reg(Sales ~ (Price = Pimin) , df)) [1.53661,0.01915] 1e-4
# Stata areg
@test_approx_eq_eps stderr(reg(Sales ~ Price |> pState, df))  [0.00980] 1e-4

# White
# Stata reg
@test_approx_eq_eps stderr(reg(Sales ~ Price , df, VcovWhite())) [1.68679,0.01670] 1e-4
# Stata ivreg
@test_approx_eq_eps stderr(reg(Sales ~ (Price = Pimin) , df, VcovWhite()))  [1.63305,0.01674] 1e-4
# Stata areg
@test_approx_eq_eps stderr(reg(Sales ~ Price |> pState, df, VcovWhite())) [0.01100] 1e-4

# cluster
@test_approx_eq_eps stderr(reg(Sales ~ Price , df, VcovCluster(:pState)))[2]  0.03792 1e-4
@test_approx_eq_eps stderr(reg(Sales ~ Price |> pState, df, VcovCluster(:pYear)))  [0.02205] 1e-4
# stata reghdfe
@test_approx_eq_eps stderr(reg(Sales ~ Price |> pState, df, VcovCluster(:pState)))  [0.03573] 1e-4
# no reference
@test_approx_eq_eps stderr(reg(Sales ~ Price, df, VcovCluster([:pState, :pYear])))[1]  6.17025 1e-4
# no reference
@test_approx_eq_eps stderr(reg(Sales ~ Price |> pState, df, VcovCluster([:pState, :pYear])))[1]   0.04037 1e-4

@test_throws ErrorException reg(Sales ~ Price , df, VcovCluster(:State))

##############################################################################
##
## subset
##
##############################################################################

result = reg(Sales ~ Price + pState, df, subset = df[:State] .<= 30)
@test length(result.esample) == size(df, 1)
@test_approx_eq_eps coef(result)  coef(reg(Sales ~ Price + pState, df[df[:State] .<= 30, :])) 1e-4
@test_approx_eq_eps vcov(result)  vcov(reg(Sales ~ Price + pState, df[df[:State] .<= 30, :])) 1e-4
@test_throws ErrorException reg(Sales ~ Price, df, subset = df[:pYear] .<= 30)


##############################################################################
##
## F Stat
##
##############################################################################

@test_approx_eq_eps reg(Sales ~ Price, df).F  147.82425 1e-4
@test_approx_eq_eps reg(Sales ~ Price |> pState, df).F  458.45825 1e-4
@test_approx_eq_eps reg(Sales ~ (Price = Pimin), df).F  117.17329 1e-4

@test_approx_eq_eps reg(Sales ~ Price, df, VcovCluster(:pState)).F    36.70275 1e-4
@test_approx_eq_eps reg(Sales ~ (Price = Pimin), df, VcovCluster(:pState)).F  39.67227 1e-4

#  xtivreg2 
@test_approx_eq_eps reg(Sales ~ (Price = Pimin) |> pState, df).F  422.46444 1e-4

##############################################################################
##
## F_kp r_kp statistics for IV. Difference degree of freedom.
## 
##############################################################################

@test_approx_eq_eps reg(Sales ~ (Price = Pimin), df).F_kp    52248.79247 1e-4
@test_approx_eq_eps reg(Sales ~ CPI + (Price = Pimin), df).F_kp   4112.37092 1e-4
@test_approx_eq_eps reg(Sales ~ (Price = Pimin + CPI), df).F_kp   27011.44080 1e-4
@test_approx_eq_eps reg(Sales ~ (Price = Pimin) |> pState, df).F_kp 100927.75710 1e-4

# exactly same with ivreg2
@test_approx_eq_eps reg(Sales ~ (Price = Pimin), df, VcovWhite()).F_kp    23160.06350 1e-4
@test_approx_eq_eps reg(Sales ~ (Price = Pimin) |> pState, df, VcovWhite()).F_kp  37662.82808 1e-4
@test_approx_eq_eps reg(Sales ~ CPI + (Price = Pimin), df, VcovWhite()).F_kp    2093.46609 1e-4
@test_approx_eq_eps reg(Sales ~ (Price = Pimin + CPI), df, VcovWhite()).F_kp  16418.21196 1e-4



# like in ivreg2 but += 5 difference. there is a degree of freedom difference. not sure where.
@test_approx_eq_eps reg(Sales ~ (Price = Pimin), df, VcovCluster(:pState)).F_kp     7249.88606 1e-4
@test_approx_eq_eps reg(Sales ~ CPI + (Price = Pimin), df, VcovCluster(:pState)).F_kp   538.40393 1e-4
@test_approx_eq_eps reg(Sales ~ CPI + (Price = Pimin), df, VcovCluster([:pState, :pYear])).F_kp   423.00342 1e-4
@test_approx_eq_eps reg(Sales ~ (Price = Pimin + CPI), df, VcovCluster(:pState)).F_kp      4080.66081 1e-4
@test_approx_eq_eps reg(Sales ~  (Price = Pimin + CPI), df, VcovCluster([:pState, :pYear])).F_kp     2877.94477 1e-4




