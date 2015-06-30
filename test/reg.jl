
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
@test_approx_eq coef(reg(Sales ~ Price, df))  [139.73446398061662,-0.22974688593485126]
@test_approx_eq coef(reg(Sales ~ Price, df, weight = :Pop))  [137.72495428982756,-0.23738785198180068]

# absorb
@test_approx_eq coef(reg(Sales ~ Price |> pState, df))  [-0.20984020448410937]
@test_approx_eq coef(reg(Sales ~ Price |> pState + pYear, df))   [-1.0847116771624785]
@test_approx_eq coef(reg(Sales ~ Price |> pState + pState&Year , df))   [-0.5347018506305184]
@test_approx_eq  coef(reg(Sales ~ Price |> pState&Year, df)) [-0.5804357763548721]
@test_approx_eq  coef(reg(Sales ~ Price |> Year&pState , df))  [-0.5804357763548721]

# absorb + weights
@test_approx_eq coef(reg(Sales ~ Price |> pState, df, weight = :Pop)) [-0.21741708996458287]
@test_approx_eq coef(reg(Sales ~ Price |> pState + pYear, df,  weight = :Pop))  [-0.8879409622635229]
@test_approx_eq coef(reg(Sales ~ Price |> pState + pState&Year, df, weight = :Pop)) [-0.4610854926901806]

# iv
@test_approx_eq coef(reg(Sales ~ (Price = Pimin), df))  [138.19479876266445,-0.20733543263106036]
@test_approx_eq coef(reg(Sales ~ Price + (Price = Pimin), df))   [139.73446398061674,-0.22974688593482195]
@test_approx_eq coef(reg(Sales ~ Price + (Price = Pimin + Pop), df))  [139.734463980616,-0.22974688593483705]

# iv + weight
@test_approx_eq coef(reg(Sales ~ (Price = Pimin), df, weight = :Pop))  [137.03637388628948,-0.22802072465514023]


# iv + weight + absorb
@test_approx_eq coef(reg(Sales ~ (Price = Pimin) |> pState, df)) [-0.20284570547536304]
@test_approx_eq coef(reg(Sales ~ (Price = Pimin) |> pState, df, weight = :Pop)) [-0.2099590966839856]
@test_approx_eq coef(reg(Sales ~ CPI + (Price = Pimin), df))   [129.12222260424062,0.5705253664930297,-0.6864653275917334]
@test_approx_eq coef(reg(Sales ~ CPI + (Price = Pimin) |> pState, df))    [0.5626533217244698,-0.6792493989912431]

# non high dimensional factors
@test_approx_eq coef(reg(Sales ~ Price + pYear |> pState, df))[1]   -1.0847116771619385
@test_approx_eq coef(reg(Sales ~ Price + pYear |> pState, df,  weight = :Pop))[1]   -0.887940962263503
@test_approx_eq coef(reg(Sales ~ NDI + (Price = Pimin) + pYear |> pState, df))[1]   -0.00525881287252726

@test_throws ErrorException reg(Sales ~ Price + (NDI + Pop = CPI), df)

##############################################################################
##
## std errors 
##
##############################################################################

# Simple
@test_approx_eq stderr(reg(Sales ~ Price , df)) [1.5212694034758718,0.018896300457166917]
# Stata ivreg
@test_approx_eq stderr(reg(Sales ~ (Price = Pimin) , df)) [1.5366130215233473,0.01915398817977868]
# Stata areg
@test_approx_eq stderr(reg(Sales ~ Price |> pState, df))  [0.00980028722024757]

# White
# Stata reg
@test_approx_eq stderr(reg(Sales ~ Price , df, VcovWhite())) [1.6867906024352297,0.016704213947769154]
# Stata ivreg
@test_approx_eq stderr(reg(Sales ~ (Price = Pimin) , df, VcovWhite()))  [1.633058917058125,0.01674332681498417]
# Stata areg
@test_approx_eq stderr(reg(Sales ~ Price |> pState, df, VcovWhite())) [0.011000532786118948]

# cluster
@test_approx_eq stderr(reg(Sales ~ Price , df, VcovCluster(:pState)))[2]  0.037922792324783315
@test_approx_eq stderr(reg(Sales ~ Price |> pState, df, VcovCluster(:pYear)))  [0.022056324143006456]
# stata reghdfe
@test_approx_eq stderr(reg(Sales ~ Price |> pState, df, VcovCluster(:pState)))  [0.03573682488445591]
# no reference
@test_approx_eq stderr(reg(Sales ~ Price, df, VcovCluster([:pState, :pYear])))[1]  6.170255492662165
# no reference
@test_approx_eq stderr(reg(Sales ~ Price |> pState, df, VcovCluster([:pState, :pYear])))[1]   0.040379251758139396

@test_throws ErrorException reg(Sales ~ Price , df, VcovCluster(:State))

##############################################################################
##
## subset
##
##############################################################################

result = reg(Sales ~ Price + pState, df, subset = df[:State] .<= 30)
@test length(result.esample) == size(df, 1)
@test_approx_eq coef(result)  coef(reg(Sales ~ Price + pState, df[df[:State] .<= 30, :]))
@test_approx_eq vcov(result)  vcov(reg(Sales ~ Price + pState, df[df[:State] .<= 30, :]))
@test_throws ErrorException reg(Sales ~ Price, df, subset = df[:pYear] .<= 30)


##############################################################################
##
## F Stat
##
##############################################################################

@test_approx_eq reg(Sales ~ Price, df).F  147.82425385390684
@test_approx_eq reg(Sales ~ Price |> pState, df).F  458.4582515791206
@test_approx_eq reg(Sales ~ (Price = Pimin), df).F  117.17329004522445

@test_approx_eq reg(Sales ~ Price, df, VcovCluster(:pState)).F    36.70275389035331
@test_approx_eq reg(Sales ~ (Price = Pimin), df, VcovCluster(:pState)).F  39.67227257135368

#  xtivreg2 
@test_approx_eq reg(Sales ~ (Price = Pimin) |> pState, df).F  422.464443431403

##############################################################################
##
## F_kp r_kp statistics for IV. Difference degree of freedom.
## They are not matched to ivreg2 (algthough they are very close). Don't really know where the difference comes from
## 1. check that ranktest, wald full gives same result before any df adjustement
## 2. check  same adjustment than ivreg2
## 
##############################################################################

@test_approx_eq reg(Sales ~ (Price = Pimin), df).F_kp   52210.93102804621
@test_approx_eq reg(Sales ~ CPI + (Price = Pimin), df).F_kp   4106.410962963775
@test_approx_eq reg(Sales ~ (Price = Pimin + CPI), df).F_kp   26972.293789145497
@test_approx_eq reg(Sales ~ (Price = Pimin) |> pState, df).F_kp 97490.36247893337


@test_approx_eq reg(Sales ~ (Price = Pimin), df, VcovWhite()).F_kp   23160.075684433177
@test_approx_eq reg(Sales ~ (Price = Pimin) |> pState, df, VcovWhite()).F_kp  36380.10858079424
@test_approx_eq reg(Sales ~ CPI + (Price = Pimin), df, VcovWhite()).F_kp    2091.950196184545
@test_approx_eq reg(Sales ~ (Price = Pimin + CPI), df, VcovWhite()).F_kp   16394.417451458183




@test_approx_eq reg(Sales ~ (Price = Pimin), df, VcovCluster(:pState)).F_kp    7097.426639083516
@test_approx_eq reg(Sales ~ CPI + (Price = Pimin), df, VcovCluster(:pState)).F_kp   526.6997786943028
@test_approx_eq reg(Sales ~ CPI + (Price = Pimin), df, VcovCluster([:pState, :pYear])).F_kp   408.903526978258
@test_approx_eq reg(Sales ~ (Price = Pimin + CPI), df, VcovCluster(:pState)).F_kp     3989.0559781012853
@test_approx_eq reg(Sales ~  (Price = Pimin + CPI), df, VcovCluster([:pState, :pYear])).F_kp    2779.995871693709




