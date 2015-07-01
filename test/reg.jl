
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
@test_approx_eq coef(reg(Sales ~ Price |> pState + pState&Year , df))    [-0.5347018506709189]
@test_approx_eq_eps  coef(reg(Sales ~ Price |> pState&Year, df))  [-0.5804357763530253] 1e-9
@test_approx_eq_eps  coef(reg(Sales ~ Price |> Year&pState , df))  [-0.5804357763530253] 1e-9

# absorb + weights
@test_approx_eq coef(reg(Sales ~ Price |> pState, df, weight = :Pop)) [-0.21741708996458287]
@test_approx_eq coef(reg(Sales ~ Price |> pState + pYear, df,  weight = :Pop))  [-0.8879409622635229]
@test_approx_eq coef(reg(Sales ~ Price |> pState + pState&Year, df, weight = :Pop))  [-0.46108549276981703]

# iv
@test_approx_eq coef(reg(Sales ~ (Price = Pimin), df))  [138.19479876266445,-0.20733543263106036]
@test_approx_eq coef(reg(Sales ~ NDI + (Price = Pimin), df))   [137.45096580480387,0.005169677634275297,-0.7627670265757879]
@test_approx_eq coef(reg(Sales ~ NDI + (Price = Pimin + Pop), df))  [137.57335924022877,0.00534407899181186,-0.7836515852263581]
result =  [125.2625186785851,0.00426797157619262,-0.4008539252879306,-0.3601265722369686,-0.3437871337260198,-0.34446698651969243,-0.4133823974894119,-0.45733784871835853,-0.5236947552358436,-0.4458389339688059,-0.3588894799553996,-0.36710183734328794,-0.3185077037497715,-0.3044293074955216,-0.2522467465631883,-0.29850950642431934,-0.3243795734788026,-0.39829347905530105,-0.41551734057304207,-0.4466928724499465,-0.46166868245781545,-0.4847134751121637,-0.5265252856149685,-0.5377055415686451,-0.5505433728335279,-0.5691301350473928,-0.588207112580848,-0.600526572322477,-0.6047075271270297,-0.5994956988287091,-0.5644067529587178] 
@test_approx_eq coef(reg(Sales ~ NDI + (Price&pYear = Pimin&pYear), df)) result


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

@test_approx_eq reg(Sales ~ (Price = Pimin), df).F_kp    52248.79247186642
@test_approx_eq reg(Sales ~ CPI + (Price = Pimin), df).F_kp   4112.37092082018
@test_approx_eq reg(Sales ~ (Price = Pimin + CPI), df).F_kp   27011.440804805163
@test_approx_eq reg(Sales ~ (Price = Pimin) |> pState, df).F_kp 100927.75710497228


@test_approx_eq reg(Sales ~ (Price = Pimin), df, VcovWhite()).F_kp    23160.06350543851
@test_approx_eq reg(Sales ~ (Price = Pimin) |> pState, df, VcovWhite()).F_kp  37662.82808814408
@test_approx_eq reg(Sales ~ CPI + (Price = Pimin), df, VcovWhite()).F_kp    2093.4660989306267
@test_approx_eq reg(Sales ~ (Price = Pimin + CPI), df, VcovWhite()).F_kp  16418.211961547782




@test_approx_eq reg(Sales ~ (Price = Pimin), df, VcovCluster(:pState)).F_kp     7249.886065558404
@test_approx_eq reg(Sales ~ CPI + (Price = Pimin), df, VcovCluster(:pState)).F_kp   538.4039346836805
@test_approx_eq reg(Sales ~ CPI + (Price = Pimin), df, VcovCluster([:pState, :pYear])).F_kp   423.00342583390733
@test_approx_eq reg(Sales ~ (Price = Pimin + CPI), df, VcovCluster(:pState)).F_kp      4080.6608113994753
@test_approx_eq reg(Sales ~  (Price = Pimin + CPI), df, VcovCluster([:pState, :pYear])).F_kp     2877.9447778386134




