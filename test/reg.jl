
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
@test_approx_eq coef(reg(Sales ~ NDI, df)) [132.9807886574938,-0.0011999855769369572]

# weight
@test_approx_eq coef(reg(Sales ~ NDI, df, weight = :Pop)) [133.9975889363506,-0.0016479139243510202]

# absorb
@test_approx_eq coef(reg(Sales ~ NDI |> pState, df))  [-0.0017046786439408952]
@test_approx_eq coef(reg(Sales ~ NDI |> pState + pYear, df))  [-0.0068438454129809674]
@test_approx_eq coef(reg(Sales ~ NDI |> pState + pState&Year , df))  [-0.005686067588968156]
@test_approx_eq  coef(reg(Sales ~ NDI |> pState&Year, df)) -0.007652680961637854
@test_approx_eq  coef(reg(Sales ~ NDI |> Year&pState , df)) -0.007652680961637854

# absorb + weights
@test_approx_eq coef(reg(Sales ~ NDI |> pState, df, weight = :Pop)) [-0.0016956069964287905]
@test_approx_eq coef(reg(Sales ~ NDI |> pState + pYear, df,  weight = :Pop)) [-0.00526264064237539]
@test_approx_eq coef(reg(Sales ~ NDI |> pState + pState&Year, df, weight = :Pop)) [-0.0046368444054828975]

# iv
@test_approx_eq coef(reg(Sales ~ (NDI = CPI), df))  [134.78303171546634,-0.0014394855828175762]
@test_approx_eq coef(reg(Sales ~ Price + (NDI = CPI), df))  [137.89120963008102,-1.2712981727404156,0.009753824282892953]
@test_approx_eq coef(reg(Sales ~ Price + (NDI = CPI + Pop), df))  [138.26748977604697,-1.0586769085529397,0.007762688103654769]

# iv + weight
@test_approx_eq coef(reg(Sales ~ (NDI = CPI), df, weight = :Pop))   [133.1064411919292,-0.0015409019511033184]

# iv + weight + absorb
@test_approx_eq coef(reg(Sales ~ (NDI = CPI) |> pState, df)) [-0.001439485582817582]
@test_approx_eq coef(reg(Sales ~ (NDI = CPI) |> pState, df, weight = :Pop)) [-0.0015040259980679328]
@test_approx_eq coef(reg(Sales ~ Price + (NDI = CPI), df))   [137.89120963008102, -1.2712981727404156,0.009753824282892953]
@test_approx_eq coef(reg(Sales ~ Price + (NDI = CPI) |> pState, df))   [-0.958100502807393,0.0069962347229067315]

# non high dimensional factors
@test_approx_eq coef(reg(Sales ~ NDI + pYear |> pState, df))[1]  [-0.006843845412979117]
@test_approx_eq coef(reg(Sales ~ NDI + pYear |> pState, df,  weight = :Pop))[1]   -0.005262640642375475
@test_approx_eq coef(reg(Sales ~ Price + (NDI = Pimin) + pYear |> pState, df))[1]  [ -0.7063946354250774]


@test_throws ErrorException reg(Sales ~ Price + (NDI + Pop = CPI), df)

##############################################################################
##
## std errors 
##
##############################################################################

# Simple
@test_approx_eq stderr(reg(Sales ~ NDI , df)) [1.537724332603002,0.00017284152555886948]
@test_approx_eq stderr(reg(Sales ~ (NDI = CPI) , df)) [1.5930914162234808,0.00018143551700490045]
@test_approx_eq stderr(reg(Sales ~ NDI |> pState, df)) [9.139033511351627e-5]

# White
@test_approx_eq stderr(reg(Sales ~ NDI , df, VcovWhite())) [1.6078664457460958,0.00015389800338122924]
@test_approx_eq stderr(reg(Sales ~ (NDI = CPI) , df, VcovWhite())) [1.755052008901572,0.00017285313708536313]
@test_approx_eq stderr(reg(Sales ~ NDI |> pState, df, VcovWhite())) [0.00011195235772951868]

# cluster
@test_approx_eq stderr(reg(Sales ~ NDI , df, VcovCluster(:pState)))[2] [0.0002707729157107782]
@test_approx_eq stderr(reg(Sales ~ NDI |> pState, df, VcovCluster(:pYear)))  [0.00028740748422023774]
@test_approx_eq stderr(reg(Sales ~ NDI |> pState, df, VcovCluster(:pState))) [0.00037490907349394426]
# to check with some good command
@test_approx_eq stderr(reg(Sales ~ NDI, df, VcovCluster([:pState, :pYear])))[1] [5.531951676890601]

@test_approx_eq stderr(reg(Sales ~ NDI, df, VcovCluster([:pState, :pYear])))[1] [5.531951676890601]

@test_throws ErrorException reg(Sales ~ NDI , df, VcovCluster(:State))

##############################################################################
##
## subset
##
##############################################################################

result = reg(Sales ~ NDI + pState, df, subset = df[:State] .<= 30)
@test length(result.esample) == size(df, 1)
@test_approx_eq coef(result)  coef(reg(Sales ~ NDI + pState, df[df[:State] .<= 30, :]))
@test_approx_eq vcov(result)  vcov(reg(Sales ~ NDI + pState, df[df[:State] .<= 30, :]))
@test_throws ErrorException reg(Sales ~ NDI, df, subset = df[:pYear] .<= 30)


##############################################################################
##
## F Stat
##
##############################################################################

@test_approx_eq reg(Sales ~ NDI, df).F 48.20098025424247
@test_approx_eq reg(Sales ~ NDI, df, VcovCluster(:pState)).F  19.640005688702146
@test_approx_eq reg(Sales ~ (NDI = CPI), df, VcovCluster(:pState)).F  18.239257999382957


##############################################################################
##
## F_kp r_kp statistics for IV. Difference degree of freedom.
## TO DO
## 1. check that ranktest, wald full gives same result before any df adjustement
## 2. check  same adjustment than ivreg2
## 
##############################################################################

@test_approx_eq reg(Sales ~ (NDI = CPI), df).F_kp  13737.438334574497
@test_approx_eq reg(Sales ~ (NDI = CPI), df, VcovWhite()).F_kp  8782.33008371174
@test_approx_eq reg(Sales ~ (NDI = CPI), df, VcovCluster(:pState)).F_kp    1675.621098237881
@test_approx_eq reg(Sales ~ Price + (NDI = CPI), df).F_kp  598.5741198798081
@test_approx_eq reg(Sales ~ Price + (NDI = CPI), df, VcovWhite()).F_kp  477.00961298510583
@test_approx_eq reg(Sales ~ Price + (NDI = CPI), df, VcovCluster(:pState)).F_kp 116.25074644651478
@test_approx_eq reg(Sales ~ Price + (NDI = CPI), df, VcovCluster([:pState, :pYear])).F_kp 119.13630113377249
@test_approx_eq reg(Sales ~ Price + (NDI = CPI + Pop), df).F_kp  366.94675893237576
@test_approx_eq reg(Sales ~ Price + (NDI = CPI + Pop), df, VcovWhite()).F_kp 310.8091548244995
@test_approx_eq reg(Sales ~ Price + (NDI = CPI + Pop), df, VcovCluster(:pState)).F_kp   71.14398558169155
@test_approx_eq reg(Sales ~ Price + (NDI = CPI + Pop), df, VcovCluster([:pState, :pYear])).F_kp  79.21891117952379
@test_approx_eq reg(Sales ~ (NDI = CPI) |> pState, df).F_kp 33219.64237335719
@test_approx_eq reg(Sales ~ (NDI = CPI) |> pState, df, VcovWhite()).F_kp 21470.797495280985


