using  RDatasets, DataFrames, FixedEffectModels
using Base.Test
# values checked from reghdfe
df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])

#
# coefs
#

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
@test_approx_eq coef(reg(Sales ~ Price + (NDI = CPI), df))  [[137.89120963008102,-1.2712981727404156,0.009753824282892953]

# iv + weight
@test_approx_eq coef(reg(Sales ~ (NDI = CPI), df, weight = :Pop))   [133.1064411919292,-0.0015409019511033184]

# iv + weight + absorb
@test_approx_eq coef(reg(Sales ~ (NDI = CPI) |> pState, df)) [-0.001439485582817582]
@test_approx_eq coef(reg(Sales ~ (NDI = CPI) |> pState, df, weight = :Pop)) [-0.0015040396125277117]
@test_approx_eq coef(reg(Sales ~ Price + (NDI = CPI), df))   [137.89120963008102, -1.2712981727404156,0.009753824282892953]
@test_approx_eq coef(reg(Sales ~ Price + (NDI = CPI) |> pState, df))   [-0.958100502807393,0.0069962347229067315]

