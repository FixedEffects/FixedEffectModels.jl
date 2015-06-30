using RDatasets, DataFrames, FixedEffectModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])


result = reg(Sales ~ Price, df, InteractiveFixedEffectModel(:pState, :pYear, 2))
@test_approx_eq reg(Sales ~ Price, df, InteractiveFixedEffectModel(:pState, :pYear, 2)).coef -0.348821
@test_approx_eq reg(Sales ~ Price |> pState, df, InteractiveFixedEffectModel(:pState, :pYear, 2)).coef  -0.425389
