using RDatasets, DataFrames, FixedEffectModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])

@test_approx_eq reg(Sales ~ Price, df, InteractiveFixedEffectModel(:pState, :pYear, 2), tol = 1e-9).coef [-0.3487521323959165]

@test_approx_eq reg(Sales ~ Price |> pState, df, InteractiveFixedEffectModel(:pState, :pYear, 2), tol = 1e-9).coef  [-0.42451721572999235]
