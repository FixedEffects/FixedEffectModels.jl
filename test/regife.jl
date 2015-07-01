using RDatasets, DataFrames, FixedEffectModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])

@test_approx_eq_eps reg(Sales ~ Price, df, InteractiveFixedEffectModel(:pState, :pYear, 2), tol = 1e-9).coef -0.348820780059964 1e-8

@test_approx_eq_eps reg(Sales ~ Price |> pState, df, InteractiveFixedEffectModel(:pState, :pYear, 2), tol = 1e-9).coef  [-0.42538935900021146] 1e-8
