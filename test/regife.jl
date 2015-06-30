using RDatasets, DataFrames, FixedEffectModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])


result = fit(InteractiveFixedEffectModel(:pState, :pYear, 2), Sales ~ Price, df, VcovSimple())
