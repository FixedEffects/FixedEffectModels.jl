using FixedEffects, RDatasets, DataArrays, DataFrames, GLM
using Base.Test
# values checked from reghdfe
df = dataset("plm", "Cigar")
df[:State] = PooledDataArray(df[:State])
df[:Year] = PooledDataArray(df[:Year])
result = fit(LinearModel, Sales ~ Price, df)
FixedEffects.vcov(FixedEffects.ErrorModel(result.model))
FixedEffects.vcov_robust(FixedEffects.ErrorModel(result.model))
FixedEffects.vcov_cluster(FixedEffects.ErrorModel(result.model), df[:State])
FixedEffects.vcov_cluster2(FixedEffects.ErrorModel(result.model), df[:State], df[:Year])