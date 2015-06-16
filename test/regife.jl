using FixedEffects, RDatasets, DataArrays, DataFrames
using Base.Test
# values checked from reghdfe
df = dataset("plm", "Cigar")
df[:State] = PooledDataArray(df[:State])
df[:Year] = PooledDataArray(df[:Year])
@test_approx_eq  regife(Sales~ Price, df, nothing ~ State +Year, 1).beta   -0.21406614380576258
@test_approx_eq  regife(Sales~ Price, df, nothing ~ State +Year, 2).beta  -0.348495324914281
