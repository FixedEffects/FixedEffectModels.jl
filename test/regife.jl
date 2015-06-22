using FixedEffectModels, RDatasets, DataArrays, DataFrames
using Base.Test
df = dataset("plm", "Cigar")
df[:State] = PooledDataArray(df[:State])
df[:Year] = PooledDataArray(df[:Year])
reg(Sales~ Price, df, FactorModel(:State, :Year, 2))

@test_approx_eq  reg(Sales~ Price, df, FactorModel(:State, :Year, 2).beta   -0.21406614380576258
@test_approx_eq  regife(Sales~ Price, df, FactorModel(:State, :Year, 2).beta  -0.348495324914281
