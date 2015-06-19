using FixedEffectModels, RDatasets, DataArrays, DataFrames
using Base.Test
# values checked from reghdfe
df = dataset("plm", "Cigar")
df[:State] = PooledDataArray(df[:State])
df[:Year] = PooledDataArray(df[:Year])

reg(Sales~NDI | State, df)
reg(Sales~NDI | State, df, VceWhite())
reg(Sales~NDI | State, df, VceCluster(:State))

@test_approx_eq  coef(reg(Sales~NDI | State , df))   -0.0017046786439408937
@test_approx_eq  coef(reg(Sales~NDI | (State + State&Year), df))  -0.005686067588968152
@test_approx_eq  coef(reg(Sales~NDI | (State&Year), df)) -0.007652680961637854
df[:Year] = PooledDataArray(df[:Year])
@test_approx_eq  coef(reg(Sales~NDI | (State + Year))   -0.00684384541298097

