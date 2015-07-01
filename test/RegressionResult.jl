using RDatasets, DataFrames, FixedEffectModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])


result = reg(Sales ~ NDI, df)
show(result)
predict(result, df)
residuals(result, df)



reg(Sales ~ CPI + (Price = Pimin), df)

result = reg(Sales ~ CPI + (Price = Pimin), df)
predict(result, df)
residuals(result, df)
model_response(result, df)
@test  nobs(result) == 1380
@test_approx_eq  vcov(result)[1]  3.5384578251636785


show(reg(Sales ~ Price |> pState, df))
show(reg(Sales ~ CPI + (Price = Pimin) |> pState, df))
