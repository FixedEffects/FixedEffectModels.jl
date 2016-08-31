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





result = reg(Sales ~ Price, df)
@test maxabs(residuals(result, df)[1:10] .- [-39.2637, -37.48801, -34.38801, -36.09743, -36.97446, -43.15547, -41.22573, -40.83648, -34.52427, -28.91617]) <= 1e-4

result = reg(Sales ~ Price |> pState, df, save = true)
@test maxabs(result.augmentdf[:residuals][1:10] .- [-22.08499, -20.33318, -17.23318, -18.97645, -19.85547, -26.1161, -24.20627, -23.87674, -17.62624, -12.01018]) <= 1e-4



