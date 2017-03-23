using RDatasets, DataFrames, FixedEffectModels, Base.Test

x = dataset("plm", "Cigar")
x[:pState] = pool(x[:State])
x[:pYear] = pool(x[:Year])


result = reg(@formula(Sales ~ NDI), x)
show(result)
predict(result, x)
residuals(result, x)




result = reg(@formula(Sales ~ CPI + (Price = Pimin)), x)
predict(result, x)
residuals(result, x)
model_response(result, x)
@test  nobs(result) == 1380
@test_approx_eq  vcov(result)[1]  3.5384578251636785


show(reg(@formula(Sales ~ Price |> pState), x))
show(reg(@formula(Sales ~ CPI + (Price = Pimin) |> pState), x))





result = reg(@formula(Sales ~ Price), x)
@test maxabs(residuals(result, x)[1:10] .- [-39.2637, -37.48801, -34.38801, -36.09743, -36.97446, -43.15547, -41.22573, -40.83648, -34.52427, -28.91617]) <= 1e-4

result = reg(@formula(Sales ~ Price |> pState), x, save = true)
@test maxabs(result.augmentdf[:residuals][1:10] .- [-22.08499, -20.33318, -17.23318, -18.97645, -19.85547, -26.1161, -24.20627, -23.87674, -17.62624, -12.01018]) <= 1e-4



