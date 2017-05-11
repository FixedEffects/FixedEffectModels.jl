using DataFrames, FixedEffectModels, Base.Test

x = readtable(joinpath(dirname(@__FILE__), "..", "dataset/Cigar.csv.gz"))
x[:pState] = pool(x[:State])
x[:pYear] = pool(x[:Year])


result = reg(x, @formula(Sales ~ NDI))
show(result)
predict(result, x)
residuals(result, x)




result = reg(x, @formula(Sales ~ CPI + (Price ~ Pimin)))
predict(result, x)
residuals(result, x)
model_response(result, x)
@test  nobs(result) == 1380
@test vcov(result)[1] ≈ 3.5384578251636785


show(reg(x, @formula(Sales ~ Price), @fe(pState)))
show(reg(x, @formula(Sales ~ CPI + (Price ~ Pimin)), @fe(pState)))





result = reg(x, @formula(Sales ~ Price))
@test residuals(result, x)[1:10] ≈ [-39.2637, -37.48801, -34.38801, -36.09743, -36.97446, -43.15547, -41.22573, -40.83648, -34.52427, -28.91617] atol = 1e-4

result = reg(x, @formula(Sales ~ Price), @fe(pState), save = true)
@test result.augmentdf[:residuals][1:10] ≈ [-22.08499, -20.33318, -17.23318, -18.97645, -19.85547, -26.1161, -24.20627, -23.87674, -17.62624, -12.01018] atol = 1e-4



