using DataFrames, FixedEffectModels, Base.Test

df = readtable(joinpath(dirname(@__FILE__), "..", "dataset/Cigar.csv.gz"))
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])


result = @reg df Sales ~ NDI
show(result)
predict(result, df)
residuals(result, df)



result = @reg df Sales ~ CPI + (Price ~ Pimin)
predict(result, df)
residuals(result, df)
model_response(result, df)
@test  nobs(result) == 1380
@test vcov(result)[1] ≈ 3.5384578251636785

result = @reg df Sales ~ Price  fe = pState
show(result)
result = @reg df  Sales ~ CPI + (Price ~ Pimin)  fe = pState
show(result)





result = @reg df Sales ~ Price
@test residuals(result, df)[1:10] ≈ [-39.2637, -37.48801, -34.38801, -36.09743, -36.97446, -43.15547, -41.22573, -40.83648, -34.52427, -28.91617] atol = 1e-4

result = @reg df Sales ~ Price fe = pState save = true
@test result.augmentdf[:residuals][1:10] ≈ [-22.08499, -20.33318, -17.23318, -18.97645, -19.85547, -26.1161, -24.20627, -23.87674, -17.62624, -12.01018] atol = 1e-4



