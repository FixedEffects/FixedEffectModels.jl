using DataFrames, Base.Test

df = CSV.read(joinpath(dirname(@__FILE__), "..", "dataset/Cigar.csv"))
df[:pState] = categorical(df[:State])
df[:pYear] = categorical(df[:Year])


model = @model Sales ~ NDI
result = reg(df, model)
show(result)
predict(result, df)
residuals(result, df)



model = @model Sales ~ CPI + (Price ~ Pimin)
result = reg(df, model)
predict(result, df)
residuals(result, df)
model_response(result, df)
@test  nobs(result) == 1380
@test vcov(result)[1] ≈ 3.5384578251636785

model = @model Sales ~ Price  fe = pState
result = reg(df, model)
show(result)
model = @model Sales ~ CPI + (Price ~ Pimin) fe = pState
result = reg(df, model)
show(result)





model = @model Sales ~ Price
result = reg(df, model)
@test residuals(result, df)[1:10] ≈ [-39.2637, -37.48801, -34.38801, -36.09743, -36.97446, -43.15547, -41.22573, -40.83648, -34.52427, -28.91617] atol = 1e-4

model = @model Sales ~ Price fe = pState save = true
result = reg(df, model)
@test result.augmentdf[:residuals][1:10] ≈ [-22.08499, -20.33318, -17.23318, -18.97645, -19.85547, -26.1161, -24.20627, -23.87674, -17.62624, -12.01018] atol = 1e-4



