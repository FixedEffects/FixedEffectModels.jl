using RDatasets, DataFrames, FixedEffectModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])


result = reg(Sales ~ NDI, df)
show(result)
predict(result, df)
residuals(result, df)


result = reg(Sales ~ Price + (NDI = CPI), df)
predict(result, df)
residuals(result, df)
model_response(result, df)
@test coefnames(result) == [symbol("(Intercept)"), :Price, :NDI]
@test  nobs(result) == 1380
@test_approx_eq  vcov(result)[1]  2.1197498605509337