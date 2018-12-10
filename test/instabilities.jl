using DataFrames, Test, Random, FixedEffectModels
Random.seed!(0)
df = DataFrame(x1 = randn(10000) * 100 )
df[:x2] = df[:x1].^4

## Make sure all coefficients are estimated
result = reg(df, @model(x1 ~ x2 ))
@test sum(abs.(coef(result)) .> 0)  == 2
