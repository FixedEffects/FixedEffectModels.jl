using RDatasets, DataFrames, FixedEffectModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])


result = fit(InteractiveFixedEffectModel(:pState, :pYear, 2), Sales ~ Price, df, VcovSimple())


residual = result.lambda * result.ft
res_vector = similar(df[:Sales])
for i in 1:length(res_vector)
    res_vector[i] = residual[result.id.refs[i], result.time.refs[i]]
end
df[:res] = res_vector

df[:pState] = pool(df[:State])

df[:Salesm] = partial_out(Sales ~ res, df)[1]
df[:Pricem] = partial_out(Price ~ res, df)[1]
reg(Sales~Price + res, df, VcovCluster(:pState))
reg(Salesm ~ Pricem, df, VcovCluster(:pState))