# Benchmark Parallel
df[:id3] = categorical(Int.(rand(1:15, N)))
df[:x3] =  cos.(id1) + sin.(id2) + randn(N)
sort!(df, [:id1])
@time reg(df, @model(y ~ x1 + id3, fe = id1 + id2 + id2&x3, weights = w, method = lsmr))