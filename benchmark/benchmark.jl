@everywhere using DataFrames, FixedEffectModels
N = 10000000
K = 100
id1 = Int.(rand(1:(N/K), N))
id2 = Int.(rand(1:K, N))
w = cos.(id1)

x1 = 5 * cos.(id1) + 5 * sin.(id2) + randn(N)
x2 =  cos.(id1) +  sin.(id2) + randn(N)
y= 3 .* x1 .+ 5 .* x2 .+ cos.(id1) .+ cos.(id2).^2 .+ randn(N)
df = DataFrame(id1 = categorical(id1), id2 = categorical(id2), x1 = x1, x2 = x2, w = w, y = y)
@time reg(df, @model(y ~ x1 + x2))
# 1.258554 seconds (723 allocations: 1.205 GB, 18.70% gc time)
#852MB to obtain the matrix etc, and then 1.205 for Regressions part
@time reg(df, @model(y ~ x1 + x2, vcov = cluster(id2)))
#  1.569679 seconds (843 allocations: 1.293 GB, 25.21% gc time)
@time reg(df, @model(y ~ x1 + x2, fe = id1))
# 1.476390 seconds (890 allocations: 1.175 GB, 20.15% gc time)
@time reg(df, @model(y ~ x1 + x2, fe = id1, vcov = cluster(id1)))
# 1.974738 seconds (1.04 k allocations: 1.255 GB, 15.85% gc time)
@time reg(df, @model(y ~ x1 + x2, fe = id1 + id2))
# 4.554836 seconds (1.01 k allocations: 1.188 GB, 9.56% gc time)
@time reg(df, @model(y ~ x1 + x2, fe = id1, weights = w))
# 2.000850 seconds (20.00 M allocations: 1.010 GB, 18.82% gc time)
@time reg(df, @model(y ~ x1 + x2, fe = id1 + id2, weights = w))
# 4.118974 seconds (20.00 M allocations: 1.018 GB, 9.60% gc time)



# Benchmark Parallel
df[:id3] = categorical(Int.(rand(1:30, N)))
df[:x3] =  cos.(id1) + sin.(id2) + randn(N)
df[:x4] =  cos.(id1) + sin.(id2) + randn(N)
df[:x5] =  cos.(id1) + sin.(id2) + randn(N)
df[:x6] =  cos.(id1) + sin.(id2) + randn(N)
df[:x7] =  cos.(id1) + sin.(id2) + randn(N)
@time reg(df, @model(y ~ x1 + x2 + x3 + x4, fe = id1 + id2 + id2&x3, weights = w, method = lsmr_parallel))

@time reg(df, @model(y ~ x1 + x2 + x3 + x4, fe = id1 + id2 + id2&x3, weights = w, method = lsmr))


@time reg(df, @model(y ~ x1 + id3, fe = id1 + id2 + id2&x3, weights = w, method = lsmr_parallel))

@time reg(df, @model(y ~ x1 + id3, fe = id1 + id2 + id2&x3, weights = w, method = lsmr))
