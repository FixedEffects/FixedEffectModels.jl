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
#0.601445 seconds (1.05 k allocations: 535.311 MiB, 31.95% gc time)
@time reg(df, @model(y ~ x1 + x2, vcov = cluster(id2)))
#  1.213357 seconds (2.01 k allocations: 878.712 MiB, 16.65% gc time)
@time reg(df, @model(y ~ x1 + x2, fe = id1))
# 1.476390 seconds (890 allocations: 1.175 GB, 20.15% gc time)
@time reg(df, @model(y ~ x1 + x2, fe = id1, vcov = cluster(id1)))
# 2.847599 seconds (702.12 k allocations: 1011.550 MiB, 17.50% gc time)
@time reg(df, @model(y ~ x1 + x2, fe = id1 + id2))
#  3.329693 seconds (201.97 k allocations: 778.576 MiB, 11.40% gc time)
@time reg(df, @model(y ~ x1 + x2, fe = id1, weights = w))
# 1.1 seconds (20.00 M allocations: 1.010 GB, 18.82% gc time)
@time reg(df, @model(y ~ x1 + x2, fe = id1 + id2, weights = w))
# 2.353851 seconds (202.01 k allocations: 550.882 MiB, 15.65% gc time)



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
