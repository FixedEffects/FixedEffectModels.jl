using DataFrames, FixedEffectModels
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
# 2.448953 seconds (500.21 k allocations: 1.052 GiB, 17.36% gc time)
@time reg(df, @model(y ~ x1 + x2, fe = id1 + id2))
#  3.639817 seconds (1.84 k allocations: 999.675 MiB, 11.25% gc time)




# Benchmark Parallel
df[:id3] = categorical(Int.(rand(1:15, N)))
df[:x3] =  cos.(id1) + sin.(id2) + randn(N)
sort!(df, [:id1])
@time reg(df, @model(y ~ x1 + id3, fe = id1 + id2 + id2&x3, weights = w, method = lsmr))
@time reg(df, @model(y ~ x1 + id3, fe = id1 + id2 + id2&x3, weights = w, method = lsmr_parallel))
@time reg(df, @model(y ~ x1 + id3, fe = id1 + id2 + id2&x3, weights = w, method = lsmr_threads))


#check that as fast as lm with no fixed effects
df[:id3] = categorical(Int.(rand(1:15, N)))
@time reg(df, @model(y ~ x1 + id3))
using GLM
@time lm(@formula(y ~ x1 + id3), df)

# compare with R data.table
using DataFrames
function f(y, id)
    cache = zeros(eltype(y), length(id.pool))
    for i in 1:length(y)
        cache[id.refs[i]] += y[i]
    end
    out = similar(y)
    for i in 1:length(y)
        out[i] = cache[id.refs[i]]
    end
    return out
end
N = 100000000
K = 100
y = randn(N)
id1 = categorical(Int.(rand(1:(N/K), N)))
id2 = categorical(Int.(rand(1:K, N)))
@time f(y, id1)
#1.9
@time f(y, id2)
#0.5

# in R
N = 100000000
K = 100
y = rnorm(N)
id1 = sample(floor(N/K), N, TRUE)
id2 = sample(K, N, TRUE)
dt = data.table(y = y, id1 = id1, id2 = id2)
setkey(dt, id1)
system.time(dt[, list(y1 = mean(y)), by = id1])
#1.3
setkey(dt, id2)
system.time(dt[, list(y1 = mean(y)), by = id2])
#1.3

