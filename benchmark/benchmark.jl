using DataFrames, FixedEffectModels, StatsModels
N = 10000000
K = 100
id1 = Int.(rand(1:(N/K), N))
id2 = Int.(rand(1:K, N))
w = cos.(id1)


x1 = 5 * cos.(id1) + 5 * sin.(id2) + randn(N)
x2 =  cos.(id1) +  sin.(id2) + randn(N)
y= 3 .* x1 .+ 5 .* x2 .+ cos.(id1) .+ cos.(id2).^2 .+ randn(N)



df = DataFrame(id1 = categorical(id1), id2 = categorical(id2), x1 = x1, x2 = x2, w = w, y = y)
@time completecases(df, [:y, :x1, :x2, :id2])




@time reg(df, @model(y ~ x1 + x2))
# 0.582029 seconds (852 allocations: 535.311 MiB, 18.28% gc time)
@time reg(df, @model(y ~ x1 + x2, vcov = cluster(id2)))
# 0.809652 seconds (1.55 k allocations: 649.821 MiB, 14.40% gc time)
@time reg(df, @model(y ~ x1 + x2, fe = id1))
# 1.655732 seconds (1.21 k allocations: 734.353 MiB, 16.88% gc time)
@time reg(df, @model(y ~ x1 + x2, fe = id1, vcov = cluster(id1)))
#  2.113248 seconds (499.66 k allocations: 811.751 MiB, 15.08% gc time)
@time reg(df, @model(y ~ x1 + x2, fe = id1 + id2))
# 3.553678 seconds (1.36 k allocations: 1005.101 MiB, 10.55% gc time))
@time reg(df, @model(y ~ x1 + x2, fe = id1 + id2, vcov = cluster(id1 + id2)))


# more regressors
df.x3 =  cos.(id1) + sin.(id2) + randn(N)
df.x4 =  cos.(id1) + sin.(id2) + randn(N)
df.x5 =  cos.(id1) + sin.(id2) + randn(N)
df.x6 =  cos.(id1) + sin.(id2) + randn(N)
df.x7 =  cos.(id1) + sin.(id2) + randn(N)
@time reg(df, @model(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7, subset = x3 .>= 0.5))

@time reg(df, @model(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7, fe = id1))
#  6.191703 seconds (537.87 k allocations: 4.545 GiB, 5.93% gc time)
@time reg(df, @model(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7, fe = id1+id2))

@time reg(df, @model(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7, fe = id1+id2), method = :lsmr_threads)

df.id1 = categorical(mod.(1:size(df, 1), Ref(5)))
df.id4 = categorical(mod.(1:size(df, 1), Ref(4)))
@time reg(df, @model(y ~ id4, fe = id1), method = :lsmr)



# Benchmark Parallel
df.id3 = categorical(Int.(rand(1:15, N)))
df.x3 =  cos.(id1) + sin.(id2) + randn(N)
sort!(df, [:id1])
@time reg(df, @model(y ~ x1 + id3, fe = id1 + id2 + id2&x3, weights = w), method = :lsmr)
@time reg(df, @model(y ~ x1 + id3, fe = id1 + id2 + id2&x3, weights = w), method = :lsmr_parallel)
@time reg(df, @model(y ~ x1 + id3, fe = id1 + id2 + id2&x3, weights = w), method = :lsmr_threads)


#check that as fast as lm with no fixed effects
df.id3 = categorical(Int.(rand(1:15, N)))
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

