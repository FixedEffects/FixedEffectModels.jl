using DataFrames, FixedEffectModels, Random

# Very simple setup
N = 10000000
K = 100
id1 = rand(1:div(N, K), N)
id2 = rand(1:K, N)
x1 = 5 * cos.(id1) + 5 * sin.(id2) + randn(N)
x2 =  cos.(id1) +  sin.(id2) + randn(N)
y= 3 .* x1 .+ 5 .* x2 .+ cos.(id1) .+ cos.(id2).^2 .+ randn(N)
df = DataFrame(id1 = categorical(id1), id2 = categorical(id2), x1 = x1, x2 = x2, y = y)
@time reg(df, @formula(y ~ x1 + x2))
# 0.582029 seconds (852 allocations: 535.311 MiB, 18.28% gc time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
# 0.809652 seconds (1.55 k allocations: 649.821 MiB, 14.40% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 1.655732 seconds (1.21 k allocations: 734.353 MiB, 16.88% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)), Vcov.cluster(:id1))
#  2.113248 seconds (499.66 k allocations: 811.751 MiB, 15.08% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
# 3.553678 seconds (1.36 k allocations: 1005.101 MiB, 10.55% gc time))


# More complicated setup
N = 800000 # number of observations
M = 40000 # number of workers
O = 5000 # number of firms
id1 = rand(1:M, N)
id2 = [rand(max(1, div(x, 8)-10):min(O, div(x, 8)+10)) for x in id1]
x1 = 5 * cos.(id1) + 5 * sin.(id2) + randn(N)
x2 =  cos.(id1) +  sin.(id2) + randn(N)
y= 3 .* x1 .+ 5 .* x2 .+ cos.(id1) .+ cos.(id2).^2 .+ randn(N)
df = DataFrame(id1 = categorical(id1), id2 = categorical(id2), x1 = x1, x2 = x2, y = y)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
# 15.738214 seconds (929.97 k allocations: 147.737 MiB, 0.95% gc time)




