using DataFrames, Random, CategoricalArrays
@time using  FixedEffectModels
#  12s precompiling
# Very simple setup
N = 10000000
K = 100
id1 = rand(1:div(N, K), N)
id2 = rand(1:K, N)
x1 = 5 * cos.(id1) + 5 * sin.(id2) + randn(N)
x2 =  cos.(id1) +  sin.(id2) + randn(N)
y= 3 .* x1 .+ 5 .* x2 .+ cos.(id1) .+ cos.(id2).^2 .+ randn(N)
df = DataFrame(id1 = id1, id2 = id2, x1 = x1, x2 = x2, y = y)
# first time
@time reg(df, @formula(y ~ x1 + x2))
# 4s
@time reg(df, @formula(y ~ x1 + x2))
# 0.497374 seconds (450 allocations: 691.441 MiB, 33.18% gc time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
# 0.605172 seconds (591 allocations: 768.939 MiB, 42.38% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 0.893835 seconds (1.03 k allocations: 929.130 MiB, 54.19% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)), Vcov.cluster(:id1))
# 1.015078 seconds (1.18 k allocations: 1008.532 MiB, 56.50% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
# 1.835464 seconds (4.02 k allocations: 1.057 GiB, 35.59% gc time)

# More complicated setup
N = 800000 # number of observations
M = 40000 # number of workers
O = 5000 # number of firms
id1 = rand(1:M, N)
id2 = [rand(max(1, div(x, 8)-10):min(O, div(x, 8)+10)) for x in id1]
x1 = 5 * cos.(id1) + 5 * sin.(id2) + randn(N)
x2 =  cos.(id1) +  sin.(id2) + randn(N)
y= 3 .* x1 .+ 5 .* x2 .+ cos.(id1) .+ cos.(id2).^2 .+ randn(N)
df = DataFrame(id1 = id1, id2 = id2, x1 = x1, x2 = x2, y = y)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
# 2.504294 seconds (75.83 k allocations: 95.525 MiB, 0.23% gc time)


+# fixest
n = 10_000_000
nb_dum = [div(n,20), floor(Int, sqrt(n)), floor(Int, n^.33)]
N = nb_dum.^3
id1 = categorical(rand(1:nb_dum[1], n))
id2 = categorical(rand(1:nb_dum[2], n))
id3 = categorical(rand(1:nb_dum[3], n))
X1 = rand(n)
ln_y = 3 .* X1 .+ rand(n) 
df = DataFrame(X1 = X1, ln_y = ln_y, id1 = id1, id2 = id2, id3 = id3)
@time reg(df, @formula(ln_y ~ X1 + fe(id1)), Vcov.cluster(:id1))
#  0.543996 seconds (873 allocations: 815.677 MiB, 34.15% gc time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2)), Vcov.cluster(:id1))
#  1.301908 seconds (3.03 k allocations: 968.729 MiB, 25.84% gc time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2) + fe(id3)), Vcov.cluster(:id1))
# 1.658832 seconds (4.17 k allocations: 1.095 GiB, 29.78% gc time)
