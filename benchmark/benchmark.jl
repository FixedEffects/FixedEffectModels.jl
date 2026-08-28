using DataFrames, Random, CategoricalArrays
@time using  FixedEffectModels
# 0.416951 seconds (632.49 k allocations: 41.519 MiB, 2.81% gc time, 1.04% compilation time)
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
# 2.079799 seconds (7.87 M allocations: 771.431 MiB, 5.29% gc time, 82.48% compilation time: 96% of which was recompilation)
@time reg(df, @formula(y ~ x1 + x2))
# 0.360490 seconds (322 allocations: 386.311 MiB, 15.31% gc time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
# 0.580137 seconds (1.60 M allocations: 542.123 MiB, 6.76% gc time, 57.67% compilation time: 14% of which was recompilation)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
# 0.267717 seconds (490 allocations: 463.822 MiB, 13.42% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 0.728273 seconds (2.05 M allocations: 804.700 MiB, 5.36% gc time, 65.15% compilation time: 63% of which was recompilation)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 0.319863 seconds (2.78 k allocations: 702.849 MiB, 14.55% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)), Vcov.cluster(:id1))
# 0.398538 seconds (148.71 k allocations: 790.275 MiB, 18.14% gc time, 6.71% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
# 0.696066 seconds (404.26 k allocations: 875.212 MiB, 6.05% gc time, 21.59% compilation time)

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
# 0.669781 seconds (76.06 k allocations: 72.184 MiB, 6.77% gc time)
@time reg(df, @formula(y ~ x1 + fe(id1) + fe(id1)&x2 + fe(id2) + fe(id2)&x2))
# 1.342872 seconds (1.87 M allocations: 205.454 MiB, 2.90% gc time, 41.52% compilation time)
@time reg(df, @formula(y ~ fe(id1)*x1 + fe(id2)*x2))
# 0.623492 seconds (684.57 k allocations: 135.691 MiB, 6.71% gc time, 17.00% compilation time)




# fixest
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
# 0.923952 seconds (4.06 M allocations: 892.628 MiB, 3.97% gc time, 69.77% compilation time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1)), Vcov.cluster(:id1))
# 0.403069 seconds (2.26 k allocations: 654.682 MiB, 10.13% gc time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2)), Vcov.cluster(:id1))
# 1.294296 seconds (1.77 M allocations: 1.189 GiB, 7.65% gc time, 38.80% compilation time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2)), Vcov.cluster(:id1))
# 1.022833 seconds (6.28 k allocations: 1.068 GiB, 7.29% gc time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2) + fe(id3)), Vcov.cluster(:id1))
# 1.053856 seconds (327.33 k allocations: 1.381 GiB, 10.49% gc time, 15.26% compilation time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2) + fe(id3)), Vcov.cluster(:id1))
# 1.302102 seconds (8.01 k allocations: 1.217 GiB, 4.49% gc time)
