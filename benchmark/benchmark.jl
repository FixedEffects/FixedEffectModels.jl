using DataFrames, Random, CategoricalArrays
@time using  FixedEffectModels
# 0.497405 seconds (633.17 k allocations: 41.566 MiB, 8.54% gc time, 0.89% compilation time)
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
# 1.336506 seconds (7.87 M allocations: 771.431 MiB, 5.21% gc time, 77.16% compilation time: 94% of which was recompilation)
@time reg(df, @formula(y ~ x1 + x2))
# 0.307350 seconds (322 allocations: 386.311 MiB, 20.61% gc time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
# 0.515049 seconds (1.60 M allocations: 542.123 MiB, 2.24% gc time, 60.82% compilation time: 11% of which was recompilation)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
# 0.270209 seconds (490 allocations: 463.822 MiB, 10.28% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 0.787299 seconds (2.05 M allocations: 804.700 MiB, 5.75% gc time, 66.80% compilation time: 60% of which was recompilation)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 0.352568 seconds (2.78 k allocations: 702.849 MiB, 14.56% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)), Vcov.cluster(:id1))
# 0.435032 seconds (148.71 k allocations: 790.274 MiB, 18.25% gc time, 6.84% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
# 0.814278 seconds (404.28 k allocations: 875.213 MiB, 5.05% gc time, 20.53% compilation time)

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
# 0.866437 seconds (96.98 k allocations: 69.788 MiB, 0.33% gc time)
@time reg(df, @formula(y ~ x1 + fe(id1) + fe(id1)&x2 + fe(id2) + fe(id2)&x2))
# 1.663603 seconds (1.89 M allocations: 205.990 MiB, 0.26% gc time, 36.53% compilation time)
@time reg(df, @formula(y ~ fe(id1)*x1 + fe(id2)*x2))
# 0.551153 seconds (673.28 k allocations: 135.326 MiB, 2.18% gc time, 21.33% compilation time)




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
# 0.957912 seconds (4.06 M allocations: 854.487 MiB, 4.34% gc time, 72.85% compilation time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1)), Vcov.cluster(:id1))
# 0.335721 seconds (2.26 k allocations: 645.557 MiB, 12.05% gc time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2)), Vcov.cluster(:id1))
# 1.241810 seconds (1.77 M allocations: 1.143 GiB, 6.78% gc time, 39.60% compilation time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2)), Vcov.cluster(:id1))
# 0.830758 seconds (6.28 k allocations: 1.015 GiB, 6.23% gc time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2) + fe(id3)), Vcov.cluster(:id1))
# 0.978856 seconds (327.33 k allocations: 1.331 GiB, 8.08% gc time, 15.23% compilation time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2) + fe(id3)), Vcov.cluster(:id1))
# 0.986598 seconds (8.01 k allocations: 1.315 GiB, 13.37% gc time)
