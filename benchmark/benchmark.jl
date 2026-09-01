using DataFrames, Random
@time using  FixedEffectModels
# 0.442555 seconds (624.29 k allocations: 41.117 MiB, 2.59% gc time, 1.01% compilation time)
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
# 0.175934 seconds (3.43 k allocations: 386.445 MiB, 5.27% gc time, 21.75% compilation time)
@time reg(df, @formula(y ~ x1 + x2))
# 0.198124 seconds (310 allocations: 386.310 MiB, 16.34% gc time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
# 0.221376 seconds (163.66 k allocations: 471.871 MiB, 19.93% gc time, 43.55% compilation time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
# 0.124714 seconds (478 allocations: 463.822 MiB, 6.91% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 0.251411 seconds (187.01 k allocations: 673.778 MiB, 15.87% gc time, 52.26% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 0.209155 seconds (2.77 k allocations: 702.849 MiB, 32.16% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)), Vcov.cluster(:id1))
# 0.242202 seconds (148.74 k allocations: 752.135 MiB, 14.90% gc time, 12.01% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
# 0.594494 seconds (404.37 k allocations: 913.358 MiB, 11.85% gc time, 26.83% compilation time)

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
# 0.727029 seconds (84.78 k allocations: 69.409 MiB, 4.53% gc time)
@time reg(df, @formula(y ~ x1 + fe(id1) + fe(id1)&x2 + fe(id2) + fe(id2)&x2))
# 1.262707 seconds (1.85 M allocations: 204.607 MiB, 3.03% gc time, 43.07% compilation time)
@time reg(df, @formula(y ~ fe(id1)*x1 + fe(id2)*x2))
# 0.525260 seconds (669.35 k allocations: 141.178 MiB, 7.81% gc time, 28.31% compilation time)




# fixest
using CategoricalArrays
n = 10_000_000
nb_dum = [div(n,20), floor(Int, sqrt(n)), floor(Int, n^.33)]
N = nb_dum.^3
id1 = categorical(rand(1:nb_dum[1], n))
id2 = categorical(rand(1:nb_dum[2], n))
id3 = categorical(rand(1:nb_dum[3], n))
X1 = rand(n)
ln_y = 3 .* X1 .+ rand(n)
df = DataFrame(X1 = X1, ln_y = ln_y, id1 = id1, id2 = id2, id3 = id3)
# the first call recompiles code invalidated by loading CategoricalArrays
@time reg(df, @formula(ln_y ~ X1 + fe(id1)), Vcov.cluster(:id1))
# 2.247757 seconds (14.46 M allocations: 1.324 GiB, 3.71% gc time, 91.03% compilation time: 67% of which was recompilation)
@time reg(df, @formula(ln_y ~ X1 + fe(id1)), Vcov.cluster(:id1))
# 0.198736 seconds (2.26 k allocations: 636.432 MiB, 23.03% gc time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2)), Vcov.cluster(:id1))
# 1.093537 seconds (1.77 M allocations: 1.125 GiB, 6.85% gc time, 43.38% compilation time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2)), Vcov.cluster(:id1))
# 0.661163 seconds (6.27 k allocations: 1.006 GiB, 12.54% gc time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2) + fe(id3)), Vcov.cluster(:id1))
# 0.855344 seconds (327.35 k allocations: 1.280 GiB, 10.46% gc time, 17.11% compilation time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2) + fe(id3)), Vcov.cluster(:id1))
# 0.782286 seconds (8.00 k allocations: 1.265 GiB, 10.83% gc time)
