using DataFrames, Random
@time using  FixedEffectModels
# 0.516901 seconds (625.23 k allocations: 41.157 MiB, 2.71% gc time, 0.94% compilation time)
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
# 0.336580 seconds (3.45 k allocations: 386.446 MiB, 9.03% gc time, 10.83% compilation time)
@time reg(df, @formula(y ~ x1 + x2))
# 0.333338 seconds (319 allocations: 386.311 MiB, 2.54% gc time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
# 0.374062 seconds (163.68 k allocations: 471.621 MiB, 10.13% gc time, 29.99% compilation time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
# 0.305426 seconds (487 allocations: 463.822 MiB, 16.44% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 0.414578 seconds (187.02 k allocations: 711.918 MiB, 8.96% gc time, 32.54% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 0.357588 seconds (2.78 k allocations: 702.849 MiB, 15.50% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)), Vcov.cluster(:id1))
# 0.431498 seconds (148.75 k allocations: 790.275 MiB, 15.95% gc time, 8.58% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
# 0.752960 seconds (404.36 k allocations: 913.359 MiB, 9.25% gc time, 21.47% compilation time)

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
# 0.849341 seconds (90.62 k allocations: 69.590 MiB)
@time reg(df, @formula(y ~ x1 + fe(id1) + fe(id1)&x2 + fe(id2) + fe(id2)&x2))
# 1.594869 seconds (1.86 M allocations: 198.844 MiB, 0.78% gc time, 38.12% compilation time)
@time reg(df, @formula(y ~ fe(id1)*x1 + fe(id2)*x2))
# 0.681447 seconds (675.00 k allocations: 147.447 MiB, 4.64% gc time, 19.71% compilation time)




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
@time reg(df, @formula(ln_y ~ X1 + fe(id1)), Vcov.cluster(:id1))
# 2.384253 seconds (14.46 M allocations: 1.324 GiB, 5.22% gc time, 88.63% compilation time: 68% of which was recompilation)
@time reg(df, @formula(ln_y ~ X1 + fe(id1)), Vcov.cluster(:id1))
# 0.281829 seconds (2.26 k allocations: 629.588 MiB, 4.09% gc time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2)), Vcov.cluster(:id1))
# 1.249629 seconds (1.77 M allocations: 1.084 GiB, 6.24% gc time, 41.52% compilation time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2)), Vcov.cluster(:id1))
# 0.812104 seconds (6.28 k allocations: 1.001 GiB, 9.90% gc time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2) + fe(id3)), Vcov.cluster(:id1))
# 1.133076 seconds (327.35 k allocations: 1.278 GiB, 7.99% gc time, 13.74% compilation time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2) + fe(id3)), Vcov.cluster(:id1))
# 1.024502 seconds (8.01 k allocations: 1.260 GiB, 11.05% gc time)
