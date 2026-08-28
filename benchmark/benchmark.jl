using DataFrames, Random, CategoricalArrays
@time using  FixedEffectModels
#  0.580477 seconds (632.49 k allocations: 41.519 MiB, 2.80% gc time, 0.84% compilation time)
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
#   1.328866 seconds (7.87 M allocations: 771.431 MiB, 4.44% gc time, 79.16% compilation time: 94% of which was recompilation)
@time reg(df, @formula(y ~ x1 + x2))
#   0.275951 seconds (322 allocations: 386.311 MiB, 19.27% gc time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
#   0.530246 seconds (1.60 M allocations: 542.123 MiB, 3.19% gc time, 60.83% compilation time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
#   0.268264 seconds (490 allocations: 463.822 MiB, 10.66% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
#   0.708850 seconds (2.05 M allocations: 804.701 MiB, 5.09% gc time, 66.27% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
#   0.317658 seconds (2.78 k allocations: 702.849 MiB, 14.03% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)), Vcov.cluster(:id1))
#   0.433026 seconds (148.71 k allocations: 790.274 MiB, 17.56% gc time, 9.16% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
#   0.903748 seconds (404.27 k allocations: 875.213 MiB, 10.17% gc time, 21.63% compilation time)

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
#   0.790119 seconds (92.25 k allocations: 75.734 MiB, 3.99% gc time)
@time reg(df, @formula(y ~ x1 + fe(id1) + fe(id1)&x2 + fe(id2) + fe(id2)&x2))
#   1.365713 seconds (1.87 M allocations: 205.596 MiB, 2.98% gc time, 39.06% compilation time)
@time reg(df, @formula(y ~ fe(id1)*x1 + fe(id2)*x2))
#   0.516583 seconds (678.56 k allocations: 144.636 MiB, 2.77% gc time, 20.37% compilation time)




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
#   0.952912 seconds (4.06 M allocations: 892.628 MiB, 4.22% gc time, 69.77% compilation time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2)), Vcov.cluster(:id1))
#   1.206156 seconds (1.77 M allocations: 1.152 GiB, 5.81% gc time, 40.53% compilation time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2) + fe(id3)), Vcov.cluster(:id1))
#   0.998796 seconds (327.33 k allocations: 1.307 GiB, 8.67% gc time, 15.16% compilation time)
