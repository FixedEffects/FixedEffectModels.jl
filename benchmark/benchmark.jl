using DataFrames, Random, CategoricalArrays
@time using  FixedEffectModels
#  13s precompiling
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
# 1.823s
@time reg(df, @formula(y ~ x1 + x2))
# 0.353469 seconds (441 allocations: 691.439 MiB, 3.65% gc time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
# 0.763999 seconds (2.96 M allocations: 967.418 MiB, 2.29% gc time, 54.39% compilation time: 5% of which was recompilation)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
# 0.401544 seconds (622 allocations: 768.943 MiB, 3.52% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 0.893835 seconds (1.03 k allocations: 929.130 MiB, 54.19% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
#   0.474160 seconds (1.13 k allocations: 933.340 MiB, 1.74% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)), Vcov.cluster(:id1))
#   0.598816 seconds (261.08 k allocations: 1.007 GiB, 8.29% gc time, 9.21% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
# 1.584573 seconds (489.64 k allocations: 1.094 GiB, 2.10% gc time, 8.53% compilation time)

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
# for some reason in 1.10 I now get worse time (iter 200)
#  4.709078 seconds (108.98 k allocations: 101.417 MiB)




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
