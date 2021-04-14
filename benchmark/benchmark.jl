using DataFrames, FixedEffectModels, Random

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
# 14s
@time reg(df, @formula(y ~ x1 + x2))
# 0.582029 seconds (852 allocations: 535.311 MiB, 18.28% gc time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
# 0.759032 seconds (631 allocations: 728.437 MiB, 4.89% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 1.316560 seconds (230.66 k allocations: 908.386 MiB, 4.28% gc time, 11.84% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)), Vcov.cluster(:id1))
# 1.501165 seconds (230.94 k allocations: 952.029 MiB, 0.84% gc time, 10.66% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
# 3.058090 seconds (331.56 k allocations: 1.005 GiB, 2.08% gc time, 5.89% compilation time)


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
#   3.190288 seconds (393.89 k allocations: 109.614 MiB, 3.63% gc time, 9.75% compilation time)


+# fixest
n = 10_000_000
nb_dum = [div(n,20), floor(Int, sqrt(n)), floor(Int, n^.33)]
N = nb_dum.^3
id1 = rand(1:nb_dum[1], n)
id2 = rand(1:nb_dum[2], n)
id3 = rand(1:nb_dum[3], n)
X1 = rand(n)
ln_y = 3 .* X1 .+ rand(n) 
df = DataFrame(X1 = X1, ln_y = ln_y, id1 = id1, id2 = id2, id3 = id3)
@time reg(df, @formula(ln_y~X1 + fe(id1)))
#  1.361091 seconds (219.04 k allocations: 731.777 MiB, 10.70% compilation time)
@time reg(df, @formula(ln_y~X1 + fe(id1) + fe(id2)))
#   2.589 seconds (284.42 k allocations: 847.170 MiB, 5.56% compilation time)
@time reg(df, @formula(ln_y~X1 + fe(id1) + fe(id2) + fe(id3)))
#  3.0547 seconds (380.25 k allocations: 967.478 MiB, 2.44% gc time, 6.26% compilation time)