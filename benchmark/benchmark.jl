using DataFrames, Random, CategoricalArrays
@time using  FixedEffectModels
#  0.418712 seconds (742.49 k allocations: 45.500 MiB, 4.07% gc time, 1.07% compilation time)
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
#   1.810739 seconds (14.05 M allocations: 1.583 GiB, 3.76% gc time, 84.28% compilation time: 91% of which was recompilation)
@time reg(df, @formula(y ~ x1 + x2))
#   0.288344 seconds (712 allocations: 920.405 MiB, 18.71% gc time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
#   0.528590 seconds (1.33 M allocations: 1.039 GiB, 8.76% gc time, 55.33% compilation time)
@time reg(df, @formula(y ~ x1 + x2),  Vcov.cluster(:id2))
#   0.268619 seconds (879 allocations: 997.916 MiB, 15.08% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
# 0.824536 seconds (3.09 M allocations: 1.426 GiB, 5.37% gc time, 61.39% compilation time: 3% of which was recompilation)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))
#    0.356793 seconds (1.41 k allocations: 1.276 GiB, 19.31% gc time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)), Vcov.cluster(:id1))
#   0.435500 seconds (495.96 k allocations: 1.381 GiB, 15.97% gc time, 10.72% compilation time)
@time reg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)))
#  1.367264 seconds (1.91 M allocations: 1.592 GiB, 5.74% gc time, 23.85% compilation time: 20% of which was recompilation)

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
#   1.546023 seconds (19.89 k allocations: 119.673 MiB, 1.70% gc time)




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
#   0.311420 seconds (1.35 k allocations: 1.052 GiB, 22.30% gc time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2)), Vcov.cluster(:id1))
#   0.808992 seconds (3.52 k allocations: 1.272 GiB, 8.68% gc time)
@time reg(df, @formula(ln_y ~ X1 + fe(id1) + fe(id2) + fe(id3)), Vcov.cluster(:id1))
# 0.950808 seconds (4.75 k allocations: 1.496 GiB, 7.48% gc time)
