using DataFrames, FixedEffectModels, Random
N = 10000000
K = 100
id1 = rand(1:div(N, K), N)
id2 = rand(1:K, N))
w = cos.(id1)


x1 = 5 * cos.(id1) + 5 * sin.(id2) + randn(N)
x2 =  cos.(id1) +  sin.(id2) + randn(N)
y= 3 .* x1 .+ 5 .* x2 .+ cos.(id1) .+ cos.(id2).^2 .+ randn(N)



df = DataFrame(id1 = categorical(id1), id2 = categorical(id2), x1 = x1, x2 = x2, w = w, y = y)
df.x3 =  cos.(id1) + sin.(id2) + randn(N)
df.x4 =  cos.(id1) + sin.(id2) + randn(N)
df.x5 =  cos.(id1) + sin.(id2) + randn(N)
df.x6 =  cos.(id1) + sin.(id2) + randn(N)
df.x7 =  cos.(id1) + sin.(id2) + randn(N)




@time reg(df, @model(y ~ x1 + x2))
# 0.582029 seconds (852 allocations: 535.311 MiB, 18.28% gc time)
@time reg(df, @model(y ~ x1 + x2, vcov = cluster(id2)))
# 0.809652 seconds (1.55 k allocations: 649.821 MiB, 14.40% gc time)
@time reg(df, @model(y ~ x1 + x2 + fe(id1)))
# 1.655732 seconds (1.21 k allocations: 734.353 MiB, 16.88% gc time)
@time reg(df, @model(y ~ x1 + x2 + fe(id1), vcov = cluster(id1)))
#  2.113248 seconds (499.66 k allocations: 811.751 MiB, 15.08% gc time)
@time reg(df, @model(y ~ x1 + x2 + fe(id1) + fe(id2)))
# 3.553678 seconds (1.36 k allocations: 1005.101 MiB, 10.55% gc time))
@time reg(df, @model(y ~ x1 + x2 + fe(id1) + fe(id2), vcov = cluster(id1 + id2)))


# more regressors
@time reg(df, @model(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + fe(id1))
#  5.654325 seconds (3.42 k allocations: 3.993 GiB, 18.19% gc time)
@time reg(df, @model(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + fe(id1) + fe(id2))
# 10.520039 seconds (3.50 k allocations: 4.334 GiB, 6.19% gc time)
@time reg(df, @model(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + fe(id1)+ fe(id2)), method = :lsmr_threads)
#  9.591568 seconds (3.94 k allocations: 6.372 GiB, 5.24% gc time)

