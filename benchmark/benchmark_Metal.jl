using DataFrames, Random, Metal, FixedEffectModels
Random.seed!(1234)
# More complicated setup
N = 8_000_000 # number of observations
M = 400_000 # number of workers
O = 50_000 # number of firms
id1 = rand(1:M, N)
id2 = [rand(max(1, div(x, 8)-10):min(O, div(x, 8)+10)) for x in id1]
x1 = 5 * cos.(id1) + 5 * sin.(id2) + randn(N)
x2 =  cos.(id1) +  sin.(id2) + randn(N)
x3 =  cos.(id1) +  sin.(id2) + randn(N)
y= 3 .* x1 .+ 5 .* x2 .+ cos.(id1) .+ cos.(id2).^2 .+ randn(N)
df = DataFrame(id1 = id1, id2 = id2, x1 = x1, x2 = x2, x3 = x3, y = y)
@time reg(df, @formula(y ~ x1 + x2 + x3 + fe(id1) + fe(id2)), maxiter = 200, double_precision = false)
#  30.002852 seconds (66.42 M allocations: 4.917 GiB, 0.83% gc time, 16.25% compilation time: 6% of which was recompilation)
@time reg(df, @formula(y ~ x1 + x2 + x3 + fe(id1) + fe(id2)), method = :Metal, maxiter = 200)
#  16.634684 seconds (3.50 M allocations: 1.407 GiB, 0.69% gc time, 1.49% compilation time: <1% of which was recompilation)



