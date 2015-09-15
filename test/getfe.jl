
using RDatasets, DataFrames, FixedEffectModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])

result = reg(Sales ~ Price |> pYear, df, save = true)
@test_approx_eq  result.augmentdf[1, :pYear] 164.77833189721005

result = reg(Sales ~ Price |> pYear + pState, df, save = true)
@test_approx_eq_eps  result.augmentdf[1, :pYear] + result.augmentdf[1, :pState]  140.6852 1e-3

result = reg(Sales ~ Price |> Year&pState, df, save = true)
@test_approx_eq_eps result.augmentdf[1, :YearxpState]  1.742779  1e-3

result = reg(Sales ~ Price |> pState + Year&pState, df, save = true)
@test_approx_eq_eps  result.augmentdf[1, :pState]  -91.690635 1e-1

result = reg(Sales ~ Price |> pState, df, save = true, subset = (df[:State] .<= 30))

@test isna(result.augmentdf[1380,:pState])


# add test with IV





srand(1234)
N = 5000
T = 500
l1 = randn(N)
l2 = randn(N)
f1 = randn(T)
f2 = randn(T)
x1 = Array(Float64, N*T)
y = Array(Float64, N*T)
id = Array(Int64, N*T)
time = Array(Int64, N*T)
index = 0
function fillin(id, time, x1, y, N, T)
	index = 0
	@inbounds for i in 1:N
		for j in 1:T
			index += 1
			id[index] = i
			time[index] = j
			x1[index] = 4 + f1[j] * l1[i] + 3 * f2[j] * l2[i] + l1[i]^2 + randn()
			y[index] = 5 + 3 * x1[index] + 4 * f1[j] * l1[i] + f2[j] * l2[i] + randn()
		end
	end
end

fillin(id, time, x1, y, N, T)
df = DataFrame(id = pool(id), time = pool(time), x1 = x1, y = y)
subset = rand(1:5, N*T) .> 1
unbalanceddf = df[subset, :]
subset = rand(1:5, N*T) .== 1
sparsedf = df[subset, :]

