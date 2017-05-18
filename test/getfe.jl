
using DataFrames, FixedEffectModels, Base.Test

df = readtable(joinpath(dirname(@__FILE__), "..", "dataset/Cigar.csv.gz"))
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])


for method in [:cholesky, :qr, :lsmr]
	result = @reg df Sales ~ Price fe = pYear save = true method = method
	@test result.augmentdf[1, :pYear] ≈ 164.77833189721005

	result = @reg df Sales ~ Price fe = pYear + pState save = true method = method
	@test result.augmentdf[1, :pYear] + result.augmentdf[1, :pState] ≈ 140.6852 atol = 1e-3

	result = @reg df Sales ~ Price fe = Year&pState save = true method = method
	@test result.augmentdf[1, :YearxpState] ≈ 1.742779  atol = 1e-3

	result = @reg df Sales ~ Price fe = pState + Year&pState save = true method = method
	@test result.augmentdf[1, :pState] ≈ -91.690635 atol = 1e-1

	result = @reg df Sales ~ Price fe = pState save = true subset = (df[:State] .<= 30) method = method

	@test isna(result.augmentdf[1380,:pState])
end
	# add test with IV & weight
