
using DataFrames, Base.Test

df = CSV.read(joinpath(dirname(@__FILE__), "..", "dataset/Cigar.csv"))
df[:pState] = categorical(df[:State])
df[:pYear] = categorical(df[:Year])


for method in [:cholesky, :qr, :lsmr]
	model = @model Sales ~ Price fe = pYear save = true method = $(method)
	result = reg(df, model)
	@test result.augmentdf[1, :pYear] ≈ 164.77833189721005

	model = @model Sales ~ Price fe = pYear + pState save = true method = $(method)
	result = reg(df, model)
	@test result.augmentdf[1, :pYear] + result.augmentdf[1, :pState] ≈ 140.6852 atol = 1e-3

	model = @model Sales ~ Price fe = Year&pState save = true method = $(method)
	result = reg(df, model)
	@test result.augmentdf[1, :YearxpState] ≈ 1.742779  atol = 1e-3

	model = @model Sales ~ Price fe = pState + Year&pState save = true method = $(method)
	result = reg(df, model)
	@test result.augmentdf[1, :pState] ≈ -91.690635 atol = 1e-1

	model = @model Sales ~ Price fe = pState save = true subset = (State .<= 30) method = $(method)
	result = reg(df, model)

	@test ismissing(result.augmentdf[1380,:pState])
end
	# add test with IV & weight
