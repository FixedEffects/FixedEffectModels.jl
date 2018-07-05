
using DataFrames, CSV, Test, FixedEffectModels

df = CSV.read(joinpath(dirname(@__FILE__), "..", "dataset/Cigar.csv"))
df[:pState] = categorical(df[:State])
df[:pYear] = categorical(df[:Year])

if Base.USE_GPL_LIBS
	method_s = [:cholesky, :qr, :lsmr, :lsmr_parallel, :lsmr_threads]
else
	method_s = [:lsmr, :lsmr_parallel, :lsmr_threads]
end

for method in method_s
	model = @model Sales ~ Price fe = pYear save = true method = $(method)
	result = reg(df, model)
	@test fes(result)[1, :pYear] ≈ 164.77833189721005

	model = @model Sales ~ Price fe = pYear + pState save = true method = $(method)
	result = reg(df, model)
	@test fes(result)[1, :pYear] + fes(result)[1, :pState] ≈ 140.6852 atol = 1e-3

	model = @model Sales ~ Price fe = Year&pState save = true method = $(method)
	result = reg(df, model)
	@test fes(result)[1, :YearxpState] ≈ 1.742779  atol = 1e-3

	model = @model Sales ~ Price fe = pState + Year&pState save = true method = $(method)
	result = reg(df, model)
	@test fes(result)[1, :pState] ≈ -91.690635 atol = 1e-1

	model = @model Sales ~ Price fe = pState save = true subset = (State .<= 30) method = $(method)
	result = reg(df, model)
	@test fes(result)[1,:pState] ≈  124.913976 atol = 1e-1
	@test ismissing(fes(result)[1380,:pState])
end
	# add test with IV & weight


df[:Price2] = df[:Price]
model = @model Sales ~ Price + Price2 fe = pYear save = true
result = reg(df, model)
@test fes(result)[1, :pYear] ≈ 164.77833189721005

model = @model Sales ~ (State ~ Price) fe = pYear save = true
result = reg(df, model)
@test fes(result)[1, :pYear] ≈ -167.48093490413623


model = @model Sales ~ (State ~ Price + Price2) fe = pYear save = true
result = reg(df, model)
@test fes(result)[1, :pYear] ≈ -167.48093490413623

# check does not change r2
model1 = @model Sales ~ Price weights = Pop save = true
result1 = reg(df, model1)
model2 = @model Sales ~ Price weights = Pop 
result2 = reg(df, model2)
@test r2(result1) ≈ r2(result2)



# check with weights
model = @model Sales ~ Price weights = Pop fe = pYear save = true
result = reg(df, model)
@test fes(result)[2, :pYear] -  fes(result)[1, :pYear] ≈ -3.0347149502496222
