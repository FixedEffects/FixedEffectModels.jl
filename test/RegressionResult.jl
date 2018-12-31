using FixedEffectModels, DataFrames, CSV, Test

df = CSV.read(joinpath(dirname(@__FILE__), "..", "dataset/Cigar.csv"))
df[:pState] = categorical(df[:State])
df[:pYear] = categorical(df[:Year])



##############################################################################
##
## Printing Results
##
##############################################################################

model = @model Sales ~ NDI
result = reg(df, model)
show(result)
predict(result, df)
residuals(result, df)



model = @model Sales ~ CPI + (Price ~ Pimin)
result = reg(df, model)
show(result)
predict(result, df)
residuals(result, df)
response(result, df)
@test  nobs(result) == 1380
@test vcov(result)[1] ≈ 3.5384578251636785

model = @model Sales ~ Price  fe = pState
result = reg(df, model)
show(result)
model = @model Sales ~ CPI + (Price ~ Pimin) fe = pState
result = reg(df, model)
show(result)




##############################################################################
##
## Saved Residuals
##
##############################################################################

model = @model Sales ~ Price
result = reg(df, model)
@test residuals(result, df)[1:10] ≈ [-39.2637, -37.48801, -34.38801, -36.09743, -36.97446, -43.15547, -41.22573, -40.83648, -34.52427, -28.91617] atol = 1e-4


model = @model Sales ~ Price fe = pState
result = reg(df, model, save = true)
@test result.augmentdf[:residuals][1:10] ≈ [-22.08499, -20.33318, -17.23318, -18.97645, -19.85547, -26.1161, -24.20627, -23.87674, -17.62624, -12.01018] atol = 1e-4

#To do: add iv, weights, subset

# test different arguments for the keyword argument save
model = @model Sales ~ Price fe = pState
result = reg(df, model, save = true)
@test :residuals ∈ names(result.augmentdf)
@test :pState ∈ names(result.augmentdf)

model = @model Sales ~ Price fe = pState
result = reg(df, model, save = :residuals)
@test :residuals ∈ names(result.augmentdf)
@test :pState ∉ names(result.augmentdf)

model = @model Sales ~ Price fe = pState
result = reg(df, model, save = :fe)
@test :residuals ∉ names(result.augmentdf)
@test :pState ∈ names(result.augmentdf)



##############################################################################
##
## Saved FixedEffects
##
##############################################################################

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


df[:Price2] = df[:Price]
model = @model Sales ~ Price + Price2 fe = pYear save = true
result = reg(df, model)
@test fes(result)[1, :pYear] ≈ 164.77833189721005

# iv
model = @model Sales ~ (State ~ Price) fe = pYear save = true
result = reg(df, model)
@test fes(result)[1, :pYear] ≈ -167.48093490413623

# weights
model = @model Sales ~ Price weights = Pop fe = pYear save = true
result = reg(df, model)
@test fes(result)[2, :pYear] -  fes(result)[1, :pYear] ≈ -3.0347149502496222

# check save does not change r2
model1 = @model Sales ~ Price weights = Pop save = true
result1 = reg(df, model1)
model2 = @model Sales ~ Price weights = Pop 
result2 = reg(df, model2)
@test r2(result1) ≈ r2(result2)

# add test with IV, weights and both





