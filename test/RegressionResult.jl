using FixedEffectModels, DataFrames, CSV, Test

df = CSV.read(joinpath(dirname(pathof(FixedEffectModels)), "../dataset/Cigar.csv"))
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
@test r2(result) ≈ 0.0968815737054879 atol = 1e-4
@test adjr2(result) ≈ 0.0962261902321246 atol = 1e-4
@test result.nobs == 1380
@test result.F ≈ 147.8242550248069 atol= 1e-4

model = @model Sales ~ Price fe = pState
result = reg(df, model, save = true)
@test result.augmentdf[:residuals][1:10] ≈ [-22.08499, -20.33318, -17.23318, -18.97645, -19.85547, -26.1161, -24.20627, -23.87674, -17.62624, -12.01018] atol = 1e-4
@test result.nobs == 1380
@test r2(result) ≈ 0.7682403747044817 atol = 1e-4
@test adjr2(result) ≈ 0.7602426682051615 atol = 1e-4
@test result.F ≈ 458.4582526109375 atol = 1e-4

#To do: add iv, weights, subset

# iv
model = @model Sales ~ CPI + (Price ~ Pimin)  fe = pState
result = reg(df, model)
@test residuals(result, df)[1:10] ≈ [16.92575, -14.83571, -12.01704, -13.4687, -14.86343, -20.03415, -18.72394, -18.16776, -11.4186, -6.72175] atol = 1e-4
@test r2(result) ≈ 0.8092066369504399 atol = 1e-4
@test adjr2(result) ≈ 0.8024744387046972
@test result.nobs == 1380
@test result.F ≈ 367.5570114542729 atol = 1e-4

# iv and weights
model = @model Sales ~ CPI + (Price ~ Pimin) weights = Pop fe = pState
result = reg(df, model)
@test residuals(result, df)[1:10] ≈ [-17.78992, -15.74522, -12.89908, -14.39665, -15.74204, -21.06022, -19.70171, -19.19843, -12.52975, -7.739035] atol = 1e-4
@test result.nobs == 1380
@test result.F ≈ 823.8453331087294 atol = 1e-4
@test r2(result) ≈ 0.79689865227595 atol = 1e-7
@test adjr2(result) ≈ 0.789732163279681 atol = 1e-7


# iv, weights and subset of states
model = @model Sales ~ CPI + (Price ~ Pimin) weights = Pop fe = pState subset = (State . <= 30)
result = reg(df, model)
@test residuals(result, df)[1:10] ≈ [-19.33225, -17.27341, -14.41067, -14.39665, -15.87715, -17.18928,-22.45813, -21.03584 -20.46070, -13.73339, -8.89981] atol = 1e-4
@test r2(result) ≈ 0.7870039142818601 atol = 1e-7
@test adjr2(result) ≈ 0.7793676909782649 atol = 1e-7
@test result.F ≈ 432.4710309765751 atol = 1e-4
@test result.nobs == 810

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


# fixed effects
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

# add test with IV and weights
model = @model Sales ~ (Price ~ Pimin) weights = Pop fe = pYear save = true
result = reg(df, model)
@test fes(result)[1, :pYear] ≈ -60.27375767845 atol = 1e-4
@test result.F ≈ 207.6844777052426 atol = 1e-4
@test result.nobs == 1380
@test r2(result) ≈ 0.4116529168557033 atol = 1e-4
@test adjr2(result) ≈ 0.3985688453254371 atol 1e-4


# add test with IV, weights and both year and state fixed effects
model = @model Sales ~ (Price ~ Pimin) weights = Pop fe = pSate + pYear save = true
result = reg(df, model)
@test fes(result)[1, :pYear] ≈ -48.88905121436 atol = 1e-4
@test fes(result)[1, :pState] ≈ -13.76664496078 atol = 1e-4
@test result.F ≈ 123.4723488142459 atol = 1e-4
@test result.nobs == 1380



# test subset with IV
model = @model Sales ~ (Price ~ Pimin) fe = pYear save = true subset = (State .<= 30)
result = reg(df, model)
@test fes(result)[1, :pYear] ≈ -38.74806204604 atol = 1e-4
@test result.F ≈ 10.68898610642372 atol = 1e-4
@test result.nobs == 810
@test r2(result) ≈ 0.1959052506984742 atol = 1e-4
@test adjr2(result)  ≈ 0.164938829030893 atol = 1e-4


# test subset with IV, weights and year fixed effects
model = @model Sales ~ (State ~ Price) weights = Pop fe = pYear save = true subset = (State .<= 30)
result = reg(df, model)
@test fes(result)[1, :pYear] ≈ -82.60387678884 atol = 1e-4
@test result.F ≈ 120.905346062158 atol = 1e-4
@test result.nobs == 1380
@test r2(result) ≈ 0.3726730979288025 atol = 1e-4
@test adjr2(result) ≈ 0.3485141671686795 atol = 1e-4

# test subset with IV, weights and year fixed effects
model = @model Sales ~ (State ~ Price) weights = Pop fe = pState + pYear save = true subset = (State .<= 30)
result = reg(df, model)
@test fes(result)[1, :pYear] ≈ -82.60387678884 atol = 1e-4
@test fes(result)[1, :pState] ≈ -69.30766771529 atol = 1e-4
@test result.F ≈ 65.78323223751296 atol = 1e-4
@test result.nobs == 1380
@test r2(result) ≈ 0.7799730045711415 atol = 1e-4
@test adjr2(result) ≈ 0.3485141671686795 atol = 1e-4
