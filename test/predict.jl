using FixedEffectModels, DataFrames, CSV, Test

df = CSV.read(joinpath(dirname(pathof(FixedEffectModels)), "../dataset/Cigar.csv"))


##############################################################################
##
## Printing Results
##
##############################################################################

model = @formula Sales ~ NDI
result = reg(df, model)
show(result)
predict(result, df)
residuals(result, df)
@test responsename(result) == "Sales"


model = @formula Sales ~ CPI + (Price ~ Pimin)
result = reg(df, model)
show(result)
predict(result, df)
residuals(result, df)
@test  nobs(result) == 1380
@test vcov(result)[1] ≈ 3.5384578251636785

model = @formula Sales ~ Price + fe(State)
result = reg(df, model)
show(result)
model = @formula Sales ~ CPI + (Price ~ Pimin) + fe(State)
result = reg(df, model)
show(result)



##############################################################################
##
## Saved Residuals
##
##############################################################################

model = @formula Sales ~ Price
result = reg(df, model)
@test residuals(result, df)[1:10] ≈ [-39.2637, -37.48801, -34.38801, -36.09743, -36.97446, -43.15547, -41.22573, -40.83648, -34.52427, -28.91617] atol = 1e-4
@test r2(result) ≈ 0.0968815737054879 atol = 1e-4
@test adjr2(result) ≈ 0.0962261902321246 atol = 1e-4
@test result.nobs == 1380
@test result.F ≈ 147.8242550248069 atol= 1e-4

#weights
model = @formula Sales ~ CPI
result = reg(df, model, weights = :Pop)
@test residuals(result, df)[1:3] ≈ [ -35.641449,  -34.0611538,  -30.860784] atol = 1e-4

# iv
model = @formula Sales ~ CPI + (Price ~ Pimin) 
result = reg(df, model)
@test residuals(result, df)[1:3] ≈ [ -33.047390, -30.9518422, -28.1371048] atol = 1e-4

# iv and weights
model = @formula Sales ~ CPI + (Price ~ Pimin)
result = reg(df, model, weights = :Pop)
@test residuals(result, df)[1:3] ≈ [ -30.2284549, -28.09507, -25.313248] atol = 1e-4

# iv, weights and subset of states
model = @formula Sales ~ CPI + (Price ~ Pimin)
result = reg(df, model, subset = df.State .<= 30, weights = :Pop)
@test residuals(result, df)[1:3] ≈ [ -34.081720, -31.906020, -29.131738] atol = 1e-4


# fixed effects
model = @formula Sales ~ Price + fe(State)
result = reg(df, model, save = true)
@test residuals(result)[1:3] ≈ [-22.08499, -20.33318, -17.23318] atol = 1e-4
@test result.nobs == 1380
@test r2(result) ≈ 0.7682403747044817 atol = 1e-4
@test adjr2(result) ≈ 0.7602426682051615 atol = 1e-4
@test result.F ≈ 458.4582526109375 atol = 1e-4

# fixed effects and weights
model = @formula Sales ~ Price + fe(State)
result = reg(df, model,  weights = :Pop, save = true)
@test residuals(result)[1:3] ≈ [ -23.413793, -21.65289, -18.55289] atol = 1e-4

# fixed effects and iv
#TO CHECK WITH IVREGHDFE, NO SUPPORT RIGHT NOW
model = @formula Sales ~ CPI + (Price ~ Pimin) + fe(State)
result = reg(df, model, save = true)
@test residuals(result)[1:3] ≈ [ -16.925748, -14.835710, -12.017037] atol = 1e-4



# test different arguments for the keyword argument save
model = @formula Sales ~ Price + fe(State)
result = reg(df, model, save = true)
@test residuals(result) !== nothing
@test "fe_State" ∈ names(fe(result))

model = @formula Sales ~ Price + fe(State)
result = reg(df, model, save = :residuals)
@test residuals(result) !== nothing
@test "fe_State" ∉ names(fe(result))

model = @formula Sales ~ Price + fe(State)
result = reg(df, model, save = :fe)
@test residuals(result) === nothing
@test "fe_State" ∈ names(fe(result))



##############################################################################
##
## Saved FixedEffects
##
##############################################################################
# check save does not change r2
model1 = @formula Sales ~ Price
result1 = reg(df, model1, weights = :Pop)
model2 = @formula Sales ~ Price
result2 = reg(df, model2, weights = :Pop)
@test r2(result1) ≈ r2(result2)

methods_vec = [:lsmr, :lsmr_threads, :lsmr_cores]


model = @formula Sales ~ Price + fe(Year)
result = reg(df, model, save = true)
@test fe(result)[1, :fe_Year] ≈ 164.77833189721005

model = @formula Sales ~ Price + fe(Year) + fe(State)
result = reg(df, model, save = true)
@test fe(result)[1, :fe_Year] + fe(result)[1, :fe_State] ≈ 140.6852 atol = 1e-3

model = @formula Sales ~ Price + Year&fe(State)
result = reg(df, model, save = true)
@test fe(result)[1, Symbol("fe_State&Year")] ≈ 1.742779  atol = 1e-3

model = @formula Sales ~ Price + fe(State) + Year&fe(State)
result = reg(df, model, save = true)
@test fe(result)[1, :fe_State] ≈ -91.690635 atol = 1e-1

model = @formula Sales ~ Price + fe(State)
result = reg(df, model, subset = df.State .<= 30, save = true)
@test fe(result)[1, :fe_State] ≈  124.913976 atol = 1e-1
@test ismissing(fe(result)[1380 , :fe_State])

model = @formula Sales ~ Price + fe(Year)
result = reg(df, model, weights = :Pop, save = true)
@test fe(result)[2, :fe_Year] -  fe(result)[1, :fe_Year] ≈ -3.0347149502496222

# fixed effects
df.Price2 = df.Price
model = @formula Sales ~ Price + Price2 + fe(Year)
result = reg(df, model, save = true)
@test fe(result)[1, :fe_Year] ≈ 164.77833189721005

# iv
model = @formula Sales ~ (State ~ Price) + fe(Year)
result = reg(df, model, save = true)
@test fe(result)[1, :fe_Year] ≈ -167.48093490413623

# weights
model = @formula Sales ~ Price + fe(Year)
result = reg(df, model, weights = :Pop, save = true)
@test fe(result)[2, :fe_Year] -  fe(result)[1, :fe_Year] ≈ -3.0347149502496222

# IV and weights
model = @formula Sales ~ (Price ~ Pimin) + fe(Year)
result = reg(df, model, weights = :Pop, save = true)
@test fe(result)[1, :fe_Year] ≈ 168.24688 atol = 1e-4


# IV, weights and both year and state fixed effects
model = @formula Sales ~ (Price ~ Pimin) + fe(State) + fe(Year)
result = reg(df, model, weights = :Pop, save = true)
@test fe(result)[1, :fe_Year] + fe(result)[1, :fe_State]≈ 147.84145 atol = 1e-4


# subset with IV
model = @formula Sales ~ (Price ~ Pimin) + fe(Year)
result = reg(df, model, subset = df.State .<= 30, save = true)
@test fe(result)[1, :fe_Year] ≈ 164.05245824240276 atol = 1e-4
@test ismissing(fe(result)[811, :fe_Year])


# subset with IV, weights and year fixed effects
model = @formula Sales ~ (Price ~ Pimin) + fe(Year)
result = reg(df, model, subset = df.State .<= 30, weights = :Pop, save = true)
@test fe(result)[1, :fe_Year] ≈ 182.71915 atol = 1e-4

# subset with IV, weights and year fixed effects
model = @formula Sales ~ (Price ~ Pimin) + fe(State) + fe(Year)
result = reg(df, model, subset = df.State .<= 30, weights = :Pop, save = true)
@test fe(result)[1, :fe_Year] + fe(result)[1, :fe_State] ≈ 158.91798 atol = 1e-4

methods_vec = [:cpu]
if FixedEffectModels.FixedEffects.has_cuarrays()
	push!(methods_vec, :gpu)
end
for method in methods_vec
	local model = @formula Sales ~ Price + fe(Year)
	local result = reg(df, model, save = true, method = method, double_precision = false)
	@test fe(result)[1, :fe_Year] ≈ 164.7 atol = 1e-1
end



