using FixedEffectModels, DataFrames, CategoricalArrays, CSV, Test



@testset "print results" begin

	df = DataFrame(CSV.File(joinpath(dirname(pathof(FixedEffectModels)), "../dataset/Cigar.csv")))
	df.StateC = categorical(df.State)

	model = @formula Sales ~ NDI
	result = reg(df, model)
	show(result)
	predict(result, df)
	residuals(result, df)
	@test responsename(result) == "Sales"

	model = @formula Sales ~ CPI + (Price ~ Pimin)
	result = reg(df, model)
	coeftable(result)
	show(result)
	predict(result, df)
	residuals(result, df)
	@test  nobs(result) == 1380
	@test vcov(result)[1] ≈ 3.5384578251636785

	# predict with interactions
	model = @formula Sales ~ CPI * Pop
	result = reg(df, model)
	@test predict(result, df)[1] ≈ 131.92991


	model = @formula Sales ~ Price + fe(State)
	result = reg(df, model)
	show(result)
	model = @formula Sales ~ CPI + (Price ~ Pimin) + fe(State)
	result = reg(df, model)
	show(result)
end

@testset "Predict" begin
    # Simple - one binary FE
    df = DataFrame(x = rand(100), g = rand(["a", "b"], 100))
    df.y = 1.0 .+ 0.5 .* df.x .+ (df.g .== "b")
    m = reg(df, @formula(y ~ x + fe(g)); save = :fe)
    pred = predict(m, df)
    @test pred ≈ df.y

    # One group only 
    df = DataFrame(x = rand(100), g = "a")
    df.y = 1.0 .+ 0.5 .* df.x
    m = reg(df, @formula(y ~ x + fe(g)); save = :fe)
    pred = predict(m, df)
    @test pred ≈ df.y

    # Two groups and predict df has a level that's missing from model
    df = DataFrame(x = rand(100), g = rand(["a", "b"], 100))
    df.y = 1.0 .+ 0.5 .* df.x .+ (df.g .== "b")
    m = reg(df, @formula(y ~ x + fe(g)); save = :fe)
    pred = predict(m, DataFrame(x = [1.0, 2.0], g = ["a", "c"]))
    @test ismissing(pred[2])

    # Two groups + missing observation of FE
    df = DataFrame(x = rand(100), g = [missing; rand(["a", "b"], 99)])
    df.y = 1.0 .+ 0.5 .* df.x .+ isequal.(df.g, "b")
    m = reg(df, @formula(y ~ x + fe(g)), save = :fe)
    pred = predict(m, df)
    @test pred isa Vector{Union{Missing, Float64}}
    @test ismissing(pred[1])
    @test norm(pred[2:end] .- df.y[2:end]) < 1e-6

    # Two groups + missing observation of non-FE
    df = DataFrame(x = [missing; rand(99)], g = rand(["a", "b"], 100))
    df.y = 1.0 .+ 0.5 .* df.x .+ isequal.(df.g, "b")
    m = reg(df, @formula(y ~ x + fe(g)), save = :fe)
    pred = predict(m, df)
    @test pred isa Vector{Union{Missing, Float64}}
    @test ismissing(pred[1])
    @test pred[2] ≈ df.y[2]

    # Two groups + two FEs
    df = DataFrame(x = rand(100), g1 = rand(["a", "b"], 100), g2 = rand(["c", "d"], 100))
    df.y = 1.0 .+ 0.5 .* df.x .+ isequal.(df.g1, "b") .+ (df.g2 .== "d") * 2
    m = reg(df, @formula(y ~ x + fe(g1) + fe(g2)); save = :fe)
    pred = predict(m, df)
    @test pred isa Vector{Float64}
    @test pred ≈ df.y

    # Two groups + two FEs, missing one FE
    df = DataFrame(x = rand(100), g1 = rand(["a", "b"], 100), g2 = [missing; rand(["c", "d"], 99)])
    df.y = 1.0 .+ 0.5 .* df.x .+ isequal.(df.g1, "b") .+ isequal.(df.g2,"d") * 2
    m = reg(df, @formula(y ~ x + fe(g1) + fe(g2)); save = :fe)
    pred = predict(m, df)
    @test pred isa Vector{Union{Missing, Float64}}
    @test ismissing(pred[1])
    @test pred[2:end] ≈ df.y[2:end]

    # Three FEs, "middle" one missing
    df = DataFrame(x = rand(100), g1 = rand(["a", "b"], 100), g2 = [missing; rand(["c", "d"], 99)],
                    g3 = rand(["e", "f"], 100))
    df.y = 1.0 .+ 0.5 .* df.x .+ isequal.(df.g1, "b") .+ isequal.(df.g2, "d") * 2 .+
            isequal.(df.g3, "e")
    m = reg(df, @formula(y ~ x + fe(g1) + fe(g2) + fe(g3)); save = :fe)
    pred = predict(m, df)
    @test pred isa Vector{Union{Missing, Float64}}
    @test ismissing(pred[1])
    @test pred[2:end] ≈ df.y[2:end]

    # Interactive FE 
    df = DataFrame(x = rand(100), g1 = rand(["a", "b"], 100), g2 = rand(["c", "d"], 100))
    df.y = 1.0 .+ 0.5 .* df.x .+ (df.g1 .== "b") .+ (df.g1 .== "b" .&& df.g2 .== "d")
    m = reg(df, @formula(y ~ x + fe(g1)&fe(g2)); save = :fe)
    pred = predict(m, df)
    @test pred isa Vector{Float64}
    @test pred ≈ df.y

    # Interactive FE + missing 
    df = DataFrame(x = rand(100), g1 = rand(["a", "b"], 100), g2 = [missing; rand(["c", "d"], 99)])
    df.y = 1.0 .+ 0.5 .* df.x .+ (df.g1 .== "b") .+ (isequal.(df.g1, "b") .&& isequal.(df.g2, "d"))
    m6 = reg(df, @formula(y ~ x + fe(g1)&fe(g2)); save = :fe)
    pred = predict(m, df)
    @test pred isa Vector{Union{Missing, Float64}}
    @test length(pred) == nrow(df)
    @test ismissing(pred[1])
    @test pred[2:end] ≈ df.y[2:end]

    # Interaction with continuous variable
    df = DataFrame(x = rand(100), g = rand(["a", "b"], 100), z = rand(100))
    df.y = 1.0 .+ 0.5 .* df.x .+ 2.0 .* (df.g .== "b") .* df.z 
    m = reg(df, @formula(y ~ x + fe(g)&z); save = :fe)
    @test_throws ArgumentError pred = predict(m, df)
    #@test pred ≈ df.y # Once implemented

    # Interaction with continuous variable, FE missing 
    df = DataFrame(x = rand(100), g = [missing; rand(["a", "b"], 99)], z = rand(100))
    df.y = 1.0 .+ 0.5 .* df.x .+ 2.0 .* (df.g .== "b") .* df.z 
    m = reg(df, @formula(y ~ x + fe(g)&z); save = :fe)
    @test_throws ArgumentError pred = predict(m, df)
    #@test pred ≈ df.y

    # Interaction with continuous variable, cont var missing
    df = DataFrame(x = rand(100), g = rand(["a", "b"], 100), z = [missing; rand(99)])
    df.y = 1.0 .+ 0.5 .* df.x .+ 2.0 .* (df.g .== "b") .* df.z 
    m = reg(df, @formula(y ~ x + fe(g)&z); save = :fe)
    @test_throws ArgumentError pred = predict(m, df)
    #@test pred ≈ df.y

    # Regular FE + another FE interacted with continuous variable
    df = DataFrame(x = rand(100), g1 = rand(["a", "b"], 100), g2 = rand(["c", "d"], 100), z = rand(100))
    df.y = 1.0 .+ 0.5 .* df.x .+ 2.0 .* (df.g2 .== "b") .* df.z  .+ 3.0 .* (df.g1 .== "b")
    m = reg(df, @formula(y ~ x + fe(g1) + fe(g2)&z); save = :fe)
    @test_throws ArgumentError pred = predict(m, df)
    #@test pred ≈ df.y

    # Two continuous/FE interactions
    m = reg(df, @formula(y ~ x + fe(g1) + fe(g2)&z + fe(g1)&x); save = :fe)
    @test_throws ArgumentError pred = predict(m, df)

    # Regular FE + interacted FE + FE/continuous interaction
    df = DataFrame(x = rand(100), g1 = rand(["a", "b"], 100), g2 = rand(["c", "d"], 100), 
        g3 = rand(["e", "f"], 100), g4 = rand(["g", "h"], 100), z = rand(100))
    df.y = 1.0 .+ 0.5 .* df.x .+ 2.0 .* (df.g1 .== "b") .+ 3.0 .* (df.g2 .== "d") .* (df.g3 .== "f") .+
        4.0 .* (df.g4 .== "h") .* df.z
    m = reg(df, @formula(y ~ x + fe(g1) + fe(g2)&fe(g3) + fe(g4)&z))
    @test_throws ArgumentError pred = predict(m, df)


	# only fixed effects
	df = DataFrame(y=rand(10), id = rand(1:2, 10), t = rand(1:2, 10))
	out1 = predict(reg(df, @formula(y ~ fe(id) + fe(t)), save = :fe), df)
	out2 = predict(reg(df, @formula(y ~ 1 + fe(id) + fe(t)), save = :fe), df)
	@test all(out1 .≈ out2)
end

@testset "Continuous/FE detection" begin
    # Regular interaction is fine as handled by StatsModels
    @test FixedEffectModels.has_cont_fe_interaction(@formula(y ~ x + y&z)) == false
    
    # FE/FE interaction also works
    @test FixedEffectModels.has_cont_fe_interaction(@formula(y ~ x + fe(y)&fe(z))) == false

    # Interaction of FEs with continuous variable requires special handling, currently not implemented
    @test FixedEffectModels.has_cont_fe_interaction(@formula(y ~ x + fe(y)&z)) == true
    @test FixedEffectModels.has_cont_fe_interaction(@formula(y ~ x + y&fe(z))) == true
    @test FixedEffectModels.has_cont_fe_interaction(@formula(y ~ x + fe(y)&fe(z)&a)) == true

    # Interaction of continuous with non-FE function term again handled by StatsModels
    @test FixedEffectModels.has_cont_fe_interaction(@formula(y ~ x + y^2&z)) == false
end


@testset "residuals" begin

	df = DataFrame(CSV.File(joinpath(dirname(pathof(FixedEffectModels)), "../dataset/Cigar.csv")))
	df.StateC = categorical(df.State)

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

	# iv with exo after endo
	model = @formula Sales ~ (Price ~ Pimin) + CPI
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

	#r2 with weights when saving residuals
	m = @formula Sales ~ Price
	result = reg(df, save = :residuals, weights = :Pop, m)
	@test r2(result) ≈ 0.24654260 atol = 1e-4


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
	@test "fe_State" ∈ names(fe(result))

	# iv recategorized
	df.Pimin2 = df.Pimin
	m = @formula Sales ~ (Pimin2 + Price ~ NDI + Pimin)
	result = reg(df, m)
	yhat = predict(result, df)
	res = residuals(result, df)

	m2 = @formula Sales ~ Pimin2 + (Price ~ NDI + Pimin)
	result2 = reg(df, m2)
	yhat2 = predict(result2, df)
	res2 = residuals(result2, df)
	@test yhat ≈ yhat2
	@test res ≈ res2

	m3 = @formula Sales ~ Pimin2 + (Price ~ NDI)
	result3 = reg(df, m3)
	yhat3 = predict(result3, df)
	res3 = residuals(result3, df)
	@test yhat ≈ yhat3
	@test res ≈ res3

	m4 = @formula Sales ~ (Price + Pimin2 ~ NDI + Pimin)
	result4 = reg(df, m4)
	yhat4 = predict(result4, df)
	res4 = residuals(result4, df)
	@test yhat ≈ yhat4
	@test res ≈ res4

	m5 = @formula Sales ~ (Price ~ NDI + Pimin) +  Pimin2
	result5 = reg(df, m5)
	yhat5 = predict(result5, df)
	res5 = residuals(result5, df)
	@test yhat ≈ yhat5
	@test res ≈ res5
end


@testset "saved fixed effects" begin

	df = DataFrame(CSV.File(joinpath(dirname(pathof(FixedEffectModels)), "../dataset/Cigar.csv")))
	df.StateC = categorical(df.State)

	# check save does not change r2
	model1 = @formula Sales ~ Price
	result1 = reg(df, model1, weights = :Pop)
	model2 = @formula Sales ~ Price
	result2 = reg(df, model2, weights = :Pop)
	@test r2(result1) ≈ r2(result2)



	model = @formula Sales ~ Price + fe(Year)
	result = reg(df, model, save = true)
	@test fe(result)[1, :fe_Year] ≈ 164.77833189721005
	@test size(fe(result), 2) == 1
	@test size(fe(result, keepkeys = true), 2) == 2

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
end


@testset "gpu" begin
	df = DataFrame(CSV.File(joinpath(dirname(pathof(FixedEffectModels)), "../dataset/Cigar.csv")))
	methods_vec = [:cpu]
	if CUDA.functional()
		push!(methods_vec, :CUDA)
	end
	#if Metal.functional()
	#	push!(methods_vec, :Metal)
	#end
	for method in methods_vec
		local model = @formula Sales ~ Price + fe(Year)
		local result = reg(df, model, save = true, method = method, double_precision = false)
		@test fe(result)[1, :fe_Year] ≈ 164.7 atol = 1e-1
	end
end


