using DataFrames, StatsBase




function reg(f::Formula, df::AbstractDataFrame, vce::AbstractVce = VceSimple(); weight::Union(Symbol, Nothing) = nothing)
	
	# get all variables
	t = DataFrames.Terms(f)
	hasfe = (typeof(t.terms[1]) == Expr) && t.terms[1].args[1] == :|
	if hasfe
		absorbexpr =  t.terms[1].args[3]
		absorbf = Formula(nothing, absorbexpr)
		absorbvars = unique(DataFrames.allvars(absorbexpr))

		rexpr =  t.terms[1].args[2]
		rf = Formula(f.lhs, rexpr)
		rvars = unique(DataFrames.allvars(rf))
	else
		rf = f
		rvars =  unique(DataFrames.allvars(f))
		absorbvars = nothing
	end
	rt = DataFrames.Terms(rf)

	vcevars = DataFrames.allvars(vce)
	allvars = setdiff(vcat(rvars, absorbvars, vcevars, weight), [nothing])
	allvars = unique(convert(Vector{Symbol}, allvars))
	print(allvars)
	# construct df without NA for all variables
	esample = complete_cases(df[allvars])
	df = df[esample, allvars]
	for v in allvars
		dropUnusedLevels!(df[v])
	end

	if weight == nothing
		w = fill(one(Float64), size(df, 1))
		sqrtw = w
	else
		w = df[weight]
		w = convert(Vector{Float64}, w)
		sqrtw = sqrt(w)
	end

	# pre demean if fe
	if  hasfe
		# construct an array of factors
		factors = AbstractFe[]
		for a in DataFrames.Terms(absorbf).terms
			push!(factors, construct_fe(df, a, sqrtw))
		end

		# in case where only interacted fixed effect,  add constant
		if all(map(z -> typeof(z) <: FeInteracted, factors))
			push!(factors, Fe(PooledDataArray(fill(1, size(df, 1))), w, :cons))
		end

		# demean each vector sequentially
		for x in rvars
			if weight == nothing
				df[x] =  demean_vector(df, factors, df[x])
			else
				dfx = df[x]
				for i in 1:length(dfx)
					@inbounds dfx[i] *= sqrtw[i]
				end
				df[x] =  demean_vector(df, factors, dfx) 
				for i in 1:length(dfx)
					@inbounds dfx[i] /= sqrtw[i]
				end
			end
		end
		# Removing intercept 
		rt = deepcopy(rt)
		rt.intercept = false
	end
	df1 = DataFrame(map(x -> df[x], rt.eterms))
	names!(df1, convert(Vector{Symbol}, map(string, rt.eterms)))
	mf = ModelFrame(df1, rt, esample)
	mm = ModelMatrix(mf)
	coefnames = DataFrames.coefnames(mf)

	y = model_response(mf)
	X = mm.m
	if weight != nothing
		X = broadcast!(*, X, X, sqrt(w))
		for i in 1:length(y)
			@inbounds y[i] *= sqrtw[i] 
		end
	end
	H = At_mul_B(X, X)
	H = inv(cholfact!(H))
	coef = H * (X' * y)
	residuals  = (y - X * coef) 

    # df
    df_fe = 0
    if hasfe 
    	for f in factors
    		df_fe += (typeof(vce) == VceCluster && in(f.name, vcevars)) ? 0 : length(f.size)
    	end
    end
    if hasfe && typeof(vce) == VceCluster
    	for f in factors
    		df_fe += in(f.name, vcevars) ? 0 : length(f.size)
    	end
    end
    nobs = size(X, 1)
    df_residual = size(X, 1) - size(X, 2) - df_fe 
    

    #ess, tss
	ess = zero(Float64)
	for i in 1:length(residuals)
    	ess += residuals[i]^2
	end
	tss = zero(Float64)
    if rt.intercept
    	m = mean(y)
    	if weight == nothing
	    	for i in 1:length(residuals)
		    	@inbounds tss += (y[i] - m)^2
			end
	    else
	    	m = m / sum(sqrtw) * nobs
	    	for i in 1:length(residuals)
		    	@inbounds tss += (y[i] - sqrtw[i] * m)^2
			end
			tss = tss
		end
    else
    	for i in 1:length(residuals)
	    	@inbounds tss += y[i]^2
		end
    end


    # r2, r2 adjusted, F
    r2 = 1 - ess / tss 
    r2_a = 1 - ess / tss * (nobs - rt.intercept) / df_residual 
    F = (tss - ess) / ((nobs - df_residual - rt.intercept) * ess/df_residual)


    # standard error
    vcovmodel = VceModelHat(X, H, residuals,  nobs, df_residual)
	vcov = StatsBase.vcov(vcovmodel, vce, df)
	
    # Output object
    RegressionResult(coef, vcov, r2, r2_a, F,  coefnames, rt.eterms[1], nobs, df_residual, esample, t)
end








