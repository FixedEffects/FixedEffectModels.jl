using DataFrames, StatsBase




function reg(f::Formula, df::AbstractDataFrame, vce::AbstractVce = VceSimple())
	
	# get all variables
	t = DataFrames.Terms(f)
	if (typeof(t.terms[1]) == Expr) && t.terms[1].args[1] == :|
		hasfe = true
		absorbexpr =  t.terms[1].args[3]
		absorbf = Formula(nothing, absorbexpr)
		absorbvars = unique(DataFrames.allvars(absorbexpr))

		rexpr =  t.terms[1].args[2]
		rf = Formula(f.lhs, rexpr)
		rvars = unique(DataFrames.allvars(rf))
	else
		hasfe = false
		rf = f
		rvars =  unique(DataFrames.allvars(f))
		absorbvars = nothing
	end
	rt = DataFrames.Terms(rf)

	vcevars = DataFrames.allvars(vce)
	allvars = setdiff(vcat(vcevars, absorbvars, rvars), [nothing])
	allvars = unique(convert(Vector{Symbol}, allvars))

	# construct df without NA for all variables
	esample = complete_cases(df[allvars])
	df = df[esample, allvars]
	for v in allvars
		dropUnusedLevels!(df[v])
	end


	# pre demean if fe
	if  hasfe
		# construct an array of factors
		factors = AbstractFe[]
		for a in DataFrames.Terms(absorbf).terms
			push!(factors, construct_fe(df, a))
		end

		# in case where only interacted fixed effect,  add constant
		if all(map(z -> typeof(z) <: FeInteracted, factors))
			push!(factors, Fe(PooledDataArray(fill(1, size(df, 1)))))
		end

		# demean each vector sequentially
		for x in rvars
			df[x] =  demean_vector(df, factors, df[x])
		end

		# Removing intercept 
		rt = deepcopy(rt)
		rt.intercept = false
	end


	mf = ModelFrame(rt, df)
	mm = ModelMatrix(mf)
	coefnames = DataFrames.coefnames(mf)

	y = model_response(mf)
	X = mm.m
	H = At_mul_B(X, X)
	H = inv(cholfact!(H))
	coef = H * (X' * y)
	residuals  = y - X * coef

    # standard error
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
    vcovmodel = VceModelHat(X, H, residuals,  nobs, df_residual)
	vcov = StatsBase.vcov(vcovmodel, vce, df)


    # Output object
    RegressionResult(coef, vcov, coefnames, rt.eterms[1], nobs, df_residual, esample, t)
end








