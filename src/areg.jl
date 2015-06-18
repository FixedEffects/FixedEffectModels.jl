using DataFrames, GLM




function areg(f::Formula, df::AbstractDataFrame, absorb::Formula, cluster::Formula = nothing)
	
	# get all variables present in f or absorb (except nothing)
	clustervars = setdiff(unique(DataFrames.allvars(cluster)), [:nothing])
	absorbvars = setdiff(unique(DataFrames.allvars(absorb)), [:nothing])
	regvars = setdiff(unique(DataFrames.allvars(f)), [:nothing])
	allvars = unique([clustervars, absorbvars, regvars])
	esample = complete_cases(df[allvars])

	df = df[esample, allvars]
	for v in allvars
		dropUnusedLevels!(df[v])
	end

	# construct an array of factors
	factors = AbstractFe[]
	for a in DataFrames.Terms(absorb).terms
		push!(factors, construct_fe(df, a))
	end

	# in case where only interacted fixed effect,  add constant
	if all(map(z -> typeof(z) <: FeInteracted, factors))
		push!(factors, Fe(PooledDataArray(fill(1, size(df, 1)))))
	end
	
	# demean each vector sequentially
	for x in DataFrames.allvars(f)
		df[x] =  demean_vector(df, factors, df[x])
	end

	# Estimate linear model after removing intercept 
	terms = DataFrames.Terms(f)
	terms_rm = deepcopy(terms)
	terms_rm.intercept = false

	mf = ModelFrame(terms_rm, df)
	mm = ModelMatrix(mf)

	coefnames = DataFrames.coefnames(mf)

	y = model_response(mf)
	X = mm.m
	H = At_mul_B(X, X)
	H = inv(cholfact!(H))
	println(mean(X))
	beta = H * (X' * y)
	residuals  = y - X * beta

    # standard error
    df_fe = 0
    for f in factors
    	df_fe += in(f.name, clustervars) ? 0 : length(f.size)
    end
    nobs = size(X, 1)
    df_residual = size(X, 1) - size(X, 2) - df_fe
    vcovmodel = VcovModelH(X, H, residuals, df_residual, nobs)
    if isequal(cluster, nothing)
    	vcov = vcov(vcovmodel)
    else
	    vcov = vcov_cluster(vcovmodel, df[[clustervars]])
	end

    # Output object
    RegressionResult(beta, vcov, coefnames, terms.eterms[1], nobs, df_residual, esample, terms_rm)
end








