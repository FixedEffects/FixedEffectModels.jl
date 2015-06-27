function reg(f::Formula, df::AbstractDataFrame, vcov_method::AbstractVcov = VcovSimple(); weight::Union(Symbol, Nothing) = nothing, subset::Union(AbstractVector{Bool}, Nothing) = nothing )

	# decompose formula into endogeneous form model, reduced form model, absorb model
	rf = deepcopy(f)
	(rf, has_absorb, absorb_vars, absorbt) = decompose_absorb!(rf)
	(rf, has_iv, iv_vars, ivt) = decompose_iv!(rf)
	rt = Terms(rf)
	# rt is Terms(y ~ exogeneousvars + endoegeneousvars)
	# ivt is Terms(y ~ exogeneousvars + instruments)
	# absorbt is Terms(nothing ~ absorbvars)
	
	# remove intercept if high dimensional categorical variables
	if has_absorb
		rt.intercept = false
		if has_iv
			ivt.intercept = false
		end
	end

	# create a dataframe without missing values & negative weights
	vars = unique(allvars(rf))
	vcov_vars = allvars(vcov_method)
	all_vars = setdiff(vcat(vars, absorb_vars, vcov_vars, iv_vars), [nothing])
	all_vars = unique(convert(Vector{Symbol}, all_vars))
	esample = complete_cases(df[all_vars])
	if weight != nothing
		esample &= isnaorneg(df[weight])
		all_vars = unique(vcat(all_vars, weight))
	end
	if subset != nothing
		length(subset) == size(df, 1) || error("the number of rows in df is $(size(df, 1)) but the length of subset is $(size(df, 2))")
		esample &= convert(BitArray, subset)
	end
	# I take a new dataframe because I dont know how to remove pooled data array otherwise from a subdataframe
	subdf = df[esample, all_vars]
	(size(subdf, 1) > 0) || error("sample is empty")
	#subdf = df[esample, all_vars]
	potential_vars = unique(convert(Vector{Symbol}, setdiff(vcat(vars, iv_vars), [nothing])))
	for v in potential_vars
		# in case subdataframe, don't construct subdf[v] if you dont need to do it
		if typeof(df[v]) <: PooledDataArray
			dropUnusedLevels!(subdf[v])
		end
	end

	# create weight vector
	if weight == nothing
		w = fill(one(Float64), size(subdf, 1))
		sqrtw = w
	else
		w = convert(Vector{Float64}, subdf[weight])
		sqrtw = sqrt(w)
	end

	# Compute factors, an array of AbtractFixedEffects
	if has_absorb
		factors = construct_fe(subdf, absorbt.terms, sqrtw)
	end

	# Compute demeaned X
	mf = simpleModelFrame(subdf, rt, esample)
	coef_names = coefnames(mf)
	X = ModelMatrix(mf).m
	if weight != nothing
		broadcast!(*, X, sqrtw, X)
	end
	if has_absorb
		for j in 1:size(X, 2)
			X[:,j] = demean_vector!(X[:,j], factors)
		end
	end

	# Compute demeaned y
	py = model_response(mf)[:]
	if eltype(py) != Float64
		y = convert(py, Float64)
	else
		y = py
	end
	if weight != nothing
		multiplication_elementwise!(y, sqrtw)
	end
	if has_absorb
		y = demean_vector!(y, factors)
	end

	# Compute demeaned Z
	if has_iv
		mf = simpleModelFrame(subdf, ivt, esample)
		Z = ModelMatrix(mf).m
		if weight != nothing
			broadcast!(*, Z, sqrtw, Z)
		end
		if has_absorb
			for j in 1:size(Z, 2)
				Z[:,j] = demean_vector!(Z[:,j], factors)
			end
		end
	end

	# Compute Xhat
	if has_iv
		Hz = At_mul_B(Z, Z)
		Hz = inv(cholfact!(Hz))
		Xhat = A_mul_Bt(Z * Hz, Z) * X
	else
		Xhat = X
	end

	# Compute coef and residuals
	H = At_mul_B(Xhat, Xhat)
	H = inv(cholfact!(H))
	coef = H * (At_mul_B(Xhat, y))
	residuals = y - X * coef

	# Compute degree of freedom
	df_absorb = 0
	if has_absorb 
		## poor man clustering adjustement of df: only if fe name == cluster name
		for fe in factors
			df_absorb += (typeof(vcov_method) == VcovCluster && in(fe.name, vcov_vars)) ? 0 : sum(fe.scale .> 0)
		end
		# still one in case all are removed
		#if df_absorb == 0
		#	df_absorb = 1
		#end
	end
	nobs = size(X, 1)
	df_residual = size(X, 1) - size(X, 2) - df_absorb 

	# compute ess, tss, r2, r2 adjusted, F
	if weight == nothing
		(ess, tss) = compute_ss(residuals, y, rt.intercept)
	else
		(ess, tss) = compute_ss(residuals, y, rt.intercept, w, sqrtw)
	end
	r2 = 1 - ess / tss 
	r2_a = 1 - ess / tss * (nobs - rt.intercept) / df_residual 
	F = (tss - ess) / ((nobs - df_residual - rt.intercept) * ess/df_residual)

	# compute standard error
	vcovmodel = VcovDataHat(Xhat, H, residuals, df_residual)
	matrix_vcov = vcov!(vcovmodel, vcov_method, subdf)

	# Return RegressionResult object
	RegressionResult(coef, matrix_vcov, esample,  coef_names, rt.eterms[1], f, nobs, df_residual, r2, r2_a, F)
end





