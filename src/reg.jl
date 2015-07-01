function reg(f::Formula, df::AbstractDataFrame, vcov_method::AbstractVcovMethod = VcovSimple(); weight::Union(Symbol, Nothing) = nothing, subset::Union(AbstractVector{Bool}, Nothing) = nothing )

	# decompose formula into endogeneous form model, reduced form model, absorb model
	rf = deepcopy(f)
	(rf, has_absorb, absorb_formula) = decompose_absorb!(rf)
	if has_absorb
		absorb_vars = allvars(absorb_formula)
		absorb_terms = Terms(absorb_formula)
	else
		absorb_vars = Symbol[]
	end

	(rf, has_instrument, instrument_formula, endo_formula) = decompose_iv!(rf)
	if has_instrument
		instrument_vars = allvars(instrument_formula)
		instrument_terms = Terms(instrument_formula)
		instrument_terms.intercept = false
		endo_vars = allvars(endo_formula)
		endo_terms = Terms(endo_formula)
		endo_terms.intercept = false
	else
		instrument_vars = Symbol[]
		endo_vars = Symbol[]
	end

	rt = Terms(rf)
	# rt is Terms(y ~ exogeneousvars + endoegeneousvars)
	# endo_terms is Terms(nothing ~ endovars)
	# instrument_terms is Terms(nothing ~ instruments)
	# absorb_terms is Terms(nothing ~ absorbvars)

	# remove intercept if high dimensional categorical variables
	if has_absorb
		rt.intercept = false
	end

	# create a dataframe without missing values & negative weights
	vars = allvars(rf)
	vcov_vars = allvars(vcov_method)
	all_vars = vcat(vars, vcov_vars, absorb_vars, endo_vars, instrument_vars)
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
	# new dataframe (not subdataframe because issue with sub pooled data array)
	subdf = df[esample, all_vars]
	(size(subdf, 1) > 0) || error("sample is empty")
	#subdf = df[esample, all_vars]
	main_vars = unique(convert(Vector{Symbol}, vcat(vars, endo_vars, instrument_vars)))
	for v in main_vars
		# in case subdataframe, don't construct subdf[v] if you dont need to do it
		if typeof(df[v]) <: PooledDataArray
			dropUnusedLevels!(subdf[v])
		end
	end

	# Compute weight vector
	if weight == nothing
		w = fill(one(Float64), size(subdf, 1))
		sqrtw = w
	else
		w = convert(Vector{Float64}, subdf[weight])
		sqrtw = sqrt(w)
	end

	# Compute factors, an array of AbtractFixedEffects
	if has_absorb
		factors = construct_fe(subdf, absorb_terms.terms, sqrtw)
	end

	# Compute data for std errors
	vcov_method_data = VcovMethodData(vcov_method, subdf)

	# Compute X
	mf = simpleModelFrame(subdf, rt, esample)
	coef_names = coefnames(mf)
	Xexo = ModelMatrix(mf).m
	if weight != nothing
		broadcast!(*, Xexo, sqrtw, Xexo)
	end
	if has_absorb
		for j in 1:size(Xexo, 2)
			Xexo[:,j] = demean_vector!(Xexo[:,j], factors)
		end
	end

	# Compute y
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

	# Compute Xendo and Z
	if has_instrument
		mf = simpleModelFrame(subdf, endo_terms, esample)
		coef_names = vcat(coef_names, coefnames(mf))
		Xendo = ModelMatrix(mf).m
		if weight != nothing
			broadcast!(*, Xendo, sqrtw, Xendo)
		end
		if has_absorb
			for j in 1:size(Xendo, 2)
				Xendo[:,j] = demean_vector!(Xendo[:,j], factors)
			end
		end
		

		mf = simpleModelFrame(subdf, instrument_terms, esample)
		Z = ModelMatrix(mf).m
		size(Z, 2) >= size(Xendo, 2) || error("Model not identified. There must be at least as many instruments as endogeneneous variables")
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
	if has_instrument
		newZ = hcat(Xexo, Z)
		crossz = cholfact!(At_mul_B(newZ, newZ))
		Pi = crossz \ (At_mul_B(newZ, Xendo))
		# Can't update X -> Xhat in place because needed to compute residuals
		Xhat = hcat(Xexo, newZ * Pi)
	else
		Xhat = Xexo
		Xall = Xexo
	end
	# Compute coef and residuals
	crossx = cholfact!(At_mul_B(Xhat, Xhat))
	coef = crossx \ At_mul_B(Xhat, y)

	if has_instrument
		X = hcat(Xexo, Xendo)
	else
		X = Xexo
	end
	residuals = y - X * coef

	# Compute degrees of freedom
	df_intercept = 0
	if has_absorb | rt.intercept
		df_intercept = 1
	end
	df_absorb = 0
	if has_absorb 
		## poor man adjustement of df for clustedered errors + fe: only if fe name != cluster name
		for fe in factors
			df_absorb += (typeof(vcov_method) == VcovCluster && in(fe.name, vcov_vars)) ? 0 : sum(fe.scale .!= zero(Float64))
		end
	end
	nobs = size(X, 1)
	df_residual = size(X, 1) - size(X, 2) - df_absorb 

	# Compute ess, tss, r2, r2 adjusted
	if weight == nothing
		(ess, tss) = compute_ss(residuals, y, rt.intercept)
	else
		(ess, tss) = compute_ss(residuals, y, rt.intercept, w, sqrtw)
	end
	r2 = 1 - ess / tss 
	r2_a = 1 - ess / tss * (nobs - rt.intercept) / df_residual 
	
	# Compute standard error
	vcov_data = VcovData{1}(inv(crossx), Xhat, residuals, df_residual)
	matrix_vcov = vcov!(vcov_method_data, vcov_data)

	# Compute Fstat
	coefF = coef
	matrix_vcovF = matrix_vcov
	if rt.intercept
 		coefF = coefF[2:end]
		matrix_vcovF = matrix_vcovF[2:end, 2:end]
	end
	F = diagm(coefF)' * inv(matrix_vcovF) * diagm(coefF)
	F = F[1] 
	if typeof(vcov_method) == VcovCluster 
		nclust = minimum(values(vcov_method_data.size))
		p = ccdf(FDist(size(X, 1) - df_intercept, nclust - 1), F)
	else
		p = ccdf(FDist(size(X, 1) - df_intercept, df_residual - df_intercept), F)
	end


	# Compute Fstat first stage based on Kleibergen-Paap
	if has_instrument
		# residualize Xendo and Z in place
		Xendo_res = BLAS.gemm!('N', 'N', -1.0, newZ, Pi, 1.0, Xendo)
		Pi2 = cholfact!(At_mul_B(Xexo, Xexo)) \ At_mul_B(Xexo, Z)
		Z_res = BLAS.gemm!('N', 'N', -1.0, Xexo, Pi2, 1.0, Z)
		Pi_res = Pi[(size(Xexo, 2) + 1):end, :]
		(F_kp, p_kp) = rank_test!(Xendo_res, Z_res, Pi_res, vcov_method_data, size(Xhat, 2), df_absorb)
		return(RegressionResultIV(coef, matrix_vcov, esample,  coef_names, rt.eterms[1], f, nobs, df_residual, r2, r2_a, F,p, F_kp, p_kp))
	else
		return(RegressionResult(coef, matrix_vcov, esample,  coef_names, rt.eterms[1], f, nobs, df_residual, r2, r2_a, F, p))
	end


end





