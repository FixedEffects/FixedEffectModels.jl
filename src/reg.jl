function reg(f::Formula, df::AbstractDataFrame, vcov_method::AbstractVcov = VcovSimple(); weight::Union(Symbol, Nothing) = nothing)


	# decompose formula into endogeneous form model, reduced form model, absorb model
	rf = deepcopy(f)
	(rf, has_absorb, absorb_vars, absorbt) = decompose_absorb!(rf)
	# rf is now y ~ exogeneousvars + (endoegeneousvars = instruments)
	# absorbt is Terms(nothing ~ absorbvars)
	(rf, has_iv, iv_vars, ivt) = decompose_iv!(rf)
	# rf is now y ~ exogeneousvars + endoegeneousvars
	# ivt is Terms(y ~ exogeneousvars + instruments)
	rt = Terms(rf)
	
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
	all_vars = setdiff(vcat(vars, absorb_vars, vcov_vars, iv_vars, weight), [nothing])
	all_vars = unique(convert(Vector{Symbol}, all_vars))
	esample = complete_cases(df[all_vars])
	if weight != nothing
		esample &= convert(BitArray{1}, df[weight] .> zero(eltype(df[weight])))
	end
	df = df[esample, all_vars]
	all_except_absorb_vars = unique(convert(Vector{Symbol}, setdiff(vcat(vars, vcov_vars, iv_vars), [nothing])))
	for v in all_except_absorb_vars
		dropUnusedLevels!(df[v])
	end

	# create weight vector
	if weight == nothing
		w = fill(one(Float64), size(df, 1))
		sqrtw = w
	else
		w = convert(Vector{Float64}, df[weight])
		sqrtw = sqrt(w)
	end

	# Compute factors, an array of AbtractFixedEffects
	if has_absorb
		factors = construct_fe(df, absorbt.terms, sqrtw)
	end

	# Compute demeaned X
	df1 = DataFrame(map(x -> df[x], rt.eterms))
	names!(df1, convert(Vector{Symbol}, map(string, rt.eterms)))
	mf = ModelFrame(df1, rt, esample)
	coef_names = coefnames(mf)
	mm = ModelMatrix(mf)
	X = mm.m
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
		df1 = DataFrame(map(x -> df[x], ivt.eterms))
		names!(df1, convert(Vector{Symbol}, map(string, ivt.eterms)))
		mf = ModelFrame(df1, ivt, esample)
		mm = ModelMatrix(mf)
		Z = mm.m
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
	vcovmodel = VcovDataHat(Xhat, H, residuals, nobs, df_residual)
	matrix_vcov = vcov(vcovmodel, vcov_method, df)

	# Output object
	RegressionResult(coef, matrix_vcov, esample,  coef_names, rt.eterms[1], f, nobs, df_residual, r2, r2_a, F)
end





