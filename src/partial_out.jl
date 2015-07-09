
function partial_out(f::Formula, df::AbstractDataFrame; add_mean = false, weight::Union(Symbol, Nothing) = nothing,  maxiter::Integer = 10000, tol::FloatingPoint = 1e-8)

	rf = deepcopy(f)

	# decompose formula into normal  vs absorbpart
	(rf, has_absorb, absorb_formula) = decompose_absorb!(rf)
	if has_absorb
		absorb_vars = allvars(absorb_formula)
		absorb_terms = Terms(absorb_formula)
	else
		absorb_vars = Symbol[]
	end
	# create a dataframe without missing values & negative weights
	vars = allvars(rf)
	all_vars = vcat(vars, absorb_vars)
	all_vars = unique(convert(Vector{Symbol}, all_vars))
	esample = complete_cases(df[all_vars])
	if weight != nothing
		esample &= isnaorneg(df[weight])
		all_vars = unique(vcat(all_vars, weight))
	end
	subdf = df[esample, all_vars]
	all_except_absorb_vars = unique(convert(Vector{Symbol}, vars))
	for v in all_except_absorb_vars
		dropUnusedLevels!(subdf[v])
	end

	# Compute weight vector
	if weight == nothing
		w = fill(one(Float64), size(subdf, 1))
		sqrtw = w
	else
		w = convert(Vector{Float64}, subdf[weight])
		sqrtw = sqrt(w)
	end
	# Build factors, an array of AbtractFixedEffects
	if has_absorb
		factors = construct_fe(subdf, absorb_terms.terms, sqrtw)
	end


	# Compute demeaned Y
	yf = Formula(nothing, rf.lhs)
	yt = Terms(yf)
	yt.intercept = false
	mfY = simpleModelFrame(subdf, yt, esample)
	Y = ModelMatrix(mfY).m
	if weight != nothing
		scale!(sqrtw, Y)
	end
	if add_mean
		m = mean(Y, 1)
	end
	if has_absorb
		(Y, iterations, converged)= demean!(Y, factors; maxiter = maxiter, tol = tol)
	end


	# Compute demeaned X
	xf = Formula(nothing, rf.rhs)
	xt = Terms(xf)
	if has_absorb
		xt.intercept = false
	end
	xvars = allvars(xf)
	if length(xvars) > 0 || xt.intercept
		if length(xvars) > 0 
			mf = simpleModelFrame(subdf, xt, esample)
			X = ModelMatrix(mf).m
		else
			X = fill(one(Float64), (size(subdf, 1), 1))
		end 	
		if weight != nothing
			scale!(sqrtw, X)
		end
		if has_absorb
			(X, iterations, converged)= demean!(X, factors; maxiter = maxiter, tol = tol)
		end
	end
	
	# Compute residuals
	if length(xvars) > 0 || xt.intercept
		H = At_mul_B(X, X)
		H = inv(cholfact!(H))
		coef = H * (At_mul_B(X, Y))
		residuals = Y - X * coef
	else
		residuals = Y
	end

	if weight != nothing
		broadcast!(/, residuals,  residuals, sqrtw)
	end
	if add_mean
		broadcast!(+, residuals, m, residuals)
	end

	# Return a dataframe
	yvars = convert(Vector{Symbol}, map(string, yt.eterms))
	out = DataFrame()
	j = 0
	for y in yvars
		j += 1
		out[y] = DataArray(Float64, size(df, 1))
		out[esample, y] = residuals[:, j]
	end
	return(out)
end





