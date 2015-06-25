
function partial_out(f::Formula, df::AbstractDataFrame; weight::Union(Symbol, Nothing) = nothing)

	rf = deepcopy(f)

	# decompose formula into normal  vs absorbpart
	(rf, has_absorb, absorb_vars, absorbt) = decompose_absorb!(rf)

	# create a dataframe without missing values & negative weights
	vars = unique(allvars(rf))
	all_vars = setdiff(vcat(vars, absorb_vars, weight), [nothing])
	all_vars = unique(convert(Vector{Symbol}, all_vars))
	esample = complete_cases(df[all_vars])
	if weight != nothing
		esample &= convert(BitArray{1}, df[weight] .> zero(eltype(df[weight])))
	end
	df = df[esample, all_vars]
	all_except_absorb_vars = unique(convert(Vector{Symbol}, setdiff(vcat(vars), [nothing])))
	for v in all_except_absorb_vars
		dropUnusedLevels!(df[v])
	end

	# Compute weight vector
	if weight == nothing
		w = fill(one(Float64), size(df, 1))
		sqrtw = w
	else
		w = convert(Vector{Float64}, df[weight])
		sqrtw = sqrt(w)
	end

	# Build factors, an array of AbtractFixedEffects
	if has_absorb
		factors = construct_fe(df, absorbt.terms, sqrtw)
	end


	# Compute demeaned Y
	yf = Formula(nothing, rf.lhs)
	yt = Terms(yf)
	yt.intercept = false
	mfY = simpleModelFrame(df, yt, esample)
	Y = ModelMatrix(mfY).m
	if weight != nothing
		broadcast!(*, Y, sqrtw, Y)
	end
	if has_absorb
		for j in 1:size(Y, 2)
			Y[:,j] = demean_vector!(Y[:,j], factors)
		end	
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
			mf = simpleModelFrame(df, xt, esample)
			X = ModelMatrix(mf).m
		else
			X = fill(one(Float64), (size(df, 1), 1))
		end 	
		if weight != nothing
			broadcast!(*, X, sqrtw, X)
		end
		if has_absorb
			for j in 1:size(X, 2)
				X[:,j] = demean_vector!(X[:,j], factors)
			end
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


	# Return a dataframe
	out = mfY.df
	for j in 1:size(Y, 2)
		out[:, j] = DataArray(Float64, size(out, 1))
		out[esample, j] = residuals[:, j]
	end

	return(out)
end





