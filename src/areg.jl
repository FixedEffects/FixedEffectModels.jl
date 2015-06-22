using DataFrames, StatsBase


# define neutral vector
immutable Ones <: AbstractVector{Float64}
    dims::Int
end
Base.size(O::Ones) = (O.dims,)
Base.getindex(O::Ones, I::Int...) = 1.0


function reg(f::Formula, df::AbstractDataFrame, vce::AbstractVce = VceSimple(); absorb::Vector = [nothing], weight::Union(Symbol, Nothing) = nothing)
	# get all variables
	rt = DataFrames.Terms(f)
	vars = unique(DataFrames.allvars(f))
	vcevars = DataFrames.allvars(vce)

	hasfe = absorb != [nothing]
	if hasfe
		absorbvars = Symbol[]
		for a in absorb
			append!(absorbvars, DataFrames.allvars(a))
		end
		absobvars = unique(absorbvars)
	else
		absorbvars = nothing
	end
	allvars = setdiff(vcat(vars, absorbvars, vcevars, weight), [nothing])
	allvars = unique(convert(Vector{Symbol}, allvars))
	# construct df without NA for all variables
	esample = complete_cases(df[allvars])
	df = df[esample, allvars]
	for v in allvars
		dropUnusedLevels!(df[v])
	end

	# create weight vector
	if weight == nothing
		w = fill(one(Float64), size(df, 1))
		sqrtw = w
	else
		w = df[weight]
		w = convert(Vector{Float64}, w)
		sqrtw = sqrt(w)
	end


	# If high dimensional fixed effects, demean all variables
	if hasfe
		# construct an array of factors
		factors = AbstractFe[]
		for a in absorb
			push!(factors, construct_fe(df, a, sqrtw))
		end
		# in case where only interacted fixed effect, add constant
		if all(map(z -> typeof(z) <: FeInteracted, factors))
			push!(factors, Fe(PooledDataArray(fill(1, size(df, 1))), sqrtw, :cons))
		end

		# demean each vector sequentially
		for x in vars
			if weight == nothing
				df[x] = demean_vector(factors, df[x])
			else
				dfx = df[x]
				for i in 1:length(dfx)
					@inbounds dfx[i] *= sqrtw[i]
				end
				df[x] = demean_vector(factors, dfx) 
				for i in 1:length(dfx)
					@inbounds dfx[i] /= sqrtw[i]
				end
			end
		end
		# Removing intercept 
		rt.intercept = false
	end
	# Estimate usual regression model
	df1 = DataFrame(map(x -> df[x], rt.eterms))
	names!(df1, convert(Vector{Symbol}, map(string, rt.eterms)))
	mf = ModelFrame(df1, rt, esample)
	mm = ModelMatrix(mf)
	coefnames = DataFrames.coefnames(mf)

	y = model_response(mf)::Vector{Float64}
	X = mm.m::Matrix{Float64}
	if weight != nothing
		X = broadcast!(*, X, X, sqrt(w))
		for i in 1:length(y)
			@inbounds y[i] *= sqrtw[i] 
		end
	end
	H = At_mul_B(X, X)
	H = inv(cholfact!(H))
	coef = H * (X' * y)
	residuals = (y - X * coef)


	# compute degree of freedom
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

	# compute ess, tss, r2, r2 adjusted, F
	if weight == nothing
		(ess, tss) = compute_ss(residuals, y, rt.intercept)
	else
		(ess, tss) = compute_ss(residuals, y, rt.intercept, w, sqrtw)
	end
	r2 = 1 - ess / tss 
	r2_a = 1 - ess / tss * (nobs - rt.intercept) / df_residual 
	F = (tss - ess) / ((nobs - df_residual - rt.intercept) * ess/df_residual)


	# standard error
	vcovmodel = VceDataHat(X, H, residuals, nobs, df_residual)
	vcov = StatsBase.vcov(vcovmodel, vce, df)
	

	# Output object
	RegressionResult(coef, vcov, r2, r2_a, F, nobs, df_residual, coefnames, rt.eterms[1], rt, esample)
end





