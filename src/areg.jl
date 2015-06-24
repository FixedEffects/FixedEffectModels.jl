using DataFrames, StatsBase



# define neutral vector
immutable Ones <: AbstractVector{Float64}
    dims::Int
end
Base.size(O::Ones) = (O.dims,)
Base.getindex(O::Ones, I::Int...) = 1.0



function reg(f::Formula, df::AbstractDataFrame, vcov::AbstractVcov = VcovSimple(); weight::Union(Symbol, Nothing) = nothing)

	rf = deepcopy(f)

	# decompose formula into normal + iv vs absorbpart
	has_absorb = false
	absorbvars = nothing
	if typeof(rf.rhs) == Expr && rf.rhs.args[1] == :(|>)
		has_absorb = true
		absorbf = Formula(nothing, rf.rhs.args[3])
		absorbvars = unique(DataFrames.allvars(absorbf))
		absorbt = DataFrames.Terms(absorbf)
		rf.rhs = rf.rhs.args[2]
	end
	

	# decompose into normal vs iv part
	has_iv = false
	ivvars = nothing
	if typeof(rf.rhs) == Expr
		if rf.rhs.head == :(=)
			has_iv = true
			ivvars = unique(DataFrames.allvars(rf.rhs.args[2]))
			ivf = deepcopy(rf)
			ivf.rhs = rf.rhs.args[2]
			ivt = DataFrames.Terms(ivf)
			rf.rhs = rf.rhs.args[1]
		else
			for i in 1:length(rf.rhs.args)
				if typeof(rf.rhs.args[i]) == Expr && rf.rhs.args[i].head == :(=)
					has_iv = true
					ivvars = unique(DataFrames.allvars(rf.rhs.args[i].args[2]))
					ivf = deepcopy(rf)
					ivf.rhs.args[i] = rf.rhs.args[i].args[2]
					ivt = DataFrames.Terms(ivf)
					rf.rhs.args[i] = rf.rhs.args[i].args[1]
				end
			end
		end
	end

	rt = DataFrames.Terms(rf)
	vars = unique(DataFrames.allvars(rf))
	
	if has_absorb
		rt.intercept = false
		if has_iv
			ivt.intercept = false
		end
	end

	# get variables used for vcov
	vcovvars = DataFrames.allvars(vcov)


	allvars = setdiff(vcat(vars, absorbvars, vcovvars, ivvars, weight), [nothing])
	allvars = unique(convert(Vector{Symbol}, allvars))
	# construct df without NA for all variables
	esample = complete_cases(df[allvars])
	# also remove elements with zero or negative weights
	if weight != nothing
		esample &= convert(BitArray{1}, df[weight] .> zero(eltype(df[weight])))
	end
	df = df[esample, allvars]

	# create weight vector
	if weight == nothing
		w = fill(one(Float64), size(df, 1))
		sqrtw = w
	else
		w = convert(Vector{Float64}, df[weight])
		sqrtw = sqrt(w)
	end

	# Similar to DataFrames.ModelFrame function
	# only remove absent levels for factors not in absorbvars
	all_except_absorbvars = unique(convert(Vector{Symbol}, setdiff(vcat(vars, vcovvars, ivvars), [nothing])))
	for v in all_except_absorbvars
		dropUnusedLevels!(df[v])
	end
	df1 = DataFrame(map(x -> df[x], rt.eterms))
	names!(df1, convert(Vector{Symbol}, map(string, rt.eterms)))
	mf = ModelFrame(df1, rt, esample)
	coefnames = DataFrames.coefnames(mf)



	# If high dimensional fixed effects, demean all variables
	if has_absorb
		# construct an array of factors
		factors = AbstractFe[]
		for a in absorbt.terms
			push!(factors, construct_fe(df, a, sqrtw))
		end
		# in case where only interacted fixed effect, add constant
		if all(map(z -> typeof(z) <: FeInteracted, factors))
			push!(factors, Fe(PooledDataArray(fill(1, size(df, 1))), sqrtw, :cons))
		end
		# demean y
		for x in setdiff(vcat(vars, ivvars), [nothing])
			xv = convert(Vector{Float64}, df[x])
			multiplication_elementwise!(xv, sqrtw)
			df[x] = demean_vector!(xv, factors)
		end
	end

	# build X
	mm = ModelMatrix(mf)
	X = mm.m

	# build y
	py = model_response(mf)[:]
	if eltype(py) != Float64
		y = convert(py, Float64)
	else
		y = py
	end

	# build Z
	if has_iv
		ivt = DataFrames.Terms(ivf)
		df1 = DataFrame(map(x -> df[x], ivt.eterms))
		names!(df1, convert(Vector{Symbol}, map(string, ivt.eterms)))
		mf = ModelFrame(df1, ivt, esample)
		mm = ModelMatrix(mf)
		Z = mm.m
	end

	if !has_absorb
		multiplication_elementwise!(y, sqrtw)
		broadcast!(*, X, sqrtw, X)
		if has_iv
			broadcast!(*, Z, sqrtw, Z)
		end
	end



	if has_iv
		Hz = At_mul_B(Z, Z)
		Hz = inv(cholfact!(Hz))
		Xhat = A_mul_Bt(Z * Hz, Z) * X
	else
		Xhat = X
	end
	H = At_mul_B(Xhat, Xhat)
	H = inv(cholfact!(H))
	coef = H * (At_mul_B(Xhat, y))
	residuals = y - X * coef

	# compute degree of freedom
	df_absorb = 0
	if has_absorb 
		## poor man adjustement when clustering: if fe name == cluster name
		for fe in factors
			df_absorb += (typeof(vcov) == VcovCluster && in(fe.name, vcovvars)) ? 0 : sum(fe.scale .> 0)
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

	# standard error
	vcovmodel = VcovDataHat(Xhat, H, residuals, nobs, df_residual)
	vcov = StatsBase.vcov(vcovmodel, vcov, df)	

	# Output object
	RegressionResult(coef, vcov, r2, r2_a, F, nobs, df_residual, coefnames, rt.eterms[1], f, esample)
end





