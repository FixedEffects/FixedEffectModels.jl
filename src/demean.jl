
##############################################################################
##
## Fe and FixedEffectSlope
##
##############################################################################


abstract AbstractFixedEffect

immutable type FixedEffectIntercept{R} <: AbstractFixedEffect
	refs::Vector{R}        # Refs corresponding to the refs field of the original PooledDataArray
	w::Vector{Float64}     # scale
	scale::Vector{Float64} # 1/(sum of scale) within each group
	name::Symbol           # Name of variable in the original dataframe
end

immutable type FixedEffectSlope{R} <: AbstractFixedEffect
	refs::Vector{R}        # Refs corresponding to the refs field of the original PooledDataArray
	w::Vector{Float64}     # weights
	scale::Vector{Float64} # 1/(sum of weights * x) for each group
	x::Vector{Float64}     # the continuous interaction 
	name::Symbol           # Name of factor variable in the original dataframe
	xname::Symbol          # Name of continuous variable in the original dataframe
end


function FixedEffectIntercept(f::PooledDataArray, w::Vector{Float64}, name::Symbol)
	scale = fill(zero(Float64), length(f.pool))
	refs = f.refs
	@inbounds @simd  for i in 1:length(refs)
		scale[refs[i]] += abs2(w[i])
	end
	@inbounds @simd  for i in 1:length(scale)
		scale[i] = scale[i] > 0 ? (one(Float64) / scale[i]) : zero(Float64)
	end
	FixedEffectIntercept(refs, w, scale, name)
end

function FixedEffectSlope(f::PooledDataArray, w::Vector{Float64}, x::Vector{Float64}, name::Symbol, xname::Symbol)
	scale = fill(zero(Float64), length(f.pool))
	refs = f.refs
	@inbounds @simd  for i in 1:length(refs)
		 scale[refs[i]] += abs2((x[i] * w[i]))
	end
	@inbounds @simd  for i in 1:length(scale)
		scale[i] = scale[i] > 0 ? (one(Float64) / scale[i]) : zero(Float64)
	end
	FixedEffectSlope(refs, w, scale, x, name, xname)
end


function construct_fe(df::AbstractDataFrame, a::Expr, w::Vector{Float64})
	if a.args[1] == :&
		if (typeof(df[a.args[2]]) <: PooledDataArray) & !(typeof(df[a.args[3]]) <: PooledDataArray)
			f = df[a.args[2]]
			x = convert(Vector{Float64}, df[a.args[3]])
			return(FixedEffectSlope(f, w, x, a.args[2], a.args[3]))
		elseif (typeof(df[a.args[3]]) <: PooledDataArray) & !(typeof(df[a.args[2]]) <: PooledDataArray)
			f = df[a.args[3]]
			x = convert(Vector{Float64}, df[a.args[2]])
			return(FixedEffectSlope(f, w, x, a.args[3], a.args[2]))
		else
			error("& is not of the form factor & nonfactor")
		end
	else
		error("Formula should be composed of & and symbols")
	end
end

function construct_fe(df::AbstractDataFrame, a::Symbol, w::Vector{Float64})
	if typeof(df[a]) <: PooledDataArray
		return(FixedEffectIntercept(df[a], w, a))
	else
		error("$(a) is not a pooled data array")
	end
end

function construct_fe(df::AbstractDataFrame, v::Vector, w::Vector{Float64})
	factors = AbstractFixedEffect[]
	for a in v
		push!(factors, construct_fe(df, a, w))
	end
	# in case where only interacted fixed effect, add constant
	if all(map(z -> typeof(z) <: FixedEffectSlope, factors))
		push!(factors, FixedEffectIntercept(PooledDataArray(fill(1, size(df, 1))), w, :cons))
	end
	return(factors)
end


##############################################################################
##
## Demean algorithm
##
##############################################################################

# Algorithm from lfe: http://cran.r-project.org/web/packages/lfe/vignettes/lfehow.pdf

function demean_factor!(ans::Vector{Float64}, fe::FixedEffectIntercept, mean::Vector{Float64})
	scale = fe.scale
	refs = fe.refs
	w = fe.w
	@inbounds @simd  for i in 1:length(ans)
		 mean[refs[i]] += ans[i] * w[i]
	end
	@inbounds @simd  for i in 1:length(scale)
		 mean[i] *= scale[i] 
	end
	@inbounds @simd  for i in 1:length(ans)
		 ans[i] -= mean[refs[i]] * w[i]
	end
end

function demean_factor!(ans::Vector{Float64}, fe::FixedEffectSlope, mean::Vector{Float64})
	scale = fe.scale
	refs = fe.refs
	x = fe.x
	w = fe.w
	@inbounds @simd  for i in 1:length(ans)
		 mean[refs[i]] += ans[i] * x[i] * w[i]
	end
	@inbounds @simd  for i in 1:length(scale)
		 mean[i] *= scale[i] 
	end
	@inbounds @simd  for i in 1:length(ans)
		 ans[i] -= mean[refs[i]] * x[i] * w[i]
	end
end

function demean!(x::Vector{Float64}, fes::Vector{AbstractFixedEffect}; maxiter::Integer = 1000, tol::FloatingPoint = 1e-8)
	# allocate array of means for each factor
	dict = Dict{AbstractFixedEffect, Vector{Float64}}()
	for fe in fes
		dict[fe] = zeros(Float64, length(fe.scale))
	end
	iterations = maxiter
	converged = false
	if length(fes) == 1 && typeof(fes[1]) <: FixedEffectIntercept
		converged = true
		iterations = 1
		maxiter = 1
	end
	delta = 1.0
	olx = similar(x)
	for iter in 1:maxiter
		@inbounds @simd  for i in 1:length(x)
			olx[i] = x[i]
		end
		for fe in fes
			mean = dict[fe]
			fill!(mean, zero(Float64))
			demean_factor!(x, fe, mean)
		end
		delta = chebyshev(x, olx)
		if delta < tol
			converged = true
			iterations = iter
			break
		end
	end
	return(x, converged, iterations)
end

function demean!(x::DataVector{Float64}, fes::Vector{AbstractFixedEffect}; maxiter::Integer = 1000, tol::FloatingPoint = 1e-8)
	demean(convert(Vector{Float64}, x), fes, maxiter = maxiter, tol = tol)
end



function demean!(X::Matrix{Float64}, fes::Vector{AbstractFixedEffect}; maxiter::Integer = 1000, tol::FloatingPoint = 1e-8)
	convergedv = Bool[]
	iterationsv = Bool[]
	for j in 1:size(X, 2)
		(X[:,j], iterations, converged) = demean!(X[:,j], fes; maxiter = maxiter, tol = tol)
		push!(iterationsv, iterations)
		push!(convergedv, converged)
	end
	return(X, convergedv, iterationsv)
end

