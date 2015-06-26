##############################################################################
##
## Fe and FixedEffectSlope
##
##############################################################################

# For each fixed effect, this stores the reference vector (ie a map of each row to a group), the weights, the size of each group, and, for FixedEffectSlope, the interaction variable

abstract AbstractFixedEffect

immutable type FixedEffectIntercept{R} <: AbstractFixedEffect
	refs::Vector{R}
	w::Vector{Float64}
	scale::Vector{Float64}
	name::Symbol
end

immutable type FixedEffectSlope{R} <: AbstractFixedEffect
	refs::Vector{R}
	w::Vector{Float64}
	scale::Vector{Float64}
	x::Vector{Float64} # the continuous interaction 
	name::Symbol
	xname::Symbol
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

function demean_vector_factor!(ans::Vector{Float64}, fe::FixedEffectIntercept, mean::Vector{Float64})
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
	return(ans)
end

function demean_vector_factor!(ans::Vector{Float64}, fe::FixedEffectSlope, mean::Vector{Float64})
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
	return(ans)
end

function demean_vector!(x::Vector{Float64}, fes::Vector{AbstractFixedEffect})
	tolerance = ((1e-8 * length(x))^2)::Float64
	delta = 1.0
	if length(fes) == 1 && typeof(fes[1]) <: FixedEffectIntercept
		max_iter = 1
	else
		max_iter = 1000
	end
	olx = similar(x)
	# allocate array of means for each factor
	dict1 = Dict{AbstractFixedEffect, Vector{Float64}}()
	dict2 = Dict{AbstractFixedEffect, Vector{Float64}}()
	for fe in fes
		dict1[fe] = zeros(Float64, length(fe.scale))
	end
	for iter in 1:max_iter
		@inbounds @simd  for i in 1:length(x)
			olx[i] = x[i]
		end
		for fe in fes
			mean = dict1[fe]
			fill!(mean, zero(Float64))
			demean_vector_factor!(x, fe, mean)
		end
		delta = sqeuclidean(x, olx)
		if delta < tolerance
			break
		end
	end
	return(x)
end

function demean_vector!(x::DataVector{Float64}, fes::Vector{AbstractFixedEffect})
	demean_vector(convert(Vector{Float64}, x), fes)
end

