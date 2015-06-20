using DataFrames, DataArrays, Distances

#
# Type Fe and FeInteracted. For each fixed effect, this stores the reference vector (ie a map of each row to a group), the size of each group, and, for FeInteracted, the interaction variable
#

abstract AbstractFe

immutable type Fe{R} <: AbstractFe
	refs::Vector{R}
	w::Vector{Float64}
	size::Vector{Float64}
	name::Symbol
end

immutable type FeInteracted{R} <: AbstractFe
	refs::Vector{R}
	w::Vector{Float64}
	size::Vector{Float64}
	x::Vector{Float64} # the continuous interaction 
	name::Symbol
	xname::Symbol
end


function Fe(f::PooledDataArray, w::Vector{Float64}, name::Symbol)
	scale = fill(zero(Float64), length(f.pool))
    refs = f.refs
    for i in 1:length(refs)
    	scale[refs[i]] += w[i]^2 
    end
    Fe(refs, w, scale, name)
end

function FeInteracted(f::PooledDataArray, w::Vector{Float64}, x::Vector{Float64}, name::Symbol, xname::Symbol)
	scale = fill(zero(Float64), length(f.pool))
    refs = f.refs
    for i in 1:length(refs)
    	scale[refs[i]] += (x[i] * w[i])^2 
    end
    FeInteracted(refs, w, scale, x, name, xname)
end


function construct_fe(df::AbstractDataFrame, a::Expr, w::Vector{Float64})
	if a.args[1] == :&
		if (typeof(df[a.args[2]]) <: PooledDataArray) & !(typeof(df[a.args[3]]) <: PooledDataArray)
			f = df[a.args[2]]
			x = convert(Vector{Float64}, df[a.args[3]])
			FeInteracted(f, w, x, a.args[2], a.args[3])

		elseif (typeof(df[a.args[3]]) <: PooledDataArray) & !(typeof(df[a.args[2]]) <: PooledDataArray)
			f = df[a.args[3]]
			x = convert(Vector{Float64}, df[a.args[2]])
			FeInteracted(f, w, x, a.args[3], a.args[2])
		else
			error("& is not of the form factor & nonfactor")
		end
	else
		error("Formula should be composed of & and symbols")
	end
end

function construct_fe(df::AbstractDataFrame, a::Symbol, w::Vector{Float64})
	if typeof(df[a]) <: PooledDataArray
		f = df[a]
		Fe(f, w, a)
	else
		error("$(a) is not a pooled data array")
	end
end


#
# demean_vector_factor. This is the main algorithm
# Algorithm from lfe: http://cran.r-project.org/web/packages/lfe/vignettes/lfehow.pdf

function demean_vector_factor(df::AbstractDataFrame, fe::Fe,  scale::Vector{Float64}, mean::Vector{Float64}, ans::Vector{Float64})
	refs = fe.refs
	w = fe.w
	@simd for i in 1:length(ans)
		@inbounds mean[refs[i]] += ans[i] * w[i]
    end
	@simd for i in 1:length(scale)
		 @inbounds mean[i] = mean[i] * scale[i] 
    end
	@simd for i in 1:length(ans)
		@inbounds ans[i] -= mean[refs[i]] * w[i]
    end
end

function demean_vector_factor(df::AbstractDataFrame, fe::FeInteracted,  scale::Vector{Float64}, mean::Vector{Float64}, ans::Vector{Float64})
	refs = fe.refs
	x = fe.x
	w = fe.w
	@simd for i in 1:length(ans)
		@inbounds mean[refs[i]] += ans[i] * x[i] * w[i]
    end
	@simd for i in 1:length(scale)
		 @inbounds mean[i] = mean[i] * scale[i] 
    end
	@simd for i in 1:length(ans)
		@inbounds ans[i] -= mean[refs[i]] * x[i] * w[i]
    end
end


#
# demean_vector applieds demean_vector_factor repeatedly and stop when convergence
#

function demean_vector(df::AbstractDataFrame, fes::Vector{AbstractFe}, x::DataVector)

	max_iter = 1000
	tolerance = ((1e-8 * length(x))^2)::Float64
	delta = 1.0
	ans = convert(Vector{Float64}, x)
	oldans = similar(ans)
	# allocate array of means for each factor
	dict1 = Dict{AbstractFe, Vector{Float64}}()
	dict2 = Dict{AbstractFe, Vector{Float64}}()
	for fe in fes
		dict1[fe] = zeros(Float64, length(fe.size))
		dict2[fe] = 1.0 ./ fe.size
	end

	for iter in 1:max_iter
		@simd for i in 1:length(x)
			@inbounds oldans[i] = ans[i]
		end
	    for fe in fes
	    	mean = dict1[fe]
	    	scale = dict2[fe]
	    	fill!(mean, 0.0)
	    	demean_vector_factor(df, fe, scale, mean,  ans)
		end
	    delta =  sqeuclidean(ans, oldans)
	    if delta < tolerance
	    	break
	    end
	end
	return(ans)
end












