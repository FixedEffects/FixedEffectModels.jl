using DataFrames, DataArrays, Distances

#
# Type Fe and FeInteracted. For each fixed effect, this stores the reference vector (ie a map of each row to a group), the size of each group, and, for FeInteracted, the interaction variable
#

abstract AbstractFe

immutable type Fe{R <: Integer} <: AbstractFe
	name::Symbol
	size::Vector{Uint64}  # store the length of each group
	refs::Vector{R} # associates to each row a group
end

immutable type FeInteracted{R <: Integer} <: AbstractFe
	name::Symbol
	size::Vector{Float64}  # store the sum of x^2 in each group
	refs::Vector{R} # associates to each row a group
	xname::Symbol
	x::Vector{Float64} # the continuous interaction 
end


function clean_fe(f::PooledDataArray)
    size = Array(Uint64, length(f.pool))
    fill!(size, 0)
    refs = f.refs
    for i in 1:length(refs)
    	size[refs[i]] += 1
    end
    (size, refs)
end




function construct_fe(df::AbstractDataFrame, a::Expr)
	if a.args[1] == :&
		if (typeof(df[a.args[2]]) <: PooledDataArray) & !(typeof(df[a.args[3]]) <: PooledDataArray)
			f = df[a.args[2]]
			x = convert(Vector{Float64}, df[a.args[3]])
			(size, refs) = clean_fe(f)
			FeInteracted(a.args[2], size, refs, a.args[3], x)
		elseif (typeof(df[a.args[3]]) <: PooledDataArray) & !(typeof(df[a.args[2]]) <: PooledDataArray)
			f = df[a.args[3]]
			x = convert(Vector{Float64}, df[a.args[2]])
			(size, refs) = clean_fe(f)
			FeInteracted(a.args[3], size, refs, a.args[2], x)
		else
			error("& is not of the form factor & nonfactor")
		end
	else
		error("Formula should be composed of & and symbols")
	end
end

function construct_fe(df::AbstractDataFrame, a::Symbol)
	if typeof(df[a]) <: PooledDataArray
		f = df[a]
		(size, refs) = clean_fe(f)
		Fe(a, size, refs)
	else
		error("$(a) is not a pooled data array")
	end
end


#
# demean_vector_factor. This is the main algorithm
# Algorithm from lfe: http://cran.r-project.org/web/packages/lfe/vignettes/lfehow.pdf


function demean_vector_factor(df::AbstractDataFrame, fe::Fe,  scale::Vector{Float64}, mean::Vector{Float64}, ans::Vector{Float64})
	refs = fe.refs
	@simd for i in 1:length(ans)
		@inbounds mean[refs[i]] += ans[i]
    end
	@simd for i in 1:length(scale)
		 @inbounds mean[i] = mean[i] * scale[i] 
    end
	@simd for i in 1:length(ans)
		@inbounds ans[i] += - mean[refs[i]]
    end
end

function demean_vector_factor(df::AbstractDataFrame, fe::FeInteracted,  scale::Vector{Float64}, mean::Vector{Float64}, ans::Vector{Float64})
	refs = fe.refs
	@simd for i in 1:length(ans)
		@inbounds mean[refs[i]] += ans[i] * fe.x[i]
    end
	@simd for i in 1:length(scale)
		 @inbounds mean[i] = mean[i] * scale[i] 
    end
	@simd for i in 1:length(ans)
		@inbounds ans[i] += - mean[refs[i]] * fe.x[i]
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












