# algorithm from lfe: http://cran.r-project.org/web/packages/lfe/vignettes/lfehow.pdf
using DataFrames, Distances

#
# Type Fe and FeInteracted. These types store reference vector, the size of each group, eventually interaction
#

abstract AbstractFe

immutable type Fe{R<: Integer} <: AbstractFe
	size::Vector{Uint64}  # store the length of each group
	refs::Vector{R} # associates to each row a group
end

function Fe(f::PooledDataArray)
	f = copy(f)
	dropUnusedLevels!(f)
	size = Array(Uint64, length(f.pool))
	fill!(size, 0)
	refs = f.refs
	for i in 1:length(refs)
		size[refs[i]] += 1
	end
	Fe(size, f.refs)
end

immutable type FeInteracted{R<: Integer} <: AbstractFe
	size::Vector{Float64}  # store the sum of x^2 in each group
	refs::Vector{R} # associates to each row a group
	x::Vector{Float64} # the continuous interaction 
end

function FeInteracted(f::PooledDataArray, x::Vector{Float64})
	f = copy(f)
	dropUnusedLevels!(f)
	size = Array(Float64, length(f.pool))
	fill!(size, 0.0)
	refs = f.refs
	for i in 1:length(refs)
		size[refs[i]] += x[i] * x[i]
	end
	FeInteracted(size, f.refs, x)
end


# custom version till Julia 0.4 is released
function dropUnusedLevels!(da::PooledDataArray)
    rr = da.refs
    uu = unique(rr)
    length(uu) == length(da.pool) && return da
    T = eltype(rr)
    su = sort!(uu)
    dict = Dict(su, map(x -> convert(T,x), 1:length(uu)))
    da.refs = map(x -> dict[x], rr)
    da.pool = da.pool[uu]
    da
end	


function construct_fe(df::AbstractDataFrame, a::Expr)
	if a.args[1] == :&
		if (typeof(df[a.args[2]]) <: PooledDataArray) & !(typeof(df[a.args[3]]) <: PooledDataArray)
			f = df[a.args[2]]
			x = convert(Vector{Float64}, df[a.args[3]])
			FeInteracted(f, x)
		elseif (typeof(df[a.args[3]]) <: PooledDataArray) & !(typeof(df[a.args[2]]) <: PooledDataArray)
			f = df[a.args[3]]
			dropUnusedLevels!(f)
			x = convert(Vector{Float64}, df[a.args[2]])
			FeInteracted(f, x)
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
		Fe(f)
	else
		error("$(a) is not a pooled data array")
	end
end


#
# demean_vector_factor. This is the main algorithm
#


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
# demean_vector demeans with respect to all vectors and stop when convergence
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




#
# demean constructs the fixed effects and the demeaned vectors
# 


function demean!(out::AbstractDataFrame, df::AbstractDataFrame, cols::Vector{Symbol}, absorb::Formula)
	# construct subdataframe wo NA
	condition = complete_cases(df)
	subdf = sub(df, condition)
	# construct an array of factors
	factors = AbstractFe[]
	for a in DataFrames.Terms(absorb).terms
		push!(factors, construct_fe(subdf, a))
	end
	# case where only interacted fixed effect : add constant
	if all(map(z -> typeof(z) <: FeInteracted, factors))
		push!(factors, Fe(PooledDataArray(fill(1, size(subdf, 1)))))
	end
	# demean each vector sequentially
	for x in cols
		out[condition, x] =  demean_vector(subdf, factors, subdf[x])
		out[!condition, x] = NA
	end
	return(out)
end

function demean!(df::AbstractDataFrame, cols::Vector{Symbol}, absorb::Formula)
	demean!(df, df, cols, absorb)
end

function demean(df::AbstractDataFrame, cols::Vector{Symbol}, absorb::Formula)
	out = DataFrame(Float64, size(df, 1), length(cols))
	names!(out, cols)
	demean!(out, df, cols, absorb)
end


# simple case with no formula: just demean
function demean!(out::AbstractDataFrame, df::AbstractDataFrame, cols::Vector{Symbol})
	# construct subdataframe wo NA
	condition = complete_cases(df)
	subdf = sub(df, condition)
	for x in cols
		out[condition, x] = subdf[x] .- mean(subdf[x])
		out[!condition, x] = NA
	end
	return(out)
end

function demean!(df::AbstractDataFrame, cols::Vector{Symbol})
	demean!(df, df, cols, absorb)
end

function demean(df::AbstractDataFrame, cols::Vector{Symbol})
	out = DataFrame(Float64, size(df, 1), length(cols))
	names!(out, cols)
	demean!(out, df, cols, absorb)
end










