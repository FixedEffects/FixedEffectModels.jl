# algorithm from lfe: http://cran.r-project.org/web/packages/lfe/vignettes/lfehow.pdf
using DataFrames, Distances



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




# Demean a vector repeatedly
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
	    	f(df, fe, scale, mean,  ans)
		end
	    delta =  sqeuclidean(ans, oldans)
	    if delta < tolerance
	    	break
	    end
	end
	return(ans)
end


function f(df::AbstractDataFrame, fe::Fe,  scale::Vector{Float64}, mean::Vector{Float64}, ans::Vector{Float64})
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

function f(df::AbstractDataFrame, fe::FeInteracted,  scale::Vector{Float64}, mean::Vector{Float64}, ans::Vector{Float64})
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


# main function
function demean!(df::AbstractDataFrame, cols::Vector{Symbol}, absorb::Formula)
	# construct subdataframe wo NA
	condition = complete_cases(df)
	subdf = sub(df, condition)
	# construct an array of factors
	factors = AbstractFe[]
	for a in DataFrames.Terms(absorb).terms
		push!(factors, construct_fe(subdf, a))
	end
	# demean each vector sequentially
	for x in cols
		subdf[x] =  demean_vector(subdf, factors, subdf[x])
	end
	return(df)
end



# just demean if no formula
function demean!(df::AbstractDataFrame, cols::Vector{Symbol})
	# construct subdataframe wo NA
	condition = complete_cases(df)
	subdf = sub(df, condition)
	# construct an array of factors
	for x in cols
		subdf[x] = subdf[x] .- mean(subdf[x])
	end
	return(df)
end


function demean!(df::AbstractDataFrame, cols::Vector{Symbol}, w::Symbol)
	# construct subdataframe wo NA
	condition = complete_cases(df)
	subdf = sub(df, condition)
	w = weights(subdf[w])
	# construct an array of factors
	for x in cols
		subdf[x] = subdf[x] .- mean(subdf[x], w)
	end
	return(df)
end






