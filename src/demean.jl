using DataFrames, DataArrays


function helper_demean!(out::AbstractDataFrame, df::AbstractDataFrame, cols::Vector{Symbol}, absorb::Formula)
	
	# construct subdataframe wo NA
	condition = complete_cases(df)
	subdf = sub(df, condition)
	
	# construct an array of factors
	factors = AbstractFe[]
	for a in DataFrames.Terms(absorb).terms
		push!(factors, construct_fe(subdf, a, sqrtw))
	end

	# case where only interacted fixed effect : add constant
	if all(map(z -> typeof(z) <: FeInteracted, factors))
		push!(factors, Fe(PooledDataArray(fill(1, size(subdf, 1))), sqrtw, :cons))
	end
	
	# demean each vector sequentially
	for x in cols
		out[condition, x] =  demean_vector(factors, subdf[x])
		out[!condition, x] = NA
	end

	out
end

function helper_demean!(out::AbstractDataFrame, df::AbstractDataFrame, cols::Vector{Symbol})
	# construct subdataframe wo NA
	condition = complete_cases(df)
	subdf = sub(df, condition)
	for x in cols
		out[condition, x] = subdf[x] .- mean(subdf[x])
		out[!condition, x] = NA
	end
	return(out)
end

#
# Exported function
#

function demean!(df::AbstractDataFrame, cols::Vector{Symbol}, absorb::Formula)
	helper_demean!(df, df, cols, absorb)
end

function demean(df::AbstractDataFrame, cols::Vector{Symbol}, absorb::Formula)
	out = DataFrame(Float64, size(df, 1), length(cols))
	names!(out, cols)
	helper_demean!(out, df, cols, absorb)
	out
end

function demean!(df::AbstractDataFrame, cols::Vector{Symbol})
	helper_demean(df, df, cols)
end

function demean(df::AbstractDataFrame, cols::Vector{Symbol})
	out = DataFrame(Float64, size(df, 1), length(cols))
	names!(out, cols)
	helper_demean!(out, df, cols)
end








