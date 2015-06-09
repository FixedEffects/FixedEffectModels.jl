module FixedEffects

export demean

using DataArrays
using DataFrames
using NumericExtensions



function demean(df::DataFrame, cols::Vector{Symbol}, absorb::Vector{Vector{Symbol}})
	# construct submatrix with no NA
	condition = complete_cases(df)
	subdf = sub(df, condition)
	# construct array of factors
	factors = construct_factors(subdf, absorb)
	out = copy(df)
	for x in cols
		newx = parse("$(x)_p")
		out[newx] = similar(df[x])
		out[condition, newx] = demean_vector(factors, subdf[x])
	end
	return(out)
end

function demean(df::DataFrame, cols::Symbol,  factors::Vector{Vector{Symbol}})
	demean(df, [cols],  factors)
end


# Each element in absorb is transformed into a Factor object
type Factor
	size::Vector{Int64}  # length of each group
	refs::Vector{Uint32} # associates to each row a group
end

function construct_factors(df::SubDataFrame, absorb::Vector{Vector{Symbol}})
	factors = Factor[]
	for a in absorb
	    g = groupby(df, a)
	    idx = g.idx
	    starts = g.starts
	    ends = g.ends

	    # construct size
	    size = ends - starts + 1

	    # construct ref
	    refs = Array(Uint32, length(idx))
	    j = 1
	    for i = 1:length(starts)
	        while (j <= ends[i])
	            refs[idx[j]] =  i
	            j += 1
	        end
	    end

	    push!(factors, Factor(size, refs))
    end
    return(factors)
end


function demean_vector(factors::Vector{Factor}, x::DataVector)
	delta = 1.0
	ans = convert(Vector, x)
	# create mean array for each factor
	dict = Dict{Factor, Vector{Float64}}()
	for factor in factors
		dict[factor] = zeros(Float64, length(factor.size))
	end
	tolerance = (1e-8 * length(ans))^2
	while (delta > tolerance)
		oldans = copy(ans)
	    for factor in factors
	        l = factor.size
	        refs = factor.refs
	        mean = dict[factor]
	        fill!(mean, 0)
	    	for i = 1:length(x)
	    		mean[refs[i]] += ans[i]
	        end
	    	for i = 1:length(l)
	    		mean[i] = mean[i]/l[i]
	        end
	    	for i = 1:length(x)
	    		ans[i] += - mean[refs[i]]
	        end
	    end
	    # check whether close from previous matrix
	    delta = vnormdiff(ans, oldans, 2)
	end
	return(ans)
end



end
