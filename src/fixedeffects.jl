module FixedEffects

export demean
using DataArrays
using DataFrames



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
	    refs = Array(Uint32, length(idx))
	    j = 1
	    for i = 1:length(starts)
	        while (j <= ends[i])
	            refs[idx[j]] =  i
	            j += 1
	        end
	    end
	    l = ends - starts + 1
	    push!(factors, Factor(l, refs))
    end
    return(factors)
end


function demean_vector(factors::Vector{Factor}, x::DataVector)
	delta = 1.0
	while (delta > 1e-6)
		oldx = copy(x)
	    for factor in factors
	        l = factor.size
	        refs = factor.refs
	        mean = zeros(Float64, length(l))
	    	for i = 1:length(x)
	    		mean[refs[i]] += x[i]
	        end
	    	for i = 1:length(l)
	    		mean[i] = mean[i]/l[i]
	        end
	    	for i = 1:length(x)
	    		x[i] = x[i] - mean[refs[i]]
	        end
	    end
	    for i = 1:length(x)
	        delta += (x[i]-oldx[i])^2
	    end
	    delta = sqrt(delta)/length(x)
	end
	return(x)
end



end
