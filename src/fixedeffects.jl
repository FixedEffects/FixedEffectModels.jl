module FixedEffects
export partial
using DataArrays
using DataFrames


function partial(df::DataFrame, demeans::Vector{Symbol},  factors::Vector{Vector{Symbol}})
	# vector of NA
	condition = complete_cases(df)
	subdf = sub(df, condition)
	# construct factors
	factorlist = construct_factors(subdf, factors)
	a = df
	for x in demeans
		a[condition, x] = demean(factorlist, subdf[x])
		a[!condition, x] = NA
	end
	return(a)
end


function construct_factors(df::SubDataFrame, factors::Vector{Vector{Symbol}})
	factorlist = (Vector{Int64}, Vector{Uint32})[]
	for factor in factors
	    g = groupby(df, factor)
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
	    push!(factorlist, (l, refs))
    end
    return(factorlist)
end


function demean(factorlist::Array{(Vector{Int64}, Vector{Uint32}), 1}, x::DataVector)
	delta = 1.0
	while (delta > 1e-6)
		oldx = copy(x)
	    for factor in factorlist
	        l = factor[1]
	        refs = factor[2]
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

