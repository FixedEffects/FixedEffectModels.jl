
module FixedEffects

using NumericExtensions
using DataFrames
export demean


# algorithm from lfe: http://cran.r-project.org/web/packages/lfe/vignettes/lfehow.pdf
max_it = 1000


# Factor is a type that stores size of group and their refs for a group defined by multiple cols
# Type
type Factor
	size::Vector{Int64}  # store the length of each group
	refs::Vector{Uint32} # associates to each row a group
end

# Constructor 
function Factor(df::SubDataFrame, cols::Vector{Symbol})
    groupeddf = groupby(df, cols)
    idx = groupeddf.idx
    starts = groupeddf.starts
    ends = groupeddf.ends

    # size
    size = ends - starts + 1

    # ref
    refs = Array(Uint32, length(idx))
    j = 1
    for i = 1:length(starts)
        while (j <= ends[i])
            refs[idx[j]] =  i
            j += 1
        end
    end

    Factor(size, refs)
end


# Demean_vector demean a vector repeatedly
function demean_vector(factors::Vector{Factor}, x::DataVector)
	delta = 1.0
	ans = convert(Vector{Float64}, x)
	# allocate array of means for each factor
	dict = Dict{Factor, Vector{Float64}}()
	for factor in factors
		dict[factor] = zeros(Float64, length(factor.size))
	end

	tolerance = (1e-8 * length(ans))^2

	for iter = 1:max_it
		oldans = copy(ans)
	    for factor in factors
	        l = factor.size
	        refs = factor.refs
	        mean = dict[factor]
	        fill!(mean, 0)
	    	for i = 1:length(x)
	    		 @inbounds mean[refs[i]] += ans[i]
	        end
	    	for i = 1:length(l)
	    		 @inbounds mean[i] = mean[i]/l[i]
	        end
	    	for i = 1:length(x)
	    		 @inbounds ans[i] += - mean[refs[i]]
	        end
	    end
	    delta = vnormdiff(ans, oldans, 2)
	    if delta < tolerance
	    	break
	    end
	end
	return(ans)
end



# main function
function demean(df::DataFrame, cols::Vector{Symbol}, absorb::Vector{Vector{Symbol}})

	# construct subdataframe wo NA
	condition = complete_cases(df)
	subdf = sub(df, condition)

	# construct an array of factors
	factors = Factor[]
	for a in absorb
		push!(factors, Factor(subdf, a))
	end

	# don't modify input dataset
	out = copy(df)

	# demean each vector sequentially
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







end
