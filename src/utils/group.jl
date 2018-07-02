
##############################################################################
##
## group transform multiple CategoricalVector into one
## Output is a PooledArray where pool is type Int64, equal to ranking of group
## NA in some row mean result has NA on this row
## 
##############################################################################
_hasmissing(x::CategoricalArray{T, N, R}) where {T, N, R} = T >: Missing


function group(x::AbstractVector) 
	v = categorical(x)
	if _hasmissing(v)
		T = Union{Int, Missing}
	else
		T = Int
	end
	# need to factorize for cluster at least, otherwise Erik error
	factorize!(v.refs)
end



function pool_combine!(x::Array{T, N}, dv::CategoricalVector, ngroups::Integer) where {T, N}
	for i in 1:length(x)
	    # if previous one is NA or this one is NA, set to NA
	    x[i] = (dv.refs[i] == 0 || x[i] == zero(T)) ? zero(T) : x[i] + (dv.refs[i] - 1) * ngroups
	end
	return x, ngroups * length(levels(dv))
end

#  drop unused levels
function factorize!(refs::Vector{T}) where {T}
	uu = unique(refs)
	sort!(uu)
	has_missing = uu[1] == 0
	dict = Dict{T, Int}(zip(uu, (1-has_missing):(length(uu)-has_missing)))
	newrefs = zeros(UInt32, length(refs))
	for i in 1:length(refs)
		 newrefs[i] = dict[refs[i]]
	end
	if has_missing
		Tout = Union{Int, Missing}
	else
		Tout = Int
	end
	CategoricalArray{Tout, 1}(newrefs, CategoricalPool(collect(1:(length(uu)-has_missing))))
end

function group(df::AbstractDataFrame) 
	isempty(df) && throw("df is empty")
	ncols = size(df, 2)
	v = df[1]
	ncols = size(df, 2)
	ncols == 1 && return(group(v))
	v = categorical(v)
	x, ngroups = convert(Vector{UInt}, v.refs), length(levels(v))
	for j = 2:ncols
		v = categorical(df[j])
		x, ngroups = pool_combine!(x, v, ngroups)
	end
	factorize!(x)
end
group(df::AbstractDataFrame, cols::Vector) =  group(df[cols])
group(df::AbstractDataFrame, args...) =  group(df, [a for a in args])
function group(args...) 
	df = DataFrame(Any[a for a in args], 
		Symbol[convert(Symbol, "v$i") for i in 1:length(args)])
	return group(df)
end