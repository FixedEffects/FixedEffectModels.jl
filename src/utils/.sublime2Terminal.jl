_hasmissing(x::CategoricalArray{T, N, R}) where {T, N, R} = T >: Missing


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



function group(x) 
	v = categorical(x)
	# need to factorize for cluster at least, otherwise Erik error
	factorize!(v.refs)
end

function group(args...)
	v = categorical(args[1])
	x, ngroups = convert(Vector{UInt}, v.refs), length(levels(v))
	for j = 2:ncols
		v = categorical(args[j])
		x, ngroups = pool_combine!(x, v, ngroups)
	end
	factorize!(x)
end




group(df::AbstractDataFrame, args...) =  group(df, [a for a in args])
group(df::AbstractDataFrame, cols::Vector) =  group((df[c] for c in cols)...)