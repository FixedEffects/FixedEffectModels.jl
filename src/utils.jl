using DataArrays

function dropUnusedLevels!(f::PooledDataArray)
	rr = f.refs
	uu = unique(rr)
	T = eltype(rr)
	su = sort!(uu)
	dict = Dict(su, map(x -> convert(T,x), 1:length(uu)))
	f.refs = map(x -> dict[x], rr)
	f.pool = f.pool[uu]
	f
end

dropUnusedLevels!(f) = f
