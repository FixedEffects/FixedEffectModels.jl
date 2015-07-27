
##############################################################################
##
## group transform multiple PooledDataVector into one
## Output is a PooledArray where pool is type Int64, equal to ranking of group
## NA in some row mean result has NA on this row
## 
##############################################################################

function reftype(sz) 
	sz <= typemax(Uint8)  ? Uint8 :
	sz <= typemax(Uint16) ? Uint16 :
	sz <= typemax(Uint32) ? Uint32 :
	Uint64
end

#  similar todropunusedlevels! but (i) may be NA (ii) change pool to integer
function factorize!(refs::Array)
	uu = unique(refs)
	sort!(uu)
	has_na = uu[1] == 0
	T = reftype(length(uu)-has_na)
	dict = Dict(uu, (1-has_na):convert(T, length(uu)-has_na))
	@inbounds @simd for i in 1:length(refs)
		 refs[i] = dict[refs[i]]
	end
	PooledDataArray(RefArray(refs), collect(1:(length(uu)-has_na)))
end

function pool_combine!{T}(x::Array{Uint64, T}, dv::PooledDataVector, ngroups::Integer)
	@inbounds for i in 1:length(x)
	    # if previous one is NA or this one is NA, set to NA
	    x[i] = (dv.refs[i] == 0 || x[i] == zero(Uint64)) ? zero(Uint64) : x[i] + (dv.refs[i] - 1) * ngroups
	end
	return(x, ngroups * length(dv.pool))
end

function group(x::AbstractVector) 
	v = PooledDataArray(x)
	PooledDataArray(RefArray(v.refs), collect(1:length(v.pool)))
end

# faster specialization
function group(x::PooledDataVector)
	PooledDataArray(RefArray(copy(x.refs)), collect(1:length(x.pool)))
end
function group(df::AbstractDataFrame) 
	ncols = size(df, 2)
	v = df[1]
	ncols = size(df, 2)
	ncols == 1 && return(group(v))
	if typeof(v) <: PooledDataVector
		x = convert(Array{Uint64}, v.refs)
	else
		v = PooledDataArray(v, v.na, Uint64)
		x = v.refs
	end
	ngroups = length(v.pool)
	for j = 2:ncols
		v = PooledDataArray(df[j])
		(x, ngroups) = pool_combine!(x, v, ngroups)
	end
	return(factorize!(x))
end
group(df::AbstractDataFrame, cols::Vector) =  group(df[cols])
