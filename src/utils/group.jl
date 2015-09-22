
##############################################################################
##
## group transform multiple PooledDataVector into one
## Output is a PooledArray where pool is type Int64, equal to ranking of group
## NA in some row mean result has NA on this row
## 
##############################################################################

function reftype(sz) 
	sz <= typemax(UInt8)  ? UInt8 :
	sz <= typemax(UInt16) ? UInt16 :
	sz <= typemax(UInt32) ? UInt32 :
	UInt64
end

#  similar todropunusedlevels! but (i) may be NA (ii) change pool to integer
function factorize!(refs::Array)
	uu = unique(refs)
	sort!(uu)
	has_na = uu[1] == 0
	T = reftype(length(uu)-has_na)
	dict = Dict{eltype(refs), T}(zip(uu, (1-has_na):convert(T, length(uu)-has_na)))
	@inbounds @simd for i in 1:length(refs)
		 refs[i] = dict[refs[i]]
	end
	PooledDataArray(RefArray(refs), collect(1:(length(uu)-has_na)))
end

function pool_combine!{T}(x::Array{UInt64, T}, dv::PooledDataVector, ngroups::Integer)
	@inbounds for i in 1:length(x)
	    # if previous one is NA or this one is NA, set to NA
	    x[i] = (dv.refs[i] == 0 || x[i] == zero(UInt64)) ? zero(UInt64) : x[i] + (dv.refs[i] - 1) * ngroups
	end
	return(x, ngroups * length(dv.pool))
end

"""
Group multiple variables into one DataArray

### Arguments
* `df` : AbstractDataFrame
* `cols` : A vector of symbols

### Returns
* `::PooledDataArray` where each value corresponds to a unique combination of values in `cols`

### Details
A typical formula is composed of one dependent variable, exogeneous variables, endogeneous variables, instruments, and high dimensional fixed effects

### Examples
```julia
using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
df[:StateYearPooled] = group(df[:State], df[:Year])
df[:StateYearPooled] = group(df, [:State, :Year])
```
"""

function group(x::AbstractVector) 
	v = PooledDataArray(x)
	PooledDataArray(RefArray(v.refs), collect(1:length(v.pool)))
end

# faster specialization
function group(x::PooledDataVector)
	PooledDataArray(RefArray(copy(x.refs)), collect(1:length(x.pool)))
end

function group(df::AbstractDataFrame) 
	isempty(df) && throw("df is empty")
	ncols = size(df, 2)
	v = df[1]
	ncols = size(df, 2)
	ncols == 1 && return(group(v))
	if typeof(v) <: PooledDataVector
		x = convert(Array{UInt64}, v.refs)
	else
		v = PooledDataArray(v, v.na, UInt64)
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
group(df::AbstractDataFrame, args...) =  group(df, [a for a in args])
function group(args...) 
	df = DataFrame(Any[a for a in args], 
		Symbol[convert(Symbol, "v$i") for i in 1:length(args)])
	return group(df)
end