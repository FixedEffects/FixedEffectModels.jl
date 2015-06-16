module FixedEffects
using DataFrames, DataArrays

export group, demean, demean_factors

function group(df::AbstractDataFrame) 
	# from groupby
	ncols = length(df)
    dv = DataArrays.PooledDataArray(df[ncols])
    dv_has_nas = (findfirst(dv.refs, 0) > 0 ? 1 : 0)
    x = copy(dv.refs) .+ dv_has_nas
    ngroups = length(dv.pool) + dv_has_nas
    for j = (ncols - 1):-1:1
        dv = DataArrays.PooledDataArray(df[j])
        dv_has_nas = (findfirst(dv.refs, 0) > 0 ? 1 : 0)
        for i = 1:DataFrames.size(df, 1)
            x[i] += (dv.refs[i] + dv_has_nas- 1) * ngroups
        end
        ngroups = ngroups * (length(dv.pool) + dv_has_nas)
    end
    # factorize
    uu = unique(x)
    T = eltype(x)
    dict = Dict(uu, map(z -> convert(T,z), 1:length(uu)))
    PooledDataArray(DataArrays.RefArray(map(z -> dict[z], x)),  [1:length(uu)])
end


abstract AbstractFe

immutable type Fe{R<: Integer} <: AbstractFe
	size::Vector{Uint64}  # store the length of each group
	refs::Vector{R} # associates to each row a group
end

function Fe(f::PooledDataArray)
	f = copy(f)
	dropUnusedLevels!(f)
	size = Array(Uint64, length(f.pool))
	fill!(size, 0)
	refs = f.refs
	for i in 1:length(refs)
		size[refs[i]] += 1
	end
	Fe(size, f.refs)
end


immutable type FeInteracted{R<: Integer} <: AbstractFe
	size::Vector{Float64}  # store the sum of x^2 in each group
	refs::Vector{R} # associates to each row a group
	x::Vector{Float64}
end

function FeInteracted(f::PooledDataArray, x::Vector{Float64})
	f = copy(f)
	dropUnusedLevels!(f)
	size = Array(Float64, length(f.pool))
	fill!(size, 0.0)
	refs = f.refs
	for i in 1:length(refs)
		size[refs[i]] += x[i] * x[i]
	end
	FeInteracted(size, f.refs, x)
end


# custome version till 0.4 is released
function dropUnusedLevels!(da::PooledDataArray)
    rr = da.refs
    uu = unique(rr)
    length(uu) == length(da.pool) && return da
    T = eltype(rr)
    su = sort!(uu)
    dict = Dict(su, map(x -> convert(T,x), 1:length(uu)))
    da.refs = map(x -> dict[x], rr)
    da.pool = da.pool[uu]
    da
end	



include("linearfixedeffects.jl")
include("factorfixedeffects.jl")

end