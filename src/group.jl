using DataArrays, DataFrames

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
