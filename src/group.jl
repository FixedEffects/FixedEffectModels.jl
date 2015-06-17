using DataArrays, DataFrames



function group(df::AbstractDataFrame; skipna = true) 
    ncols = length(df)
    dv = DataArrays.PooledDataArray(df[ncols])
    if skipna
        x = map(z -> convert(Uint32, z), dv.refs)
        ngroups = length(dv.pool)
        for j = (ncols - 1):-1:1
            dv = DataArrays.PooledDataArray(df[j])
            for i = 1:DataFrames.size(df, 1)
                x[i] += ((dv.refs[i] == 0 | x[i] == 0) ? 0 : (dv.refs[i] - 1) * ngroups)
            end
            ngroups = ngroups * length(dv.pool)
        end
        # factorize
        uu = unique(x)
        T = eltype(x)
        vv = setdiff(uu, zero(T))
        dict = Dict(vv, 1:(length(vv)))
        compact(PooledDataArray(DataArrays.RefArray(map(z -> z == 0 ? zero(T) : dict[z], x)),  [1:length(vv);]))
    else
        # code from groupby
        dv_has_nas = (findfirst(dv.refs, 0) > 0 ? 1 : 0)
        x = map(z -> convert(Uint32, z) + dv_has_nas, dv.refs)
        ngroups = length(dv.pool) + dv_has_nas
        for j = (ncols - 1):-1:1
            dv = DataArrays.PooledDataArray(df[j])
            dv_has_nas = (findfirst(dv.refs, 0) > 0 ? 1 : 0)
            for i = 1:DataFrames.size(df, 1)
                x[i] += (dv.refs[i] + dv_has_nas- 1) * ngroups
            end
            ngroups = ngroups * (length(dv.pool) + dv_has_nas)
        end
        # end of code from groupby
        # factorize
        uu = unique(x)
        T = eltype(x)
        dict = Dict(uu, 1:length(uu))
        compact(PooledDataArray(DataArrays.RefArray(map(z -> dict[z], x)),  [1:length(uu);]))
    end
end


group(df::AbstractDataFrame, cols::Vector; skipna = true) =  group(df[cols]; skipna = skipna)
