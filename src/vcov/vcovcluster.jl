VcovFormula(::Type{Val{:cluster}}, x) = VcovClusterFormula(Terms(Formula(nothing, x)).terms)

mutable struct VcovClusterFormula <: AbstractVcovFormula
    _::Vector{Any}
end
allvars(x::VcovClusterFormula) =  vcat([allvars(a) for a in x._]...)


mutable struct VcovClusterMethod <: AbstractVcovMethod
    clusters::DataFrame
    size::Dict{Symbol, Int}
end


function VcovMethod(df::AbstractDataFrame, vcovcluster::VcovClusterFormula) 
    clusters = vcovcluster._
    vclusters = DataFrame(Vector, size(df, 1), 0)
    vsize = Dict{Symbol, Int}()
    for c in clusters
        if isa(c, Symbol)
            cname = c
            p = df[c]
            typeof(p) <: PooledDataVector || error("Cluster variable $(c) is of type $(typeof(p)), but should be a PooledDataVector.")
        elseif isa(c, Expr)
            factorvars, interactionvars = _split(df, allvars(c))
            cname = _name(factorvars)
            p = group(df, factorvars)
        end
        vclusters[cname] = p
        # may be subset / NA
        vsize[cname] = length(unique(p.refs))
    end
    return VcovClusterMethod(vclusters, vsize)
end

df_FStat(v::VcovClusterMethod, ::VcovData, ::Bool) = minimum(values(v.size)) - 1


function vcov!(v::VcovClusterMethod, x::VcovData)
    S = shat!(v, x)
    out = sandwich(x.crossmatrix, S)
    # Cameron, Gelbach, & Miller (2011)
    pinvertible(out)
    return out
end
function shat!(v::VcovClusterMethod, x::VcovData{T, 1}) where {T}
    # Cameron, Gelbach, & Miller (2011).
    clusternames = names(v.clusters)
    X = x.regressors
    X .= X .* x.residuals
    S = fill(zero(Float64), (size(X, 2), size(X, 2)))
    for i in 1:length(clusternames)
        for c in combinations(clusternames, i)
            if length(c) == 1
                f = (v.clusters)[c[1]]
                # no need to group in this case
                fsize = (v.size)[c[1]]
            else
                df = v.clusters[c]
                f = group(df)
                fsize = length(f.pool)
            end
            if rem(length(c), 2) == 1
                S += helper_cluster(X, f, fsize)
            else
                S -= helper_cluster(X, f, fsize)
            end
        end
    end
    scale!(S, (size(X, 1) - 1) / x.df_residual)
    return S
end

function helper_cluster(Xu::Matrix{Float64}, f::PooledDataVector, fsize::Int)
    if fsize == size(Xu, 1)
        # if only one obs by pool, use White, as in Petersen (2009) & Thomson (2011)
        return At_mul_B(Xu, Xu)
    else
        # otherwise
        X2 = fill(zero(Float64), (fsize, size(Xu, 2)))
        for j in 1:size(Xu, 2)
             @inbounds @simd for i in 1:size(Xu, 1)
                X2[f.refs[i], j] += Xu[i, j]
            end
        end
        out = At_mul_B(X2, X2)
        scale!(out, fsize / (fsize- 1))
        return out
    end
end

function shat!(v::VcovClusterMethod, x::VcovData{T, 2}) where {T}
    # Cameron, Gelbach, & Miller (2011).
    clusternames = names(v.clusters)
    X = x.regressors
    res = x.residuals
    dim = (size(X, 2) *size(res, 2))
    S = fill(zero(Float64), (dim, dim))
    for i in 1:length(clusternames)
        for c in combinations(clusternames, i)
            if length(c) == 1
                f = (v.clusters)[c[1]]
                # no need to group in this case
                fsize = (v.size)[c[1]]
            else
                df = v.clusters[c]
                f = group(df)
                fsize = length(f.pool)
            end
            if rem(length(c), 2) == 1
                S += helper_cluster(X, res, f, fsize)
            else
                S -= helper_cluster(X, res, f, fsize)
            end
        end
    end
    scale!(S, (size(X, 1) - 1) / x.df_residual)
    return S
end

# S_{(l-1) * K + k, (l'-1)*K + k'} = \sum_g (\sum_{i in g} X[i, k] res[i, l]) (\sum_{i in g} X[i, k'] res[i, l'])
function helper_cluster(X::Matrix{Float64}, res::Matrix{Float64}, f::PooledDataVector, fsize::Int)
    dim = size(X, 2) * size(res, 2)
    nobs = size(X, 1)
    S = fill(zero(Float64), (dim, dim))
    temp = fill(zero(Float64), fsize, dim)
    if fsize == nobs
        index = 0
        for l in 1:size(res, 2), k in 1:size(X, 2)
            index += 1
            @simd for i in 1:nobs
                temp[i, index] = X[i, k] * res[i, l]
            end
        end
        S = At_mul_B(temp, temp)
    else
        index = 0
        for l in 1:size(res, 2), k in 1:size(X, 2)
            index += 1
            @simd for i in 1:nobs
                temp[f.refs[i], index] += X[i, k] * res[i, l]
            end
        end
        S = At_mul_B(temp, temp)
        scale!(S, fsize / (fsize - 1))
    end
    return S
end


function pinvertible(A::Matrix, tol = eps(real(float(one(eltype(A))))))
    SVD         = svdfact(A, thin=true)
    Stype       = eltype(SVD.S)
    small = SVD.S .<= tol
    if any(small)
        warn("estimated covariance matrix of moment conditions not of full rank.
                 model tests should be interpreted with caution.")
        SVD.S[small] = 0
        return  SVD.U * diagm(SVD.S) * SVD.Vt
    else
        return A
    end
end