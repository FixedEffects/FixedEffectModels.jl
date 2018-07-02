VcovFormula(::Type{Val{:cluster}}, x) = VcovClusterFormula(Terms(@eval(@formula($nothing ~ $x))).terms)

type VcovClusterFormula <: AbstractVcovFormula
    _::Vector{Any}
end
allvars(x::VcovClusterFormula) =  vcat([allvars(a) for a in x._]...)


type VcovClusterMethod <: AbstractVcovMethod
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
            typeof(p) <: CategoricalVector || error("Cluster variable $(c) is of type $(typeof(p)), but should be a CategoricalVector.")
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
    return pinvertible(out)
end
function shat!(v::VcovClusterMethod, x::VcovData{T, 1}) where {T}
    # Cameron, Gelbach, & Miller (2011).
    clusternames = names(v.clusters)
    X = x.regressors .* x.residuals
    S = fill(zero(Float64), (size(X, 2), size(X, 2)))
    for i in 1:length(clusternames)
        for c in combinations(clusternames, i)
            #note that I only want the pools that are actually used, so group() returns a categorical arrays where all pools are used
            f = group(v.clusters[c])
            if rem(length(c), 2) == 1
                S += helper_cluster(X, f)
            else
                S -= helper_cluster(X, f)
            end
        end
    end
    scale!(S, (size(X, 1) - 1) / x.df_residual)
    return S
end

function helper_cluster(Xu::Matrix{Float64}, f::CategoricalVector)
    if length(f.pool) == size(Xu, 1)
        # if only one obs by pool, use White, as in Petersen (2009) & Thomson (2011)
        return At_mul_B(Xu, Xu)
    else
        # otherwise
        X2 = fill(zero(Float64), (length(f.pool), size(Xu, 2)))
        for j in 1:size(Xu, 2)
             for i in 1:size(Xu, 1)
                X2[f.refs[i], j] += Xu[i, j]
            end
        end
        out = At_mul_B(X2, X2)
        scale!(out, length(f.pool) / (length(f.pool)- 1))
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
            f = group(v.clusters[c])
            if rem(length(c), 2) == 1
                S += helper_cluster(X, res, f)
            else
                S -= helper_cluster(X, res, f)
            end
        end
    end
    scale!(S, (size(X, 1) - 1) / x.df_residual)
    return S
end

# S_{(l-1) * K + k, (l'-1)*K + k'} = \sum_g (\sum_{i in g} X[i, k] res[i, l]) (\sum_{i in g} X[i, k'] res[i, l'])
function helper_cluster(X::Matrix{Float64}, res::Matrix{Float64}, f::CategoricalVector)
    dim = size(X, 2) * size(res, 2)
    nobs = size(X, 1)
    S = fill(zero(Float64), (dim, dim))
    temp = fill(zero(Float64), length(f.pool), dim)
    index = 0
    for l in 1:size(res, 2), k in 1:size(X, 2)
        index += 1
        for i in 1:nobs
            temp[f.refs[i], index] += X[i, k] * res[i, l]
        end
    end
    S = At_mul_B(temp, temp)
    scale!(S, length(f.pool) / (length(f.pool) - 1))
    return S
end


function pinvertible(A::Matrix, tol = eps(real(float(one(eltype(A))))))
    eigval, eigvect = eig(Symmetric(A))
    small = eigval .<= tol
    if any(small)
        warn("estimated covariance matrix of moment conditions not of full rank.
                 model tests should be interpreted with caution.")
        eigval[small] = 0
        return  eigvect' * diagm(eigval) * eigvect
    else
        return A
    end
end