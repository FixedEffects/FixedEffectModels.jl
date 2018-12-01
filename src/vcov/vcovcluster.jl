VcovFormula(::Type{Val{:cluster}}, x) = VcovClusterFormula(Terms(@eval(@formula($nothing ~ $x))).terms)

struct VcovClusterFormula <: AbstractVcovFormula
    _::Vector{Any}
end
allvars(x::VcovClusterFormula) =  vcat([allvars(a) for a in x._]...)


struct VcovClusterMethod <: AbstractVcovMethod
    clusters::DataFrame
end


function VcovMethod(df::AbstractDataFrame, vcovcluster::VcovClusterFormula) 
    clusters = vcovcluster._
    vclusters = DataFrame(Vector, size(df, 1), 0)
    for c in clusters
        if isa(c, Symbol)
            typeof(df[c]) <: CategoricalVector || error("Cluster variable $(c) is of type $(typeof(df[c])), but should be a CategoricalVector.")
            cname = c
            p = group(df[c])
        elseif isa(c, Expr)
            factorvars, interactionvars = _split(df, allvars(c))
            cname = _name(factorvars)
            p = group((df[v] for v in factorvars)...)
        end
        vclusters[cname] = p
    end
    return VcovClusterMethod(vclusters)
end

df_FStat(v::VcovClusterMethod, ::VcovData, ::Bool) = minimum((length(v.clusters[c].pool) for c in names(v.clusters))) - 1


function vcov!(v::VcovClusterMethod, x::VcovData)
    S = shat!(v, x)
    out = sandwich(x.crossmatrix, S)
    return pinvertible(out)
end

function shat!(v::VcovClusterMethod, x::VcovData{T, N}) where {T, N}
    # Cameron, Gelbach, & Miller (2011).
    dim = size(x.regressors, 2) * size(x.residuals, 2)
    S = fill(zero(Float64), (dim, dim))
    for c in combinations(names(v.clusters))
        if length(c) == 1
            # no need for group
            f = v.clusters[c[1]]
        else
            f = group((v.clusters[var] for var in c)...)
        end
        if rem(length(c), 2) == 1
            S += helper_cluster(x.regressors, x.residuals, f)
        else
            S -= helper_cluster(x.regressors, x.residuals, f)
        end
    end
    rmul!(S, (size(x.regressors, 1) - 1) / x.dof_residual)
    return S
end



# Matrix version is used for IV
function helper_cluster(X::Matrix{Float64}, res::Union{Vector{Float64}, Matrix{Float64}}, f::CategoricalVector)
    dim = size(X, 2) * size(res, 2)
    X2 = fill(zero(Float64), length(f.pool), dim)
    index = 0
    for k in 1:size(res, 2)
        for j in 1:size(X, 2)
            index += 1
            @inbounds @simd for i in 1:size(X, 1)
                X2[f.refs[i], index] += X[i, j] * res[i, k]
            end
        end
    end
    S2 = X2' * X2
    if length(f.pool) < size(X, 1)
        # if only one obs by pool, for instance cluster(year state)
        # use White, as in Petersen (2009) & Thomson (2011) 
        rmul!(S2, length(f.pool) / (length(f.pool) - 1))
    end
    return S2
end


function pinvertible(A::Matrix, tol = eps(real(float(one(eltype(A))))))
    eigval, eigvect = eigen(Symmetric(A))
    small = eigval .<= tol
    if any(small)
        warn("estimated covariance matrix of moment conditions not of full rank.
                 model tests should be interpreted with caution.")
        eigval[small] = 0
        return  eigvect' * Diagonal(eigval) * eigvect
    else
        return A
    end
end