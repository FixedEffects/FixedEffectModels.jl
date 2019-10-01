Vcov(::Type{Val{:cluster}}, x::Expr) = VcovCluster(@eval(@formula(nothing ~ $x)).rhs)
Vcov(::Type{Val{:cluster}}, x::Symbol) = VcovCluster(StatsModels.Term(x))
Vcov(::Type{Val{:cluster}}, x::Tuple) = VcovCluster((StatsModels.Term(t) for t in x))

struct VcovCluster <: AbstractVcov
    _::Any
end
allvars(x::VcovCluster) =  vcat([allvars(a) for a in eachterm(x._)]...)

struct VcovClusterMethod <: AbstractVcovMethod
    clusters::DataFrame
end

function VcovMethod(df::AbstractDataFrame, vcovcluster::VcovCluster)
    clusters = vcovcluster._
    vclusters = DataFrame(Matrix{Vector}(undef, size(df, 1), 0))
    for c in eachterm(clusters)
        if isa(c, Term)
            c = Symbol(c)
            vclusters[!, c] = group(df[!, c])
        elseif isa(c, InteractionTerm)
            factorvars = Symbol.(terms(c))
            vclusters[!, Symbol(reduce((x1, x2) -> string(x1)*"&"*string(x2), factorvars))] = group((df[!, v] for v in factorvars)...)
        end
    end
    return VcovClusterMethod(vclusters)
end

function df_FStat(v::VcovClusterMethod, ::VcovData, ::Bool)
    minimum((length(v.clusters[!, c].pool) for c in names(v.clusters))) - 1
end

function vcov!(v::VcovClusterMethod, x::VcovData)
    S = shat!(v, x)
    invcrossmatrix = inv(x.crossmatrix)
    return pinvertible(Symmetric(invcrossmatrix * S * invcrossmatrix))
end

function shat!(v::VcovClusterMethod, x::VcovData{T, N}) where {T, N}
    # Cameron, Gelbach, & Miller (2011): section 2.3
    dim = size(x.regressors, 2) * size(x.residuals, 2)
    S = zeros(dim, dim)
    G = typemax(Int)
    for c in combinations(names(v.clusters))
        # no need for group in case of one fixed effect, since was already done in VcovMethod
        f = (length(c) == 1) ? v.clusters[!, c[1]] : group((v.clusters[!, var] for var in c)...)
        # capture length of smallest dimension of multiway clustering in G
        G = min(G, length(f.pool))
        S += (-1)^(length(c) - 1) * helper_cluster(x.regressors, x.residuals, f)
    end
    # scale total vcov estimate by ((N-1)/(N-K)) * (G/(G-1))
    # another option would be to adjust each matrix given by helper_cluster by number of its categories
    # both methods are presented in Cameron, Gelbach and Miller (2011), section 2.3
    # I choose the first option following reghdfe
    rmul!(S, (size(x.regressors, 1) - 1) / x.dof_residual * G / (G - 1))
end

# res is a Vector in OLS, Matrix in IV
function helper_cluster(X::Matrix, res::Union{Vector, Matrix}, f::CategoricalVector)
    X2 = zeros(eltype(X), length(f.pool), size(X, 2) * size(res, 2))
    index = 0
    for k in 1:size(res, 2)
        for j in 1:size(X, 2)
            index += 1
            @inbounds @simd for i in 1:size(X, 1)
                X2[f.refs[i], index] += X[i, j] * res[i, k]
            end
        end
    end
    return Symmetric(X2' * X2)
end

function pinvertible(A::Symmetric, tol = eps(real(float(one(eltype(A))))))
    eigval, eigvect = eigen(A)
    small = eigval .<= tol
    if any(small)
        @warn "estimated covariance matrix of moment conditions not of full rank.
                 model tests should be interpreted with caution."
        eigval[small] .= 0
        return Symmetric(eigvect' * Diagonal(eigval) * eigvect)
    else
        return A
    end
end
