VcovFormula(::Type{Val{:cluster}}, x) = VcovClusterFormula(@eval(@formula(nothing ~ $x)).rhs)

struct VcovClusterFormula <: AbstractVcovFormula
    _::Any
end
allvars(x::VcovClusterFormula) =  vcat([allvars(a) for a in eachterm(x._)]...)

struct VcovClusterMethod <: AbstractVcovMethod
    clusters::DataFrame
end

function VcovMethod(df::AbstractDataFrame, vcovcluster::VcovClusterFormula)
    clusters = vcovcluster._
    vclusters = DataFrame(Matrix{Vector}(undef, size(df, 1), 0))
    for c in eachterm(clusters)
        if isa(c, Term)
            c = Symbol(c)
            v = df[!, c]
            if !isa(v, CategoricalVector)
                v = categorical(v)
            end
            vclusters[!, c] = group(v)
        elseif isa(c, InteractionTerm)
            factorvars, interactionvars = _split(df, c)
            vclusters[!, _name(factorvars)] = group((df[!, v] for v in factorvars)...)
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

# Matrix version is used for IV
function helper_cluster(X::Matrix{Float64}, res::Union{Vector{Float64}, Matrix{Float64}}, f::CategoricalVector)
    X2 = fill(zero(Float64), length(f.pool), size(X, 2) * size(res, 2))
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
