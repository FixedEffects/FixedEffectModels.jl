struct ClusterCovariance{T} <: CovarianceEstimator
    clusters::T
end

cluster(x::Symbol) = ClusterCovariance((x,))
cluster(args...) = ClusterCovariance(args)

Vcov.completecases(df::AbstractDataFrame, v::ClusterCovariance) = DataFrames.completecases(df, collect(v.clusters))

function materialize(df::AbstractDataFrame, v::ClusterCovariance)
    ClusterCovariance(NamedTuple{v.clusters}(ntuple(i -> group(df[!, v.clusters[i]]), length(v.clusters))))
end

function df_FStat(x::RegressionModel, v::ClusterCovariance, ::Bool)
    minimum((length(levels(x)) for x in values(v.clusters))) - 1
end

function S_hat(x::RegressionModel, v::ClusterCovariance) 
    # Cameron, Gelbach, & Miller (2011): section 2.3
    dim = size(modelmatrix(x), 2) * size(residuals(x), 2)
    S = zeros(dim, dim)
    G = typemax(Int)
    for c in combinations(keys(v.clusters))
        # no need for group in case of one fixed effect, since was already done in VcovMethod
        f = (length(c) == 1) ? v.clusters[c[1]] : group((v.clusters[var] for var in c)...)
        # capture length of smallest dimension of multiway clustering in G
        G = min(G, length(f.pool))
        S += (-1)^(length(c) - 1) * helper_cluster(modelmatrix(x), residuals(x), f)
    end
    # scale total vcov estimate by ((N-1)/(N-K)) * (G/(G-1))
    # another option would be to adjust each matrix given by helper_cluster by number of its categories
    # both methods are presented in Cameron, Gelbach and Miller (2011), section 2.3
    # I choose the first option following reghdfe
    rmul!(S, (size(modelmatrix(x), 1) - 1) / dof_residual(x) * G / (G - 1))
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

function StatsBase.vcov(x::RegressionModel, v::ClusterCovariance)
    xtx = inv(crossmodelmatrix(x))
    pinvertible(Symmetric(xtx * S_hat(x, v) * xtx))
end



