function tss(y::AbstractVector, hasintercept::Bool, weights::AbstractWeights)
    if hasintercept
        m = mean(y, weights)
        return sum((y[i] - m)^2 * weights[i] for i in eachindex(y))
    else
        return sum(y[i]^2 * weights[i] for i in eachindex(y))
    end
end

function Fstat(coef::Vector{Float64}, matrix_vcov::AbstractMatrix{Float64}, has_intercept::Bool)
    coefF = copy(coef)
    # TODO: check I can't do better
    length(coef) == has_intercept && return NaN
    if has_intercept
        coefF = coefF[2:end]
        matrix_vcov = matrix_vcov[2:end, 2:end]
    end
    return (coefF' * (matrix_vcov \ coefF)) / length(coefF)
end


##############################################################################
##
## VcovData stores data you need to compute errors
##
##############################################################################
struct VcovData{T, N} <: RegressionModel
    modelmatrix::Matrix{Float64}       # X
    crossmodelmatrix::T                    # X'X in the simplest case. Can be Matrix but preferably Factorization
    residuals::Array{Float64, N}      # vector or matrix of residuals (matrix in the case of IV, residuals of Xendo on (Z, Xexo))
    dof_residual::Int
end
StatsBase.modelmatrix(x::VcovData) = x.modelmatrix
Vcov.crossmodelmatrix(x::VcovData) = x.crossmodelmatrix
StatsBase.residuals(x::VcovData) = x.residuals
StatsBase.dof_residual(x::VcovData) = x.dof_residual
