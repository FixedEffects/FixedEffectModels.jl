# Use to have generator, but I think this allows SIMD.
function tss(y::AbstractVector, hasintercept::Bool, weights::AbstractWeights)
    m = hasintercept ? mean(y, weights) : zero(Float64)
    out = zero(Float64)
    @inbounds @simd for i in eachindex(y)
        out += (y[i] - m)^2 * weights[i]
    end
    return out
end

function Fstat(coef::Vector{Float64}, matrix_vcov::AbstractMatrix{Float64}, has_intercept::Bool)
    coefF = copy(coef)
    # TODO: check I can't do better
    length(coef) == has_intercept && return NaN
    if has_intercept
        coefF = coefF[2:end]
        matrix_vcov = matrix_vcov[2:end, 2:end]
    end
    try
        return (coefF' * (cholesky(Symmetric(matrix_vcov)) \ coefF)) / length(coefF)
    catch
        @info "The variance-covariance matrix is not invertible. F-statistic not computed "
        return NaN
    end
end
