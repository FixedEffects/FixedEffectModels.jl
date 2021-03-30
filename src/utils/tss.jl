function tss(y::AbstractVector, hasintercept::Bool, weights::AbstractWeights)
    if hasintercept
        m = mean(y, weights)
        sum(@inbounds (y[i] - m)^2 * weights[i] for i in eachindex(y))
    else
        sum(@inbounds y[i]^2 * weights[i] for i in eachindex(y))
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
    try
        return (coefF' * (matrix_vcov \ coefF)) / length(coefF)
    catch
        @info "The variance-covariance matrix is not invertible. F-statistic not computed "
        return NaN
    end
end
