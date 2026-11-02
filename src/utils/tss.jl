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
    length(coef) == has_intercept && return NaN, 0
    if has_intercept
        coefF = coefF[2:end]
        matrix_vcov = matrix_vcov[2:end, 2:end]
    end
    all(isfinite, matrix_vcov) || return NaN, length(coefF)
    # Test only the linearly independent coefficients when the matrix is singular, e.g. after
    # the Cameron, Gelbach & Miller adjustment or with fewer clusters than coefficients.
    scale = sqrt.(diag(matrix_vcov))
    scale[iszero.(scale)] .= 1
    correlation = matrix_vcov ./ (scale * scale')
    independent = basis!(Symmetric(Matrix(correlation)))
    rank_vcov = count(independent)
    rank_vcov == 0 && return NaN, 0
    try
        z = (coefF ./ scale)[independent]
        return (z' * (cholesky(Symmetric(correlation[independent, independent])) \ z)) / rank_vcov, rank_vcov
    catch
        @info "The variance-covariance matrix is not invertible. F-statistic not computed "
        return NaN, rank_vcov
    end
end
