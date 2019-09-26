function tss(y::AbstractVector, hasintercept::Bool, sqrtw::AbstractVector)
    if hasintercept
        m = (mean(y) / sum(sqrtw) * length(y))::Float64
        return sum(abs2(y[i] - sqrtw[i] * m) for i in eachindex(y))
    else
        return sum(abs2, y)
    end
end