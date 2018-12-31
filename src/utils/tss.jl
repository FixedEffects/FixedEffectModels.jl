function compute_tss(y::Vector{Float64}, hasintercept::Bool, sqrtw::AbstractVector)
    if hasintercept
        tss = zero(Float64)
        m = (mean(y) / sum(sqrtw) * length(y))::Float64
        @inbounds @simd for i in 1:length(y)
            tss += abs2(y[i] - sqrtw[i] * m)
        end
    else
        tss = sum(abs2, y)
    end
    return tss
end