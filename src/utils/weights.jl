##############################################################################
##
## Weight
## 
##############################################################################
function get_weights(df::AbstractDataFrame, esample::AbstractVector, weights::Symbol) 
    # there are no NA in it. DataVector to Vector
    out = convert(Vector{Float64}, df[esample, weights])
    map!(sqrt, out, out)
    return out
end
get_weights(df::AbstractDataFrame, esample::AbstractVector, ::Nothing) = Ones{Float64}(sum(esample))

#  remove observations with missing or negative weights
function isnaorneg(a::Vector{T}) where {T}
    out = BitArray(undef, length(a))
    @inbounds @simd for i in 1:length(a)
        out[i] = !ismissing(a[i]) & (a[i] > zero(T))
    end
    return out
end