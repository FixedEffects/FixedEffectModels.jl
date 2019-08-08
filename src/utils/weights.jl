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
isnaorneg(a::AbstractVector) = BitArray(!ismissing(x) & (x > 0) for x in a)
