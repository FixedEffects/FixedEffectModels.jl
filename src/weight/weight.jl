##############################################################################
##
## Weight
## 
##############################################################################
function get_weight(df::AbstractDataFrame, esample::AbstractVector, weight::Symbol) 
    # there are no NA in it. DataVector to Vector
    out = convert(Vector{Float64}, df[esample, weight])
    map!(sqrt, out, out)
    return out
end
get_weight(df::AbstractDataFrame, esample::AbstractVector, ::Void) = Ones{Float64}(sum(esample))

