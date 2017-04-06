##############################################################################
##
## Weight
## 
##############################################################################
type WeightFormula
    arg::Union{Symbol, Void}
end
macro weight()
    return WeightFormula(nothing)
end
macro weight(arg)
    return Expr(:call, :WeightFormula, Base.Meta.quot(arg))
end

function get_weight(df::AbstractDataFrame, esample, weightformula::WeightFormula) 
    if weightformula.arg == nothing
        Ones{Float64}(sum(esample))
    else
        out = df[esample, weightformula.arg]
        # there are no NA in it. DataVector to Vector
        out = convert(Vector{Float64}, out)
        map!(sqrt, out, out)
        return out
    end
end