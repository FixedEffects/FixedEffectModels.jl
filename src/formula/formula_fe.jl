##############################################################################
##
## Parse FixedEffect
##
##
##############################################################################

function parse_fixedeffect(df::AbstractDataFrame, feformula)
    fe = FixedEffect[]
    id = Symbol[]
    for term in feformula.terms
        result = parse_fixedeffect(df, term)
        if result != nothing
            push!(fe, result[1])
            push!(id, result[2])
        end
    end
    return fe, id
end

# Constructor
parse_fixedeffect(x::Nothing) = nothing

# Constructors from dataframe + symbol
function parse_fixedeffect(df::AbstractDataFrame, a::Symbol)
    v = df[a]
    if isa(v, CategoricalVector)
        # x from x*id -> x + id + x&id
        return FixedEffect(v), a
    end
end

# Constructors from dataframe + expression
function parse_fixedeffect(df::AbstractDataFrame, a::Expr)
    _check(a) || throw("Expression $a should only contain & and variable names")
    factorvars, interactionvars = _split(df, allvars(a))
    if !isempty(factorvars)
        # x1&x2 from (x1&x2)*id
        fe = FixedEffect((df[v] for v in factorvars)...; interaction = _multiply(df, interactionvars))
        id = _name(allvars(a))
        return fe, id
    end
end


function _check(a::Expr)
    a.args[1] == :& && check(a.args[2]) && check(a.args[3])
end
check(a::Symbol) = true


function _split(df::AbstractDataFrame, ss::Vector{Symbol})
    catvars, contvars = Symbol[], Symbol[]
    for s in ss
        isa(df[s], CategoricalVector) ? push!(catvars, s) : push!(contvars, s)
    end
    return catvars, contvars
end

function _multiply(df, ss::Vector{Symbol})
    if isempty(ss)
        out = Ones(size(df, 1))
    else
        out = ones(size(df, 1))
        for j in 1:length(ss)
            _multiply!(out, df[ss[j]])
        end
    end
    return out
end

function _multiply!(out, v)
    for i in 1:length(out)
        if v[i] === missing
            # may be missing when I remove singletons
            out[i] = 0.0
        else
            out[i] = out[i] * v[i]
        end
    end
end

function _name(s::Vector{Symbol})
    if isempty(s)
        out = nothing
    else
        out = Symbol(reduce((x1, x2) -> string(x1)*"x"*string(x2), s))
    end
    return out
end



