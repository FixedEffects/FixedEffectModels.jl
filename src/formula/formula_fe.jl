##############################################################################
##
## Parse FixedEffect
##
##
##############################################################################

fe(x) = nothing
has_fe(x::FunctionTerm{typeof(fe)}) = true
has_fe(x::InteractionTerm) = any(has_fe(x) for x in x.terms)
has_fe(x::AbstractTerm) = false
has_fe(x::FormulaTerm) = any(has_fe(term) for term in eachterm(x.rhs))



function parse_fixedeffect(df::AbstractDataFrame, formula::FormulaTerm)
    fe = FixedEffect[]
    id = Symbol[]
    for term in eachterm(formula.rhs)
        result = parse_fixedeffect(df, term)
        if result != nothing
            push!(fe, result[1])
            push!(id, result[2])
        end
    end
    formula = FormulaTerm(formula.lhs, tuple((term for term in eachterm(formula.rhs) if !has_fe(term))...))
    return fe, id, formula
end

# Constructors from dataframe + Term
function parse_fixedeffect(df::AbstractDataFrame, a::FunctionTerm{typeof(fe)})
    sa = Symbol(first(a.args_parsed))
    return FixedEffect(df[!, sa]), Symbol("fe(" * string(sa) * ")")
end

# Constructors from dataframe + InteractionTerm
function parse_fixedeffect(df::AbstractDataFrame, a::InteractionTerm)
    fes = (x for x in a.terms if has_fe(x))
    interactions = (x for x in a.terms if !has_fe(x))
    if !isempty(fes)
        # x1&x2 from (x1&x2)*id
        fe_names = [Symbol(first(x.args_parsed)) for x in fes]
        fe = FixedEffect((df[!, fe_name] for fe_name in fe_names)...; interaction = _multiply(df, Symbol.(interactions)))
        interactions = setdiff(Symbol.(terms(a)), fe_names)
        s = vcat(["fe(" * string(fe_name) * ")" for fe_name in fe_names], string.(interactions))
        return fe, Symbol(reduce((x1, x2) -> x1*"&"*x2, s))
    end
end

function parse_fixedeffect(df::AbstractDataFrame, a::AbstractTerm)
    nothing
end

function _multiply(df, ss::Vector)
    if isempty(ss)
        out = Ones(size(df, 1))
    else
        out = ones(size(df, 1))
        for j in eachindex(ss)
            _multiply!(out, df[!, ss[j]])
        end
    end
    return out
end
function _multiply!(out, v)
    for i in eachindex(out)
        if v[i] === missing
            # may be missing when I remove singletons
            out[i] = 0.0
        else
            out[i] = out[i] * v[i]
        end
    end
end


