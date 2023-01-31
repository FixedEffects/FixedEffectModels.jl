##############################################################################
##
## Iterate on terms
##
##############################################################################

eachterm(@nospecialize(x::AbstractTerm)) = (x,)
eachterm(@nospecialize(x::NTuple{N, AbstractTerm})) where {N} = x
# copied from StatsModels
const TermOrTerms = Union{AbstractTerm, Tuple{AbstractTerm, Vararg{AbstractTerm}}}
const TupleTerm = Tuple{TermOrTerms, Vararg{TermOrTerms}}
hasintercept(f::FormulaTerm) = hasintercept(f.rhs)
hasintercept(t::AbstractTerm) = t == InterceptTerm{true}() || t == ConstantTerm(1)
hasintercept(t::TupleTerm) = any(hasintercept, t)
hasintercept(t::MatrixTerm) = hasintercept(t.terms)
omitsintercept(f::FormulaTerm) = omitsintercept(f.rhs)
omitsintercept(t::AbstractTerm) =
    t == InterceptTerm{false}() ||
    t == ConstantTerm(0) ||
    t == ConstantTerm(-1)
omitsintercept(t::TupleTerm) = any(omitsintercept, t)
omitsintercept(t::MatrixTerm) = omitsintercept(t.terms)
##############################################################################
##
## Parse IV
##
##############################################################################

function parse_iv(@nospecialize(f::FormulaTerm))
	for term in eachterm(f.rhs)
		if term isa FormulaTerm
            both = intersect(eachterm(term.lhs), eachterm(term.rhs))
            endos = setdiff(eachterm(term.lhs), both)
            exos = setdiff(eachterm(term.rhs), both)
            # otherwise empty collection. Later, there will be check
            isempty(endos) && throw("There are no endogeneous variables")
            length(exos) < length(endos) && throw("Model not identified. There must be at least as many instrumental variables as endogeneneous variables")
			formula_endo = FormulaTerm(ConstantTerm(0), tuple(ConstantTerm(0), endos...))
			formula_iv = FormulaTerm(ConstantTerm(0), tuple(ConstantTerm(0), exos...))
            formula_exo = FormulaTerm(f.lhs, tuple((term for term in eachterm(f.rhs) if !isa(term, FormulaTerm))..., both...))
            return formula_exo, formula_endo, formula_iv
		end
	end
	return f, nothing, nothing
end

##############################################################################
##
## Parse FixedEffect
##
##############################################################################
struct FixedEffectTerm <: AbstractTerm
    x::Symbol
end
StatsModels.termvars(t::FixedEffectTerm) = [t.x]
fe(x::Term) = FixedEffectTerm(Symbol(x))
fe(s::Symbol) = FixedEffectTerm(s)

has_fe(::FixedEffectTerm) = true
has_fe(::FunctionTerm{typeof(fe)}) = true
has_fe(t::InteractionTerm) = any(has_fe(x) for x in t.terms)
has_fe(::AbstractTerm) = false
has_fe(@nospecialize(t::FormulaTerm)) = any(has_fe(x) for x in eachterm(t.rhs))


fesymbol(t::FixedEffectTerm) = t.x
fesymbol(t::FunctionTerm{typeof(fe)}) = Symbol(t.args[1])

"""
    parse_fixedeffect(data, formula::FormulaTerm)
    parse_fixedeffect(data, ts::NTuple{N, AbstractTerm})

Construct any `FixedEffect` specified with a `FixedEffectTerm`.

# Returns
- `Vector{FixedEffect}`: a collection of all `FixedEffect`s constructed.
- `Vector{Symbol}`: names assigned to the fixed effect estimates (can be used as column names).
- `Vector{Symbol}`: names of original fixed effects.
- `FormulaTerm` or `NTuple{N, AbstractTerm}`: `formula` or `ts` without any term related to fixed effects (an intercept may be explicitly omitted if necessary).
"""
function parse_fixedeffect(data, @nospecialize(formula::FormulaTerm))
    fes = FixedEffect[]
    ids = Symbol[]
    fekeys = Symbol[]
     for term in eachterm(formula.rhs)
        result = _parse_fixedeffect(data, term)
        if result !== nothing
            push!(fes, result[1])
            push!(ids, result[2])
            append!(fekeys, result[3])
        end
    end
    if !isempty(fes)
        if any(fe.interaction isa UnitWeights for fe in fes)
            formula = FormulaTerm(formula.lhs, (InterceptTerm{false}(), (term for term in eachterm(formula.rhs) if !isa(term, Union{ConstantTerm,InterceptTerm}) && !has_fe(term))...))
        else
            formula = FormulaTerm(formula.lhs, Tuple(term for term in eachterm(formula.rhs) if !has_fe(term)))
        end
    end
    return fes, ids, unique(fekeys), formula
end

# Method for external packages
function parse_fixedeffect(data, @nospecialize(ts::NTuple{N, AbstractTerm})) where N
    fes = FixedEffect[]
    ids = Symbol[]
    fekeys = Symbol[]
    for term in eachterm(ts)
        result = _parse_fixedeffect(data, term)
        if result !== nothing
            push!(fes, result[1])
            push!(ids, result[2])
            append!(fekeys, result[3])
        end
    end
    if !isempty(fes)
        if any(fe.interaction isa UnitWeights for fe in fes)
            ts = (InterceptTerm{false}(), (term for term in eachterm(ts) if !isa(term, Union{ConstantTerm,InterceptTerm}) && !has_fe(term))...)
        else
            ts = Tuple(term for term in eachterm(ts) if !has_fe(term))
        end
    end
    return fes, ids, unique(fekeys), ts
end

# Construct FixedEffect from a generic term
function _parse_fixedeffect(data, @nospecialize(t::AbstractTerm))
    if has_fe(t)
        st = fesymbol(t)
        return FixedEffect(Tables.getcolumn(data, st)), Symbol(:fe_, st), [st]
    end
end

# Construct FixedEffect from an InteractionTerm
function _parse_fixedeffect(data, @nospecialize(t::InteractionTerm))
    fes = (x for x in t.terms if has_fe(x))
    interactions = (x for x in t.terms if !has_fe(x))
    if !isempty(fes)
        # x1&x2 from (x1&x2)*id
        fe_names = [fesymbol(x) for x in fes]
        v1 = _multiply(data, Symbol.(interactions))
        fe = FixedEffect((Tables.getcolumn(data, fe_name) for fe_name in fe_names)...; interaction = v1)
        interactions = string.(interactions)
        s = vcat(["fe_" * string(fe_name) for fe_name in fe_names], interactions)
        return fe, Symbol(reduce((x1, x2) -> x1*"&"*x2, s)), fe_names
    end
end

function _multiply(data, ss::AbstractVector)
    if isempty(ss)
        return uweights(size(data, 1))
    elseif length(ss) == 1
        # in case it has missing (for some reason *(missing) not defined) iv VERSION < 1.6
        # do NOT use ! since it would modify the vector
        return convert(AbstractVector{Float64}, replace(Tables.getcolumn(data, ss[1]), missing => 0))
    else
        return convert(AbstractVector{Float64}, replace!(.*((Tables.getcolumn(data, x) for x in ss)...), missing => 0))
    end
end