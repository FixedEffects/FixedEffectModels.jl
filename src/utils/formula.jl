##############################################################################
##
## Iterate on terms
##
##############################################################################

eachterm(@nospecialize(x::AbstractTerm)) = (x,)
eachterm(@nospecialize(x::NTuple{N, AbstractTerm})) where {N} = x
TermOrTerms = Union{AbstractTerm, NTuple{N, AbstractTerm} where N}
hasintercept(@nospecialize(t::TermOrTerms)) =
    InterceptTerm{true}() ∈ terms(t) ||
    ConstantTerm(1) ∈ terms(t)
omitsintercept(@nospecialize(f::FormulaTerm)) = omitsintercept(f.rhs)
omitsintercept(@nospecialize(t::TermOrTerms)) =
    InterceptTerm{false}() ∈ terms(t) ||
    ConstantTerm(0) ∈ terms(t) ||
    ConstantTerm(-1) ∈ terms(t)
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
            !isempty(endos) && !isempty(exos) || throw("Model not identified. There must be at least as many ivs as endogeneneous variables")
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
fesymbol(t::FunctionTerm{typeof(fe)}) = Symbol(t.args_parsed[1])


function parse_fixedeffect(df::AbstractDataFrame, @nospecialize(formula::FormulaTerm))
    fes = FixedEffect[]
    ids = Symbol[]
    for term in eachterm(formula.rhs)
        result = parse_fixedeffect(df, term)
        if result != nothing
            push!(fes, result[1])
            push!(ids, result[2])
        end
    end
    if !isempty(fes)
        if any(fe.interaction isa UnitWeights for fe in fes)
            formula = FormulaTerm(formula.lhs, tuple(InterceptTerm{false}(), (term for term in eachterm(formula.rhs) if (term != ConstantTerm(1)) & (term != InterceptTerm{true}()) & !has_fe(term))...))
        else
            formula = FormulaTerm(formula.lhs, Tuple(term for term in eachterm(formula.rhs) if !has_fe(term)))
        end
    end
    return fes, ids, formula
end

# Constructors from dataframe + Term
function parse_fixedeffect(df::AbstractDataFrame, t::AbstractTerm)
    if has_fe(t)
        st = fesymbol(t)
        return FixedEffect(df[!, st]), Symbol(:fe_, st)
    end
end

# Constructors from dataframe + InteractionTerm
function parse_fixedeffect(df::AbstractDataFrame, t::InteractionTerm)
    fes = (x for x in t.terms if has_fe(x))
    interactions = (x for x in t.terms if !has_fe(x))
    if !isempty(fes)
        # x1&x2 from (x1&x2)*id
        fe_names = [fesymbol(x) for x in fes]
        v1 = _multiply(df, Symbol.(interactions))
        fe = FixedEffect((df[!, fe_name] for fe_name in fe_names)...; interaction = v1)
        interactions = setdiff(Symbol.(terms(t)), fe_names)
        s = vcat(["fe_" * string(fe_name) for fe_name in fe_names], string.(interactions))
        return fe, Symbol(reduce((x1, x2) -> x1*"&"*x2, s))
    end
end

function _multiply(df, ss::AbstractVector)
    if isempty(ss)
        return uweights(size(df, 1))
    elseif length(ss) == 1
        # in case it has missing (for some reason *(missing) not defined))
        # do NOT use ! since it would modify the vector
        return convert(AbstractVector{Float64}, replace(df[!, ss[1]], missing => 0))
    else
        return convert(AbstractVector{Float64}, replace!(.*((df[!, x] for x in ss)...), missing => 0))
    end
end