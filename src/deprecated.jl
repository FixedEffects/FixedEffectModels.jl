
  
##############################################################################
##
## Old one
##
##
##############################################################################

function oldparse_fixedeffect(df::AbstractDataFrame, feformula::FormulaTerm)
    fe = FixedEffect[]
    id = Symbol[]
    for term in eachterm(feformula.rhs)
        result = oldparse_fixedeffect(df, term, feformula)
        if result != nothing
            push!(fe, result[1])
            push!(id, result[2])
        end
    end
    return fe, id
end

# Constructors from dataframe + Term
function oldparse_fixedeffect(df::AbstractDataFrame, a::Term, feformula::FormulaTerm)
    v = df[!, Symbol(a)]
    if isa(v, CategoricalVector)
        return FixedEffect(v), Symbol(a)
    else
        # x from x*id -> x + id + x&id
        if !any(isa(term, InteractionTerm) & (a ∈ terms(term)) for term in eachterm(feformula.rhs))
               error("The term $(a) in fe= is a continuous variable. Convert it to a categorical variable using 'categorical'.")
        end
    end
end

# Constructors from dataframe + InteractionTerm
function oldparse_fixedeffect(df::AbstractDataFrame, a::InteractionTerm, feformula::FormulaTerm)
    factorvars, interactionvars = _split(df, a)
    if !isempty(factorvars)
        # x1&x2 from (x1&x2)*id
        fe = FixedEffect((df[!, v] for v in factorvars)...; interaction = old_multiply(df, interactionvars))
        id = old_name(Symbol.(terms(a)))
        return fe, id
    end
end

function _split(df::AbstractDataFrame, a::InteractionTerm)
    factorvars, interactionvars = Symbol[], Symbol[]
    for s in terms(a)
        s = Symbol(s)
        isa(df[!, s], CategoricalVector) ? push!(factorvars, s) : push!(interactionvars, s)
    end
    return factorvars, interactionvars
end

function old_multiply(df, ss::Vector{Symbol})
    if isempty(ss)
        out = Ones(size(df, 1))
    else
        out = ones(size(df, 1))
        for j in eachindex(ss)
            old_multiply!(out, df[!, ss[j]])
        end
    end
    return out
end

function old_multiply!(out, v)
    for i in eachindex(out)
        if v[i] === missing
            # may be missing when I remove singletons
            out[i] = 0.0
        else
            out[i] = out[i] * v[i]
        end
    end
end

function old_name(s::Vector{Symbol})
    if isempty(s)
        out = nothing
    else
        out = Symbol(reduce((x1, x2) -> string(x1)*"x"*string(x2), s))
    end
    return out
end


struct ModelTerm
    f::FormulaTerm
    dict::Dict{Symbol, Any}
end

ModelTerm(f::FormulaTerm; kwargs...) = ModelTerm(f, Dict(pairs(kwargs)...))
function Base.show(io::IO, m::ModelTerm)
    println(io, m.f)
    for (k, v) in m.dict
        println(io, k, ": ", v)
    end
end

import StatsModels: capture_call
macro model(ex, kws...)
    @warn "@model is deprecated, please use @formula"
    f = StatsModels.terms!(StatsModels.sort_terms!(StatsModels.parse!(ex)))
    d = Dict{Symbol, Any}()
    for kw in kws
       isa(kw, Expr) &&  kw.head== :(=) || throw("All arguments of @model, except the first one, should be keyboard arguments")
       if kw.args[1] == :fe
            @warn "The keyword argument fe is deprecated. Instead of @model(y ~ x, fe = state + year),  write @formula(y ~ x + fe(state) + fe(year))"
            d[:feformula] = kw.args[2]
        elseif kw.args[1] == :ife
                 @warn "The keyword argument ife is deprecated. Instead of @model(y ~ x, ife = (state + year, 2)),  write @formula(y ~ x + ife(state, year, 2))"
            d[:ifeformula] = kw.args[2]
        elseif kw.args[1] == :vcov
            d[:vcovformula] = kw.args[2] 
            @warn "The keyword argument vcov is deprecated. Instead of reg(df, @model(y ~ x, vcov = cluster(State))),  write reg(df, @formula(y ~ x), Vcov.cluster(:State))"
        elseif kw.args[1] == :subset
            d[:subsetformula] = kw.args[2] 
            @warn "The keyword argument subset is deprecated. Instead of reg(df, @model(y ~ x, subset = State .>= 30),  write reg(df, @formula(y ~ x), subset = df.State .>= 30))"
        elseif kw.args[1] == :weight
                 d[:weight] = kw.args[2] 
                 @warn "The keyword argument weight is deprecated. Instead of reg(df, @model(y ~ x, weight = Pop),  write reg(df, @formula(y ~ x), weight = :Pop)"
        else
           d[kw.args[1]] = kw.args[2]
       end
    end
    :(ModelTerm($f, $d))
end


function evaluate_subset(df, ex::Expr)
    if ex.head == :call
        return Expr(ex.head, ex.args[1], (evaluate_subset(df, ex.args[i]) for i in 2:length(ex.args))...)
    else
        return Expr(ex.head, (evaluate_subset(df, ex.args[i]) for i in 1:length(ex.args))...)
    end
end
evaluate_subset(df, ex::Symbol) = df[!, ex]
evaluate_subset(df, ex)  = ex


function reg(df, m::ModelTerm;kwargs...)
    reg(df, m.f; m.dict..., kwargs...)
end

function partial_out(df, m::ModelTerm; kwargs...)
    partial_out(DataFrame(df), m.f; m.dict..., kwargs...)
end


function fes(args...)
    @warn "fes() is deprecated. Use fe()"
    fe(args...)
end

