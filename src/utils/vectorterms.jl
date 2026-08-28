#========================================================
Vector-based replicas of the StatsModels formula machinery, for latency.

StatsModels represents a formula right-hand side as a *tuple* of terms and the
data as a NamedTuple keyed by *column names*. Both encode information in the
type domain, so every new formula shape (number of terms) and every new set of
column names compiles fresh specializations of schema/apply_schema/modelcols —
roughly 0.2-0.3s of compilation per regression with a formula never seen
before, even when every individual term type has been precompiled.

The functions here operate on Vector{AbstractTerm} and look up columns by
runtime Symbol. Dispatch happens once per term *type* (ContinuousTerm,
CategoricalTerm, InteractionTerm, ...), so after the precompile workload a
regression with a brand-new formula runs entirely warm. The semantics
replicate the StatsModels tuple path exactly; in particular the stateful
FullRank contrast promotion is sequential over terms, so applying it in a
vector loop is equivalent to applying it to a tuple.
========================================================#

# FormulaTerm rhs (tuple of terms or a single term) -> Vector{AbstractTerm}
function termvector(@nospecialize(x))
    ts = AbstractTerm[]
    if x isa Tuple
        for t in x
            push!(ts, t)
        end
    else
        push!(ts, x)
    end
    return ts
end

# same definitions as StatsModels.hasintercept/omitsintercept, on a term vector
_hasintercept(ts::Vector{AbstractTerm}) =
    any(t -> t == InterceptTerm{true}() || t == ConstantTerm(1), ts)
_omitsintercept(ts::Vector{AbstractTerm}) =
    any(t -> t == InterceptTerm{false}() || t == ConstantTerm(0) || t == ConstantTerm(-1), ts)

function termvars_vector(ts::Vector{AbstractTerm})
    out = Symbol[]
    for t in ts
        append!(out, StatsModels.termvars(t))
    end
    return unique!(out)
end

# Split the first (endogenous ~ instruments) term out of rhs (mutating it into
# the exogenous part). Returns endo and iv term vectors, empty when there is no
# IV term. Variables appearing on both sides are treated as exogenous controls.
function parse_iv!(rhs::Vector{AbstractTerm})
    i = findfirst(t -> t isa FormulaTerm, rhs)
    i === nothing && return AbstractTerm[], AbstractTerm[]
    ivpart = rhs[i]::FormulaTerm
    endo_all = termvector(ivpart.lhs)
    iv_all = termvector(ivpart.rhs)
    both = intersect(endo_all, iv_all)
    endos = setdiff(endo_all, both)
    exos = setdiff(iv_all, both)
    isempty(endos) && throw(ArgumentError("There are no endogeneous variables"))
    length(exos) < length(endos) && throw(ArgumentError("Model not identified. There must be at least as many instrumental variables as endogeneneous variables"))
    filter!(t -> !isa(t, FormulaTerm), rhs)
    append!(rhs, both)
    # the leading ConstantTerm(0) suppresses the intercept when the schema is
    # applied, so categorical endogenous variables get full dummy coding
    return pushfirst!(endos, ConstantTerm(0)), pushfirst!(exos, ConstantTerm(0))
end

# Split fixed-effect terms out of rhs (mutating it). A main-effect term exactly
# spanned by a continuous-slope fixed effect is removed as well (see parse_fe).
function parse_fe!(rhs::Vector{AbstractTerm})
    fe_ts = AbstractTerm[t for t in rhs if has_fe(t)]
    isempty(fe_ts) && return fe_ts
    filter!(t -> !has_fe(t) && !any(fe_t -> _is_absorbed_fe_slope(t, fe_t), fe_ts), rhs)
    return fe_ts
end

function parse_fixedeffect(data, ts::Vector{AbstractTerm})
    fes = FixedEffect[]
    feids = Symbol[]
    fekeys = Symbol[]
    for t in ts
        result = _parse_fixedeffect(data, t)
        if result !== nothing
            push!(fes, result[1])
            push!(feids, result[2])
            append!(fekeys, result[3])
        end
    end
    return fes, feids, unique(fekeys)
end

getcol(data, s::Symbol) = Tables.getcolumn(data, s)

# name-free equivalent of schema(f, data, contrasts): one concrete term per
# variable, computed from the column looked up by runtime symbol
function build_schema(vars::Vector{Symbol}, data, contrasts::Dict)
    sch = StatsModels.Schema()
    for s in vars
        t = Term(s)
        sch.schema[t] = StatsModels.concrete_term(t, getcol(data, s), get(contrasts, s, nothing))
    end
    return sch
end

# equivalent of apply_schema(ts, FullRank(sch), Mod) on a tuple: FullRank
# promotion mutates its `already` set term by term, left to right
function apply_schema_vector(ts::Vector{AbstractTerm}, sch::StatsModels.Schema, Mod::Type,
                             already_intercept::Bool = false)
    fullrank = StatsModels.FullRank(sch)
    already_intercept && push!(fullrank.already, InterceptTerm{true}())
    out = AbstractTerm[]
    for t in ts
        applied = apply_schema(t, fullrank, Mod)
        if applied isa Tuple
            for a in applied
                push!(out, a)
            end
        else
            push!(out, applied)
        end
    end
    return out
end

# per-term model columns; block eltypes are converted when copied into the
# Float64 model matrix
modelcols_term(t::InterceptTerm{true}, data, n::Integer) = ones(n)
modelcols_term(t::InterceptTerm{false}, data, n::Integer) = Matrix{Float64}(undef, n, 0)
modelcols_term(t::ContinuousTerm, data, n::Integer) = getcol(data, t.sym)
modelcols_term(t::CategoricalTerm, data, n::Integer) = t.contrasts[getcol(data, t.sym), :]
modelcols_term(t::InteractionTerm, data, n::Integer) =
    StatsModels.row_kron_insideout(*, (modelcols_term(x, data, n) for x in t.terms)...)
modelcols_term(t::FunctionTerm, data, n::Integer) =
    t.f.((modelcols_term(x, data, n) for x in t.args)...)
# bare Term and ConstantTerm only appear inside (protected) FunctionTerm arguments
modelcols_term(t::Term, data, n::Integer) = getcol(data, t.sym)
modelcols_term(t::ConstantTerm, data, n::Integer) = t.n
# fallback for exotic term types: use the generic StatsModels path on the
# columns the term needs
function modelcols_term(@nospecialize(t::AbstractTerm), data, n::Integer)
    vars = StatsModels.termvars(t)
    nt = NamedTuple{(vars...,)}(((getcol(data, s) for s in vars)...,))
    return modelcols(t, nt)
end

function modelmatrix_vector(ts::Vector{AbstractTerm}, data, n::Integer)
    blocks = [modelcols_term(t, data, n) for t in ts if StatsModels.width(t) > 0]
    ncols = 0
    for b in blocks
        ncols += size(b, 2)
    end
    X = Matrix{Float64}(undef, n, ncols)
    j = 1
    for b in blocks
        k = size(b, 2)
        copyto!(view(X, :, j:(j + k - 1)), b)
        j += k
    end
    return X
end

function coefnames_vector(ts::Vector{AbstractTerm})
    out = String[]
    for t in ts
        StatsModels.width(t) > 0 || continue
        append!(out, StatsModels.vectorize(coefnames(t)))
    end
    return out
end
