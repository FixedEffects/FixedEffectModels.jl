##############################################################################
##
## FixedEffect
##
## The categoricalarray may have pools that are never referred. Note that the pool does not appear in FixedEffect anyway.
##
##############################################################################

struct FixedEffect{R <: Integer, W <: AbstractVector{Float64}, I <: AbstractVector{Float64}}
    refs::Vector{R}         # refs of the original CategoricalVector
    sqrtw::W                # weights
    scale::Vector{Float64}  # 1/(∑ sqrt(w) * interaction) within each group
    interaction::I          # the continuous interaction 
    factorname::Symbol      # Name of factor variable 
    interactionname::Union{Symbol, Nothing} # Name of continuous variable in the original dataframe
    id::Symbol              # Name of new variable if save = true
end


# Constructor
FixedEffect(x::Nothing, sqrtw::AbstractVector{Float64}) = nothing

# Constructors from dataframe + terms
function FixedEffect(df::AbstractDataFrame, feformula, sqrtw::AbstractVector{Float64})
    out = FixedEffect[]
    for term in feformula.terms
        result = FixedEffect(df, term, sqrtw)
        if result != nothing
            push!(out,  result)
        end
    end
    return out
end

# Constructors from dataframe + symbol
function FixedEffect(df::AbstractDataFrame, a::Symbol, sqrtw::AbstractVector{Float64})
    v = df[a]
    if isa(v, CategoricalVector)
        # x from x*id -> x + id + x&id
        return FixedEffect(v.refs, sqrtw, zeros(length(v.pool)), Ones{Float64}(length(v)), a, nothing, a)
    end
end

# Constructors from dataframe + expression
function FixedEffect(df::AbstractDataFrame, a::Expr, sqrtw::AbstractVector{Float64})
    _check(a) || throw("Expression $a should only contain & and variable names")
    factorvars, interactionvars = _split(df, allvars(a))
    if !isempty(factorvars)
        # x1&x2 from (x1&x2)*id
        z = group(df, factorvars)
        interaction = _multiply(df, interactionvars)
        factorname = _name(factorvars)
        interactionname = _name(interactionvars)
        id = _name(allvars(a))
        return FixedEffect(z.refs, sqrtw, zeros(length(z.pool)), interaction, factorname, interactionname, id)
    end
end


function _check(a::Expr)
    a.args[1] == :& && check(a.args[2]) && check(a.args[3])
end
check(a::Symbol) = true

function _name(s::Vector{Symbol})
    if isempty(s)
        out = nothing
    else
        out = Symbol(reduce((x1, x2) -> string(x1)*"x"*string(x2), s))
    end
    return out
end

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

##############################################################################
##
## Subset FixedEffect for esample
##
##############################################################################


function getindex(x::FixedEffect, idx)
    refs = x.refs[idx]
    sqrtw = x.sqrtw[idx]
    interaction = x.interaction[idx]

    scale = copy(x.scale)
    for i in 1:length(refs)
        scale[refs[i]] += abs2(interaction[i] * sqrtw[i])
    end
    for i in 1:length(scale)
        scale[i] = scale[i] > 0 ? (1.0 / sqrt(scale[i])) : 0.
    end
    FixedEffect(refs, sqrtw, scale, interaction, x.factorname, x.interactionname, x.id)
end


##############################################################################
##
## Remove singletons
##
##############################################################################
function remove_singletons!(esample, fixedeffects::Vector{FixedEffect})
    for f in fixedeffects
        remove_singletons!(esample, f.refs, zeros(Int, length(f.scale)))
    end
end

function remove_singletons!(esample, refs::Vector, cache::Vector{Int})
    for i in 1:length(esample)
        if esample[i]
            cache[refs[i]] += 1
        end
    end
    for i in 1:length(esample)
        if esample[i] && cache[refs[i]] <= 1
            esample[i] = false
        end
    end
end

