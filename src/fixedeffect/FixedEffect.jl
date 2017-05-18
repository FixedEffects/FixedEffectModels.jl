##############################################################################
##
## Fixed  Effect Formula
##
##############################################################################

struct FixedEffectFormula
    arg::Union{Symbol, Expr, Void}
end
allvars(feformula::FixedEffectFormula) = allvars(feformula.arg)
Terms(feformula::FixedEffectFormula) = Terms(Formula(nothing, feformula.arg))

##############################################################################
##
## FixedEffect
##
##############################################################################

struct FixedEffect{R <: Integer, W <: AbstractVector{Float64}, I <: AbstractVector{Float64}}
    refs::Vector{R}         # refs of the original PooledDataVector
    sqrtw::W                # weights
    scale::Vector{Float64}  # 1/(∑ sqrt(w) * interaction) within each group
    interaction::I          # the continuous interaction 
    factorname::Symbol      # Name of factor variable 
    interactionname::Symbol # Name of continuous variable in the original dataframe
    id::Symbol              # Name of new variable if save = true
end

# Constructor
function FixedEffect{R <: Integer}(
    refs::Vector{R}, l::Integer, sqrtw::AbstractVector{Float64}, 
    interaction::AbstractVector{Float64}, factorname::Symbol, 
    interactionname::Symbol, id::Symbol)
    scale = zeros(Float64, l)
    @inbounds @simd for i in 1:length(refs)
         scale[refs[i]] += abs2(interaction[i] * sqrtw[i])
    end
    @inbounds @simd for i in 1:l
           scale[i] = scale[i] > 0 ? (1.0 / sqrt(scale[i])) : 0.
       end
    FixedEffect(refs, sqrtw, scale, interaction, factorname, interactionname, id)
end

# Constructors from dataframe + terms
function FixedEffect(df::AbstractDataFrame, feformula::FixedEffectFormula, sqrtw::AbstractVector{Float64})
    out = FixedEffect[]
    for term in Terms(feformula).terms
        result = FixedEffect(df, term, sqrtw)
        if isa(result, FixedEffect)
            push!(out, result)
        elseif isa(result, Vector{FixedEffect})
            append!(out, result)
        end
    end
    return out
end

# Constructors from dataframe + symbol
function FixedEffect(df::AbstractDataFrame, a::Symbol, sqrtw::AbstractVector{Float64})
    v = df[a]
    if isa(v, PooledDataVector)
        return FixedEffect(v.refs, length(v.pool), sqrtw, Ones{Float64}(length(v)), a, :none, a)
    else
        # x from x*id -> x + id + x&id
        return nothing
    end
end

# Constructors from dataframe + expression
function FixedEffect(df::AbstractDataFrame, a::Expr, sqrtw::AbstractVector{Float64})
    _check(a) || throw("Expression $a should only contain & and variable names")
    factorvars, interactionvars = _split(df, allvars(a))
    if isempty(factorvars)
        # x1&x2 from (x1&x2)*id
        return nothing
    end
    z = group(df, factorvars)
    interaction = _multiply(df, interactionvars)
    factorname = _name(factorvars)
    interactionname = _name(interactionvars)
    id = _name(allvars(a))
    l = length(z.pool)
    return FixedEffect(z.refs, l, sqrtw, interaction, factorname, interactionname, id)
end


function _check(a::Expr)
    a.args[1] == :& && check(a.args[2]) && check(a.args[3])
end
check(a::Symbol) = true

function _name(s::Vector{Symbol})
    if isempty(s)
        out = :none
    else
        out = convert(Symbol, reduce((x1, x2) -> string(x1)*"x"*string(x2), s))
    end
    return out
end

function _split(df::AbstractDataFrame, ss::Vector{Symbol})
    catvars, contvars = Symbol[], Symbol[]
    for s in ss
        isa(df[s], PooledDataVector) ? push!(catvars, s) : push!(contvars, s)
    end
    return catvars, contvars
end

function _multiply(df, ss::Vector{Symbol})
    if isempty(ss)
        out = Ones(size(df, 1))
    else
        if isa(df[ss[1]], Vector{Float64})
            out = deepcopy(df[ss[1]])
        else
            out = convert(Vector{Float64}, df[ss[1]])
        end
        for i in 2:length(ss)
            broadcast!(*, out, out, df[ss[i]])
        end
    end
    return out
end

