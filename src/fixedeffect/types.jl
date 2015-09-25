
##############################################################################
##
## FixedEffect
##
##############################################################################

type FixedEffect{R <: Integer, W <: AbstractVector{Float64}, I <: AbstractVector{Float64}}
    refs::Vector{R}         # refs of the original PooledDataVector
    sqrtw::W                # weights
    scale::Vector{Float64}  # ∑ w * interaction^2) within each group
    interaction::I          # the continuous interaction 
    factorname::Symbol      # Name of factor variable 
    interactionname::Symbol # Name of continuous variable in the original dataframe
    id::Symbol              # Name of new variable if save = true
end



##############################################################################
##
## Constructor
##
##############################################################################
# Constructor
function FixedEffect{R <: Integer}(
    refs::Vector{R}, l::Integer, sqrtw::AbstractVector{Float64}, 
    interaction::AbstractVector{Float64}, factorname::Symbol, 
    interactionname::Symbol, id::Symbol)
    scale = fill(zero(Float64), l)
    @inbounds @simd for i in 1:length(refs)
         scale[refs[i]] += abs2(interaction[i] * sqrtw[i])
    end
    FixedEffect(refs, sqrtw, scale, interaction, factorname, interactionname, id)
end

# Constructors from dataframe + terms
function FixedEffect(df::AbstractDataFrame, terms::Terms, sqrtw::AbstractVector{Float64})
    out = FixedEffect[]
    for term in terms.terms
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
        return FixedEffect(v.refs, length(v.pool), sqrtw, Ones(length(v)), a, :none, a)
    else
        # x from x*id -> x + id + x&id
        return nothing
    end
end

# Constructors from dataframe + expression
function FixedEffect(df::AbstractDataFrame, a::Expr, sqrtw::AbstractVector{Float64})
    _check(a) || throw("Expression $a shouyld only contain & and variable names")
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

##############################################################################
## 
## FixedEffectVector 
##
## We define these methods used in lsmr! (duck typing): 
## copy!, fill!, scale!, axpy!, norm
##
##############################################################################

# Vector in the space of solutions (vector x in A'Ax = A'b)
type FixedEffectVector
    _::Vector{Vector{Float64}}
end

function FixedEffectVector(fes::Vector{FixedEffect})
    out = Vector{Float64}[]
    for fe in fes
        push!(out, similar(fe.scale))
    end
    return FixedEffectVector(out)
end

function copy!(fev2::FixedEffectVector, fev1::FixedEffectVector)
    for i in 1:length(fev1._)
        copy!(fev2._[i], fev1._[i])
    end
    return fev2
end

function fill!(fev::FixedEffectVector, x)
    for i in 1:length(fev._)
        fill!(fev._[i], x)
    end
end

function scale!(fev::FixedEffectVector, α::Number)
    for i in 1:length(fev._)
        scale!(fev._[i], α)
    end
    return fev
end

function axpy!(α::Number, fev1::FixedEffectVector, fev2::FixedEffectVector)
    for i in 1:length(fev1._)
        axpy!(α, fev1._[i], fev2._[i])
    end
    return fev2
end

function norm(fev::FixedEffectVector)
    out = zero(Float64)
    for i in 1:length(fev._)
        out += sumabs2(fev._[i])
    end
    return sqrt(out)
end

function map!(f, out::FixedEffectVector,  fev::FixedEffectVector...)
    for i in 1:length(out._)
        map!(f, out._[i], map(x -> x._[i], fev)...)
    end
    return out
end

##############################################################################
## 
## FixedEffectMatrix
##
## A is the model matrix of categorical variables
## normalized by diag(1/a1, ..., 1/aN) (Jacobi preconditoner)
##
## We define these methods used in lsmr! (duck typing):
## A_mul_B!(α, A, b, β, c) updates c -> α Ab + βc
## Ac_mul_B!(α, A, b, β, c) updates c -> α A'b + βc
##############################################################################

type FixedEffectMatrix
    _::Vector{FixedEffect}
end

# Define x -> A * x
function A_mul_B_helper!(α::Number, fe::FixedEffect, 
                        x::Vector{Float64}, y::AbstractVector{Float64})
    @inbounds @simd for i in 1:length(y)
        y[i] += α * x[fe.refs[i]] * fe.interaction[i] * fe.sqrtw[i]
    end
end
function A_mul_B!(α::Number, fem::FixedEffectMatrix, fev::FixedEffectVector, 
                β::Number, y::AbstractVector{Float64})
    if β == 0.0
        fill!(y, zero(Float64))
    elseif β != 1.0
        scale!(y, β)
    end
    for i in 1:length(fev._)
        A_mul_B_helper!(α, fem._[i], fev._[i], y)
    end
    return y
end

# Define x -> A' * x
function Ac_mul_B_helper!(α::Number, fe::FixedEffect, 
                        y::AbstractVector{Float64}, x::Vector{Float64})
    @inbounds @simd for i in 1:length(y)
        x[fe.refs[i]] += α * y[i] * fe.interaction[i] * fe.sqrtw[i]
    end
end
function Ac_mul_B!(α::Number, fem::FixedEffectMatrix, 
                y::AbstractVector{Float64}, β::Number, fev::FixedEffectVector)
    if β == 0.0
        fill!(fev, zero(Float64))
    elseif β != 1.0
        scale!(fev, β)
    end
    for i in 1:length(fev._)
        Ac_mul_B_helper!(α, fem._[i], y, fev._[i])
    end
    return fev
end

# define diag(A'A)
function colsumabs2!(fev::FixedEffectVector, fem::FixedEffectMatrix)
    for i in 1:length(fev._)
        copy!(fev._[i], fem._[i].scale)
    end
end
