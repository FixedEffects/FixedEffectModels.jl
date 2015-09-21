
##############################################################################
##
## FixedEffect
##
##############################################################################

type FixedEffect{R <: Integer, W <: AbstractVector{Float64}, I <: AbstractVector{Float64}}
    refs::Vector{R}         # refs of the original PooledDataVector
    sqrtw::W                # weights
    scale::Vector{Float64}  # 1/(∑ sqrt(w) * interaction) within each group
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
    @inbounds @simd for i in 1:l
           scale[i] = scale[i] > 0 ? (1.0 / sqrt(scale[i])) : 0.
       end
    FixedEffect(refs, sqrtw, scale, interaction, factorname, interactionname, id)
end

# Constructors from dataframe + expression
function FixedEffect(df::AbstractDataFrame, a::Expr, sqrtw::AbstractVector{Float64})
    if a.args[1] == :& && isa(a.args[2], Symbol) && isa(a.args[3], Symbol)
        id = convert(Symbol, "$(a.args[2])x$(a.args[3])")
        v1 = df[a.args[2]]
        v2 = df[a.args[3]]
        if isa(v1, PooledDataVector) && !isa(v2, PooledDataVector)
            refs = v1.refs
            l = length(v1.pool)
            interaction = convert(Vector{Float64}, v2)
            factorname = a.args[2]
            interactionname = a.args[3]
        elseif isa(v2, PooledDataVector) && !isa(v1, PooledDataVector)
            refs = v2.refs
            l = length(v2.pool)
            interaction = convert(Vector{Float64}, v1)
            factorname = a.args[3]
            interactionname = a.args[2]
        else 
            error("Expression $(a) should be of the form factor&nonfactor or nonfactor&factor")
        end
        return FixedEffect(refs, l, sqrtw, interaction, factorname, interactionname, id)
    else
        error("Exp $(a) should be of the form var1&var2 or var1*var2")
    end
end

function FixedEffect(df::AbstractDataFrame, a::Symbol, sqrtw::AbstractVector{Float64})
    v = df[a]
    if !isa(v, PooledDataVector)
        v = pool(v)
    end
    return FixedEffect(v.refs, length(v.pool), sqrtw, Ones(length(v)), a, :none, a)
end


function FixedEffect(df::AbstractDataFrame, terms::Terms, sqrtw::AbstractVector{Float64})
    out = FixedEffect[]
    for term in terms.terms
        result = FixedEffect(df, term, sqrtw)
        if isa(result, FixedEffect)
            push!(out, result)
        else
            append!(out, result)
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

function scale!(fev::FixedEffectVector, α::Float64)
    for i in 1:length(fev._)
        scale!(fev._[i], α)
    end
    return fev
end

function axpy!(α::Float64, fev1::FixedEffectVector, fev2::FixedEffectVector)
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
function A_mul_B_helper!(α::Float64, fe::FixedEffect, 
                        x::Vector{Float64}, y::AbstractVector{Float64})
    @inbounds @simd for i in 1:length(y)
        y[i] += α * x[fe.refs[i]] * fe.scale[fe.refs[i]] * fe.interaction[i] * fe.sqrtw[i]
    end
end
function A_mul_B!(α::Float64, fem::FixedEffectMatrix, fev::FixedEffectVector, 
                β::Float64, y::AbstractVector{Float64})
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
function Ac_mul_B_helper!(α::Float64, fe::FixedEffect, 
                        y::AbstractVector{Float64}, x::Vector{Float64})
    @inbounds @simd for i in 1:length(y)
        x[fe.refs[i]] += α * y[i] * fe.scale[fe.refs[i]] * fe.interaction[i] * fe.sqrtw[i]
    end
end
function Ac_mul_B!(α::Float64, fem::FixedEffectMatrix, 
                y::AbstractVector{Float64}, β::Float64, fev::FixedEffectVector)
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
