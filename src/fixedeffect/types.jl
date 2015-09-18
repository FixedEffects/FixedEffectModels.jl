
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

##############################################################################
##
## Build from DataFrame
##
##############################################################################

# Constructors from dataframe + expression
function FixedEffect(df::AbstractDataFrame, a::Expr, sqrtw::AbstractVector{Float64})
    if a.args[1] == :&
        id = convert(Symbol, "$(a.args[2])x$(a.args[3])")
        v1 = df[a.args[2]]
        v2 = df[a.args[3]]
        if isa(v1, PooledDataVector) && !isa(v2, PooledDataVector)
            f = v1
            x = convert(Vector{Float64}, v2)
            return FixedEffect(f.refs, length(f.pool), sqrtw, x, a.args[2], a.args[3], id)
        elseif isa(v2, PooledDataVector) && !isa(v1, PooledDataVector)
            f = v2
            x = convert(Vector{Float64}, v1)
            return FixedEffect(f.refs, length(f.pool), sqrtw, x, a.args[3], a.args[2], id)
        else
            v1 = pool(v1)
            x =  convert(Vector{Float64}, v2)
            return FixedEffect(v1.refs, length(v1.pool), sqrtw, x, a.args[3], a.args[2], id)
        end
    else
        error("Exp $(a) should be of the form factor&nonfactor")
    end
end

function FixedEffect(df::AbstractDataFrame, a::Symbol, sqrtw::AbstractVector{Float64})
    v = df[a]
    if !isa(v, PooledDataVector)
        v = pool(v)
    end
    return FixedEffect(v.refs, length(v.pool), sqrtw, Ones(length(v)), a, :none, a)
end

##############################################################################
## 
## FixedEffectVector 
##
## We need to define these methods used in lsmr! (duck typing)
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

getindex(fev::FixedEffectVector, i::Integer) = fev._[i]
length(fev::FixedEffectVector) = length(fev._)

function copy!(fev2::FixedEffectVector, fev1::FixedEffectVector)
    for i in 1:length(fev1)
        copy!(fev2[i], fev1[i])
    end
    return fev2
end

function axpy!(α::Float64, fev1::FixedEffectVector, fev2::FixedEffectVector)
    for i in 1:length(fev1)
        axpy!(α, fev1[i], fev2[i])
    end
    return fev2
end

function scale!(fev::FixedEffectVector, α::Float64)
    for i in 1:length(fev)
        scale!(fev[i], α)
    end
    return fev
end

function sumabs2(fev::FixedEffectVector)
    out = zero(Float64)
    for i in 1:length(fev)
        out += sumabs2(fev[i])
    end
    return out
end
norm(fev::FixedEffectVector) = sqrt(sumabs2(fev))

function fill!(fev::FixedEffectVector, x)
    for i in 1:length(fev)
        fill!(fev[i], x)
    end
end

##############################################################################
## 
## FixedEffectMatrix
##
## A is the model matrix of categorical variables
## normalized by diag(1/a1, ..., 1/aN) (Jacobi preconditoner)
##
##############################################################################

type FixedEffectMatrix
    _::Vector{FixedEffect}
end
getindex(fem::FixedEffectMatrix, i::Integer) = fem._[i]

# Define x -> A * x
function A_mul_B_helper!{R, W, I}(y::AbstractVector{Float64}, 
                                  fe::FixedEffect{R, W, I}, x::Vector{Float64})
    @inbounds @simd for i in 1:length(y)
        y[i] += x[fe.refs[i]] * fe.scale[fe.refs[i]] * fe.interaction[i] * fe.sqrtw[i]
    end
end
function A_mul_B!(y::AbstractVector{Float64}, fem::FixedEffectMatrix, 
                  fev::FixedEffectVector)
    fill!(y, zero(Float64))
    for i in 1:length(fev)
        A_mul_B_helper!(y, fem[i], fev[i])
    end
    return y
end

# Define x -> A' * x
function Ac_mul_B_helper!{R, W, I}(x::Vector{Float64}, fe::FixedEffect{R, W, I}, 
                                    y::AbstractVector{Float64})
    fill!(x, zero(Float64))
    @inbounds @simd for i in 1:length(y)
        x[fe.refs[i]] += y[i] * fe.interaction[i] * fe.sqrtw[i]
    end
    @inbounds @simd for i in 1:length(x)
        x[i] *= fe.scale[i]
    end
end
function Ac_mul_B!(fev::FixedEffectVector, fem::FixedEffectMatrix, 
                   y::AbstractVector{Float64})
    for i in 1:length(fev)
        Ac_mul_B_helper!(fev[i], fem[i], y)
    end
    return fev
end
