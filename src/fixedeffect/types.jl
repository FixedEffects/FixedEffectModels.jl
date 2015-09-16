
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
    refs::Vector{R}, l::Int, sqrtw::AbstractVector{Float64}, 
    interaction::AbstractVector{Float64}, factorname::Symbol, 
    interactionname::Symbol, id::Symbol)
    scale = fill(zero(Float64), l)
    @inbounds @simd for i in 1:length(refs)
         scale[refs[i]] += abs2(interaction[i] * sqrtw[i])
    end
    @inbounds @simd for i in 1:l
        scale[i] = scale[i] != 0 ? (1.0 / sqrt(scale[i])) : zero(Float64)
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
        if isa(df[a.args[2]], PooledDataVector) && !isa(df[a.args[3]], PooledDataVector)
            f = df[a.args[2]]
            x = convert(Vector{Float64}, df[a.args[3]])
            return FixedEffect(f.refs, length(f.pool), sqrtw, x, a.args[2], a.args[3], id)
        elseif isa(df[a.args[3]], PooledDataVector) && !isa(df[a.args[2]], PooledDataVector)
            f = df[a.args[3]]
            x = convert(Vector{Float64}, df[a.args[2]])
            return FixedEffect(f.refs, length(f.pool), sqrtw, x, a.args[3], a.args[2], id)
        else
            error("Exp $(a) should be of the form factor&nonfactor")
        end
    else
        error("Exp $(a) should be of the form factor&nonfactor")
    end
end

function FixedEffect(df::AbstractDataFrame, a::Symbol, sqrtw::AbstractVector{Float64})
    v = df[a]
    if typeof(v) <: PooledDataVector
        return FixedEffect(v.refs, length(v.pool), sqrtw, Ones(length(v)), a, :none, a)
    else
        error("$(a) is not a pooled data array")
    end
end

##############################################################################
## 
## We know defined an FixedEffectVector and a FixedEffectMatrix
## which correspond respectibely to x and A in (A'A)X = A'y
## We need to define these methods used in cgls!
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

function fill!(fev::FixedEffectVector, x)
    for i in 1:length(fev)
        fill!(fev[i], x)
    end
end


function broadcast!(f, out::FixedEffectVector, fev1::FixedEffectVector, fev2::FixedEffectVector)
    for i in 1:length(fev1)
        broadcast!(f, out[i], fev1[i], fev2[i])
    end
end

function scale!(fev2::FixedEffectVector, fev1::FixedEffectVector, α::Float64)
    for i in 1:length(fev2)
        scale!(fev2[i], fev1[i], α)
    end
    return fev2
end

function dot(fev1::FixedEffectVector, fev2::FixedEffectVector)
    out = zero(Float64)
    for i in 1:length(fev2)
        out += dot(fev1[i], fev2[i])
    end
    return out
end



# Matrix
# A is the model matrix multiplied by diag(1/a1^2, ..., 1/aN^2) (preconditoner)
type FixedEffectMatrix <: AbstractMatrix{Float64}
    _::Vector{FixedEffect}
end

# Define x -> A * x
function A_mul_B_helper!{R, W, I}(y::AbstractVector{Float64}, 
                                  fe::FixedEffect{R, W, I}, x::Vector{Float64})
    @inbounds @simd for i in 1:length(y)
        y[i] += x[fe.refs[i]] * fe.interaction[i] * fe.sqrtw[i]
    end
end
function A_mul_B!(y::AbstractVector{Float64}, fem::FixedEffectMatrix, 
                  fev::FixedEffectVector)
    fill!(y, zero(Float64))
    fes = fem._
    for i in 1:length(fes)
        A_mul_B_helper!(y, fes[i], fev[i])
    end
    return y
end

# Define x -> A' * x
function Ac_mul_B_helper!{R, W, I}(x::Vector{Float64}, 
                                   fe::FixedEffect{R, W, I}, y::AbstractVector{Float64})
    fill!(x, zero(Float64))
    @inbounds @simd for i in 1:length(y)
        x[fe.refs[i]] += y[i] * fe.interaction[i] * fe.sqrtw[i]
    end
end
function Ac_mul_B!(fev::FixedEffectVector, fem::FixedEffectMatrix, 
                   y::AbstractVector{Float64})
    fes = fem._
    for i in 1:length(fes)
        Ac_mul_B_helper!(fev[i], fes[i], y)
    end
    return fev
end


function sumabs2!(fev::FixedEffectVector, fem::FixedEffectMatrix) 
    for i in 1:length(fev)
        copy!(fev[i], fem._[i].scale)
    end
end


##############################################################################
##
## FixedEffectProblem stores some arrays to solve (A'A)X = A'y multiple times
##
##############################################################################

type FixedEffectProblem
    m::FixedEffectMatrix
    q::Vector{Float64}
    normalization::FixedEffectVector
    s::FixedEffectVector
    p::FixedEffectVector
    z::FixedEffectVector
    ptmp::FixedEffectVector
end


function FixedEffectProblem(fes::Vector{FixedEffect})
    fem = FixedEffectMatrix(fes)
    q = Array(Float64, length(fes[1].refs))
    normalization = FixedEffectVector(fes)
    s = FixedEffectVector(fes)
    p = FixedEffectVector(fes)
    z = FixedEffectVector(fes)
    ptmp = FixedEffectVector(fes)
    FixedEffectProblem(fem, q, normalization, s, p, z, ptmp)
end

function cgls!(x, r, pfe::FixedEffectProblem; tol::Real=1e-8, maxiter::Integer=1000)
    cgls!(x, r, pfe.m, pfe.q, pfe.normalization, pfe.s, pfe.p, pfe.z, pfe.ptmp; tol = tol, maxiter = maxiter)
end

