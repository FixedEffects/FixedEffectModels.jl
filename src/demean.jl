
##############################################################################
##
## FixedEffect
##
##############################################################################
type FixedEffect{R <: Integer, W <: AbstractVector{Float64}, I <: AbstractVector{Float64}}
    refs::Vector{R}         # refs of the original PooledDataVector
    sqrtw::W                # weights
    scale::Vector{Float64}  # 1/(∑ w * interaction^2) within each group
    mean::Vector{Float64}
    interaction::I          # the continuous interaction 
    factorname::Symbol      # Name of factor variable 
    interactionname::Symbol # Name of continuous variable in the original dataframe
    id::Symbol              # Name of new variable if save = true
end

# Constructors the scale vector
function FixedEffect{R <: Integer}(
    refs::Vector{R}, l::Int, sqrtw::AbstractVector{Float64}, 
    interaction::AbstractVector{Float64}, factorname::Symbol, 
    interactionname::Symbol, id::Symbol)
    scale = fill(zero(Float64), l)
    @inbounds @simd for i in 1:length(refs)
         scale[refs[i]] += abs2((interaction[i] * sqrtw[i]))
    end
    @inbounds @simd for i in 1:l
        scale[i] = scale[i] > 0 ? (1.0 / scale[i]) : zero(Float64)
    end
    FixedEffect(refs, sqrtw, scale, similar(scale), interaction, factorname, interactionname, id)
end

# Constructors from dataframe + expression
function FixedEffect(df::AbstractDataFrame, a::Expr, sqrtw::AbstractVector{Float64})
    if a.args[1] == :&
        id = convert(Symbol, "$(a.args[2])x$(a.args[3])")
        if (typeof(df[a.args[2]]) <: PooledDataVector) && !(typeof(df[a.args[3]]) <: PooledDataVector)
            f = df[a.args[2]]
            x = convert(Vector{Float64}, df[a.args[3]])
            return FixedEffect(f.refs, length(f.pool), sqrtw, x, a.args[2], a.args[3], id)
        elseif (typeof(df[a.args[3]]) <: PooledDataVector) && !(typeof(df[a.args[2]]) <: PooledDataVector)
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
## FixedEffectMatrix
##
##############################################################################
type MatrixFixedEffect <: AbstractMatrix{Float64}
    _::Vector{FixedEffect}
end


function Base.A_mul_B!{R, W, I}(y::AbstractVector{Float64}, fe::FixedEffect{R, W, I})
    @inbounds @simd for i in 1:length(y)
        y[i] += fe.mean[fe.refs[i]] * fe.interaction[i] * fe.sqrtw[i]
    end
end

function Base.A_mul_B!(y::AbstractVector{Float64}, mfe::MatrixFixedEffect, x::AbstractVector{Float64})
    fill!(y, zero(Float64))
    fes = mfe._
    idx = 0
    for fe in fes
        @inbounds @simd for i in 1:length(fe.scale)
            idx += 1
            fe.mean[i] = x[idx] 
        end
    end
    for fe in fes
        A_mul_B!(y, fe)
    end
    return y
end

function Base.Ac_mul_B!{R, W, I}(fe::FixedEffect{R, W, I}, y::AbstractVector{Float64})
    fill!(fe.mean, zero(Float64))
    @inbounds @simd for i in 1:length(y)
        fe.mean[fe.refs[i]] += y[i] * fe.interaction[i] * fe.sqrtw[i]
    end
end

function Base.Ac_mul_B!(x::AbstractVector{Float64}, mfe::MatrixFixedEffect, y::AbstractVector{Float64})
    fes = mfe._
    for fe in fes
        Ac_mul_B!(fe, y)
    end
    idx = 0
    for fe in fes
        @inbounds @simd for i in 1:length(fe.scale)
            idx += 1
            x[idx] = fe.mean[i] 
        end
    end
    return x
end

Base.eltype(mfe::MatrixFixedEffect) = Float64

function Base.size(mfe::MatrixFixedEffect, i::Integer) 
    fes = mfe._
    if i == 1
        return length(fes[1].refs)
    elseif i == 2
        out = zero(Int)
        for fe in fes
            out += length(fe.scale)
        end
        return out
    end
end

##############################################################################
##
## FixedEffectProblem
##
##############################################################################

type ProblemFixedEffect <: AbstractMatrix{Float64}
    m::MatrixFixedEffect
    r::Vector{Float64}
    b::Vector{Float64}
    q::Vector{Float64}
    s::Vector{Float64}
    p::Vector{Float64}
end

function ProblemFixedEffect(m::MatrixFixedEffect)
    r = Array(Float64, size(m, 1))
    b = similar(r)
    q = similar(r)
    s = Array(Float64, size(m, 2))
    p = similar(s)
    ProblemFixedEffect(m, r, b, q, s, p)
end

function ProblemFixedEffect(fes::Vector{FixedEffect})
    ProblemFixedEffect(MatrixFixedEffect(fes))
end

function cgls!(x::Union{AbstractVector{Float64}, Nothing}, pfe::ProblemFixedEffect; tol::Real=1e-10, maxiter::Integer=1000)
    cgls!(x, pfe.r, pfe.m, pfe.b, pfe.s, pfe.p, pfe.q; tol = tol, maxiter = maxiter)
end



##############################################################################
##
## Dispatching accroding to typeof(x)
##
##############################################################################

function demean!(X::Matrix{Float64}, iterationsv::Vector{Int}, convergedv::Vector{Bool}, 
                 fes::Vector{FixedEffect}; maxiter::Int = 1000, tol::Float64 = 1e-8)
    pfe = ProblemFixedEffect(fes)
    for j in 1:size(X, 2)
        copy!(pfe.b, slice(X, :, j))
        iterations, converged = cgls!(nothing,  pfe; tol = tol, maxiter = maxiter)
        push!(iterationsv, iterations)
        push!(convergedv, converged)
        copy!(slice(X, :, j), pfe.r)
    end
end

function demean!(x::AbstractVector{Float64}, iterationsv::Vector{Int}, convergedv::Vector{Bool},fes::Vector{FixedEffect} ; maxiter::Int = 1000, tol::Float64 = 1e-8)
    pfe = ProblemFixedEffect(fes)
    copy!(pfe.b, x)
    iterations, converged = cgls!(nothing, pfe; tol = tol, maxiter = maxiter)
    copy!(x, pfe.r)
    push!(iterationsv, iterations)
    push!(convergedv, converged)
end

function demean(x::DataVector{Float64}, fes::Vector{FixedEffect}; 
                maxiter::Int = 1000, tol::Float64 = 1e-8)
    x = convert(Vector{Float64}, x)
    pfe = ProblemFixedEffect(fes)
    copy!(pfe.b, x)
    iterations, converged = clgs!(nothing, pfe; tol = tol, maxiter = maxiter)
    return pfe.r, iterations, converged
end

function demean!(::Array, ::Vector{Int}, ::Vector{Bool}, ::Nothing; 
                 maxiter::Int = 1000, tol::Float64 = 1e-8)
    nothing
end

function demean(x::DataVector{Float64},fes::Vector{FixedEffect}; 
                maxiter::Int = 1000, tol::Float64 = 1e-8)
    x = convert(Vector{Float64}, x)
    iterations = Int[]
    converged = Bool[]
    demean!(x, iterations, converged, fes, maxiter = maxiter, tol = tol)
    return x, iterations, converged
end

function demean!(::Array, ::Vector{Int}, ::Vector{Bool}, ::Nothing; 
                 maxiter::Int = 1000, tol::Float64 = 1e-8)
    nothing
end



