
##############################################################################
##
## FixedEffect
##
##############################################################################

type FixedEffect{R <: Integer, W <: AbstractVector{Float64}, I <: AbstractVector{Float64}}
    refs::Vector{R}         # refs of the original PooledDataVector
    sqrtw::W                # weights
    scale::Vector{Float64}  # 1/(∑ sqrt(w) * interaction) within each group
    mean::Vector{Float64}
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
         scale[refs[i]] += abs2((interaction[i] * sqrtw[i]))
    end
    @inbounds @simd for i in 1:l
        scale[i] = scale[i] > 0 ? (1.0 / sqrt(scale[i])) : zero(Float64)
    end
    FixedEffect(refs, sqrtw, scale, similar(scale), interaction, factorname, interactionname, id)
end


##############################################################################
##
## Denote M model matrix of fixed effects
## Conjugate gradient will solve (A'A)X = A'y
## where A = M * diag(1/a_1, 1/a_2... , 1/a_n) 
## We need to define what it means to multiply by A and At 
##
##############################################################################

type FixedEffectModelMatrix <: AbstractMatrix{Float64}
    _::Vector{FixedEffect}
end

function size(mfe::FixedEffectModelMatrix, i::Integer) 
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

function _add!{R, W, I}(y::AbstractVector{Float64}, fe::FixedEffect{R, W, I})
    @inbounds @simd for i in 1:length(y)
        y[i] += fe.mean[fe.refs[i]] * fe.interaction[i] * fe.sqrtw[i]
    end
end

function A_mul_B!(y::AbstractVector{Float64}, mfe::FixedEffectModelMatrix, x::AbstractVector{Float64})
    fill!(y, zero(Float64))
    fes = mfe._
    idx = 0
    for fe in fes
        @inbounds @simd for i in 1:length(fe.scale)
            idx += 1
            fe.mean[i] = x[idx] * fe.scale[i] # so that A'A is close to identity
        end
    end
    for fe in fes
        _add!(y, fe)
    end
    return y
end

function _sum!{R, W, I}(fe::FixedEffect{R, W, I}, y::AbstractVector{Float64})
    fill!(fe.mean, zero(Float64))
    @inbounds @simd for i in 1:length(y)
        fe.mean[fe.refs[i]] += y[i] * fe.interaction[i] * fe.sqrtw[i]
    end
end

function Ac_mul_B!(x::AbstractVector{Float64}, mfe::FixedEffectModelMatrix, y::AbstractVector{Float64})
    fes = mfe._
    for fe in fes
        _sum!(fe, y)
    end
    idx = 0
    for fe in fes
        @inbounds @simd for i in 1:length(fe.scale)
            idx += 1
            x[idx] = fe.mean[i] * fe.scale[i] # so that A'A is close to identity
        end
    end
    return x
end

##############################################################################
##
## FixedEffectProblem stores some arrays to solve (A'A)X = A'y multiple times
##
##############################################################################

type FixedEffectProblem <: AbstractMatrix{Float64}
    m::FixedEffectModelMatrix
    q::Vector{Float64}
    s::Vector{Float64}
    p::Vector{Float64}
end

function FixedEffectProblem(m::FixedEffectModelMatrix)
    q = Array(Float64, size(m, 1))
    s = Array(Float64, size(m, 2))
    p = similar(s)
    FixedEffectProblem(m, q, s, p)
end

function FixedEffectProblem(fes::Vector{FixedEffect})
    FixedEffectProblem(FixedEffectModelMatrix(fes))
end

function cgls!(x::Union(AbstractVector{Float64}, Nothing), r::AbstractVector{Float64}, pfe::FixedEffectProblem; tol::Real=1e-10, maxiter::Integer=1000)
    cgls!(x, r, pfe.m, pfe.s, pfe.p, pfe.q; tol = tol, maxiter = maxiter)
end


##############################################################################
##
## demean! is higher level function used in reg etc
##
##############################################################################

function demean!(X::Matrix{Float64}, iterationsv::Vector{Int}, convergedv::Vector{Bool}, 
                 pfe::FixedEffectProblem ; maxiter::Int = 1000, tol::Float64 = 1e-8)
    for j in 1:size(X, 2)
        iterations, converged = cgls!(nothing,  slice(X, :, j), pfe; tol = tol, maxiter = maxiter)
        push!(iterationsv, iterations)
        push!(convergedv, converged)
    end
end


function demean!(x::AbstractVector{Float64}, iterationsv::Vector{Int}, convergedv::Vector{Bool}, pfe::FixedEffectProblem ; maxiter::Int = 1000, tol::Float64 = 1e-8)
    iterations, converged = cgls!(nothing, x, pfe; tol = tol, maxiter = maxiter)
    push!(iterationsv, iterations)
    push!(convergedv, converged)
end


function demean!(::Array, ::Vector{Int}, ::Vector{Bool}, ::Nothing; 
                 maxiter::Int = 1000, tol::Float64 = 1e-8)
    nothing
end

