##############################################################################
## 
## Least Squares using LSMR
##
##############################################################################



##############################################################################
## 
## FixedEffectVector : vector x in A'Ax = A'b
##
## We define these methods used in lsmr! (duck typing): 
## copy!, fill!, scale!, axpy!, norm
##
##############################################################################

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

eltype(fem::FixedEffectVector) = Float64

length(fev::FixedEffectVector) = reduce(+, map(length, fev._))

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
##
##############################################################################

type FixedEffectMatrix
    _::Vector{FixedEffect}
    m::Int
    n::Int
    cache::Vector{Vector{Float64}}
end

function FixedEffectMatrix(fev::Vector{FixedEffect})
    m = length(fev[1].refs)
    n = reduce(+, map(x -> length(x.scale),  fev))
    caches = Vector{Float64}[]
    for i in 1:length(fev)
        push!(caches, cache(fev[i]))
    end
    return FixedEffectMatrix(fev, m, n, caches)
end

function cache(fe::FixedEffect)
    out = zeros(Float64, length(fe.refs))
    @inbounds @simd for i in 1:length(out)
        out[i] = fe.scale[fe.refs[i]] * fe.interaction[i] * fe.sqrtw[i]
    end
    return out
end

eltype(fem::FixedEffectMatrix) = Float64

size(fem::FixedEffectMatrix, dim::Integer) = (dim == 1) ? fem.m :
                                            (dim == 2) ? fem.n : 1

# Define x -> A * x
function A_mul_B_helper!(α::Number, fe::FixedEffect, 
                        x::Vector{Float64}, y::AbstractVector{Float64}, cache::Vector{Float64})
    @inbounds for (i, j) in zip(1:length(y), eachindex(y))
        y[j] += α * x[fe.refs[i]] * cache[i]
    end
end
function A_mul_B!(α::Number, fem::FixedEffectMatrix, fev::FixedEffectVector, 
                β::Number, y::AbstractVector{Float64})
    safe_scale!(y, β)
    for i in 1:length(fev._)
        A_mul_B_helper!(α, fem._[i], fev._[i], y, fem.cache[i])
    end
    return y
end

# Define x -> A' * x
function Ac_mul_B_helper!(α::Number, fe::FixedEffect, 
                        y::AbstractVector{Float64}, x::Vector{Float64}, cache::Vector{Float64})
    @inbounds for (i, j) in zip(1:length(y), eachindex(y))
        x[fe.refs[i]] += α * y[j] * cache[i]
    end
end
function Ac_mul_B!(α::Number, fem::FixedEffectMatrix, 
                y::AbstractVector{Float64}, β::Number, fev::FixedEffectVector)
   safe_scale!(fev, β)
    for i in 1:length(fev._)
       # @code_warntype Ac_mul_B_helper!(α, fem._[i], y, fev._[i])
        Ac_mul_B_helper!(α, fem._[i], y, fev._[i], fem.cache[i])
    end
    return fev
end

function safe_scale!(x, β)
    if β != 1
        β == 0 ? fill!(x, zero(eltype(x))) : scale!(x, β)
    end
end

##############################################################################
##
## FixedEffectProblem is a wrapper around a FixedEffectMatrix 
## with some storage arrays used when solving (A'A)X = A'y 
##
##############################################################################

type LSMRFixedEffectProblem <: FixedEffectProblem
    m::FixedEffectMatrix
    x::FixedEffectVector
    v::FixedEffectVector
    h::FixedEffectVector
    hbar::FixedEffectVector
    u::Vector{Float64}
end

function FixedEffectProblem(fes::Vector{FixedEffect}, ::Type{Val{:lsmr}})
    m = FixedEffectMatrix(fes)
    x = FixedEffectVector(fes)
    v = FixedEffectVector(fes)
    h = FixedEffectVector(fes)
    hbar = FixedEffectVector(fes)
    u = Array(Float64, size(m, 1))
    return LSMRFixedEffectProblem(m, x, v, h, hbar, u)
end

get_fes(fep::LSMRFixedEffectProblem) = fep.m._

function solve!(fep::LSMRFixedEffectProblem, r::AbstractVector{Float64}; 
    tol::Real = 1e-8, maxiter::Integer = 100_000)
    fill!(fep.x, zero(Float64))
    copy!(fep.u, r)
    x, ch = lsmr!(fep.x, fep.m, fep.u, fep.v, fep.h, fep.hbar; 
        atol = tol, btol = tol, conlim = 1e8, maxiter = maxiter)
    return div(ch.mvps, 2), ch.isconverged
end

function solve_residuals!(fep::LSMRFixedEffectProblem, r::AbstractVector{Float64}; kwargs...)
    iterations, converged = solve!(fep, r; kwargs...)
    A_mul_B!(-1.0, fep.m, fep.x, 1.0, r)
    return r, iterations, converged
end

function solve_coefficients!(fep::LSMRFixedEffectProblem, r::AbstractVector{Float64}; kwargs...)
    iterations, converged = solve!(fep, r; kwargs...)
    for i in 1:length(fep.x._)
        broadcast!(*, fep.x._[i], fep.x._[i], fep.m._[i].scale)
    end
    return fep.x._, iterations, converged
end


