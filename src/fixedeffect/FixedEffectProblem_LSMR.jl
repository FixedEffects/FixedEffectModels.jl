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
## copyto!, fill!, rmul!, axpy!, norm
##
##############################################################################

struct FixedEffectVector
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

function norm(fev::FixedEffectVector)
    sqrt(reduce(+, map(fe -> sum(abs2, fe),  fev._)))
end

function fill!(fev::FixedEffectVector, x)
    for fe in fev._
        fill!(fe, x)
    end
end

function rmul!(fev::FixedEffectVector, α::Number)
    for fe in fev._
        rmul!(fe, α)
    end
    return fev
end

function copyto!(fev2::FixedEffectVector, fev1::FixedEffectVector)
    for i in 1:length(fev1._)
        copyto!(fev2._[i], fev1._[i])
    end
    return fev2
end


function axpy!(α::Number, fev1::FixedEffectVector, fev2::FixedEffectVector)
    for i in 1:length(fev1._)
        axpy!(α, fev1._[i], fev2._[i])
    end
    return fev2
end



##############################################################################
## 
## FixedEffectMatrix
##
## A is the model matrix of categorical variables
## normalized by diag(1/a1, ..., 1/aN) (Jacobi preconditoner)
##
## We define these methods used in lsmr! (duck typing):
## mul!
##
##############################################################################

struct FixedEffectMatrix
    _::Vector{FixedEffect}
    m::Int
    n::Int
    cache::Vector{Vector{Float64}}
end

function FixedEffectMatrix(fev::Vector{FixedEffect})
    m = length(fev[1].refs)
    n = reduce(+, map(fe -> length(fe.scale),  fev))
    caches = Vector{Float64}[]
    for i in 1:length(fev)
        push!(caches, cache(fev[i]))
    end
    return FixedEffectMatrix(fev, m, n, caches)
end

function cache(fe::FixedEffect)
    out = zeros(Float64, length(fe.refs))
    @fastmath @inbounds @simd for i in 1:length(out)
        out[i] = fe.scale[fe.refs[i]] * fe.interaction[i] * fe.sqrtw[i]
    end
    return out
end

eltype(fem::FixedEffectMatrix) = Float64

size(fem::FixedEffectMatrix, dim::Integer) = (dim == 1) ? fem.m :
                                            (dim == 2) ? fem.n : 1
Base.adjoint(fem) = Adjoint(fem)

function mul!(y::AbstractVector{Float64}, fem::FixedEffectMatrix, fev::FixedEffectVector, α::Number, β::Number)
    safe_rmul!(y, β)
    for i in 1:length(fev._)
        helperN!(α, fem._[i], fev._[i], y, fem.cache[i])
    end
    return y
end
# Define x -> A * x
function helperN!(α::Number, fe::FixedEffect, 
    x::Vector{Float64}, y::AbstractVector{Float64}, cache::Vector{Float64})
    @inbounds @simd for i in 1:length(y)
        y[i] += α * x[fe.refs[i]] * cache[i]
    end
end

function mul!(fev::FixedEffectVector, Cfem::Adjoint{T, FixedEffectMatrix}, y::AbstractVector{Float64}, α::Number, β::Number) where {T}
    fem = adjoint(Cfem)
    safe_rmul!(fev, β)
    for i in 1:length(fev._)
        helperC!(α, fem._[i], y, fev._[i], fem.cache[i])
    end
    return fev
end

# Define x -> A' * x
function helperC!(α::Number, fe::FixedEffect, 
                        y::AbstractVector{Float64}, x::Vector{Float64}, cache::Vector{Float64})
    @inbounds @simd for i in 1:length(y)
        x[fe.refs[i]] += α * y[i] * cache[i]
    end
end

function safe_rmul!(x, β)
    if !(β ≈ 1.0)
        β ≈ 0.0 ? fill!(x, zero(eltype(x))) : rmul!(x, β)
    end
end



##############################################################################
##
## FixedEffectProblem is a wrapper around a FixedEffectMatrix 
## with some storage arrays used when solving (A'A)X = A'y 
##
##############################################################################

struct LSMRFixedEffectProblem <: FixedEffectProblem
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
    u = Array{Float64}(undef, size(m, 1))
    return LSMRFixedEffectProblem(m, x, v, h, hbar, u)
end

get_fes(fep::LSMRFixedEffectProblem) = fep.m._

function solve!(fep::LSMRFixedEffectProblem, r::AbstractVector{Float64}; 
    tol::Real = 1e-8, maxiter::Integer = 100_000)
    fill!(fep.x, zero(Float64))
    copyto!(fep.u, r)
    x, ch = lsmr!(fep.x, fep.m, fep.u, fep.v, fep.h, fep.hbar; 
        atol = tol, btol = tol, conlim = 1e8, maxiter = maxiter)
    return div(ch.mvps, 2), ch.isconverged
end

function solve_residuals!(fep::LSMRFixedEffectProblem, r::AbstractVector{Float64}; kwargs...)
    iterations, converged = solve!(fep, r; kwargs...)
    mul!(r, fep.m, fep.x, -1.0, 1.0)
    return r, iterations, converged
end

function solve_coefficients!(fep::LSMRFixedEffectProblem, r::AbstractVector{Float64}; kwargs...)
    iterations, converged = solve!(fep, r; kwargs...)
    for i in 1:length(fep.x._)
        fep.x._[i] .= fep.x._[i] .* fep.m._[i].scale
    end
    return fep.x._, iterations, converged
end


##############################################################################
##
## LSMR Parallel
##
## One needs to construct a new fe matrix / fe vectirs for each LHS/RHS
##
##############################################################################

struct LSMRParallelFixedEffectProblem <: FixedEffectProblem
    fes::Vector{FixedEffect}
end

FixedEffectProblem(fes::Vector{FixedEffect}, ::Type{Val{:lsmr_parallel}}) = LSMRParallelFixedEffectProblem(fes)
get_fes(fep::LSMRParallelFixedEffectProblem) = fep.fes


function residualize!(X::Union{AbstractVector{Float64}, Matrix{Float64}}, fep::LSMRParallelFixedEffectProblem, iterationsv::Vector{Int}, convergedv::Vector{Bool}; kwargs...)
    # parallel
    result = pmap(x -> solve_residuals!(fep, x ;kwargs...), [X[:, j] for j in 1:size(X, 2)])
    for j in 1:size(X, 2)
        X[:, j] = result[j][1]
        push!(iterationsv, result[j][2])
        push!(convergedv, result[j][3])
    end
end

function solve_residuals!(fep::LSMRParallelFixedEffectProblem, r::AbstractVector{Float64}; kwargs...)
    newfep = FixedEffectProblem(get_fes(fep), Val{:lsmr})
    result = solve_residuals!(newfep, r; kwargs...)
    result
end
function solve_coefficients!(fep::LSMRParallelFixedEffectProblem, r::AbstractVector{Float64}; kwargs...)
    solve_coefficients!(FixedEffectProblem(get_fes(fep), Val{:lsmr}), r; kwargs...)
end


##############################################################################
##
## LSMR MultiThreaded
##
## One needs to construct a new fe matrix / fe vectirs for each LHS/RHS
##
##############################################################################

struct LSMRThreadslFixedEffectProblem <: FixedEffectProblem
    fes::Vector{FixedEffect}
end

FixedEffectProblem(fes::Vector{FixedEffect}, ::Type{Val{:lsmr_threads}}) = LSMRThreadslFixedEffectProblem(fes)
get_fes(fep::LSMRThreadslFixedEffectProblem) = fep.fes


function residualize!(X::Union{AbstractVector{Float64}, Matrix{Float64}}, fep::LSMRThreadslFixedEffectProblem, iterationsv::Vector{Int}, convergedv::Vector{Bool}; kwargs...)
   iterations_X = Vector{Int}(size(X, 2))
   converged_X = Vector{Bool}(size(X, 2))
   Threads.@threads for j in 1:size(X, 2)
        r, iterations, converged = solve_residuals!(fep, view(X, :, j); kwargs...)
        iterations_X[j] = iterations
        converged_X[j] = converged
   end
   append!(iterationsv, iterations_X)
   append!(convergedv, converged_X)
end

function solve_residuals!(fep::LSMRThreadslFixedEffectProblem, r::AbstractVector{Float64}; kwargs...)
    #x0 = now()
    solve_residuals!(FixedEffectProblem(get_fes(fep), Val{:lsmr}), r; kwargs...)
    #@show Threads.threadid(), now() - x0

end
function solve_coefficients!(fep::LSMRThreadslFixedEffectProblem, r::AbstractVector{Float64}; kwargs...)
    solve_coefficients!(FixedEffectProblem(get_fes(fep), Val{:lsmr}), r; kwargs...)
end

