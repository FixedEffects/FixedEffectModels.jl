##############################################################################
##
## Least Squares via Cholesky Factorization
##
##############################################################################

struct CholeskyFixedEffectProblem{T} <: FixedEffectProblem
    fes::Vector{FixedEffect}
    m::SparseMatrixCSC{Float64,Int}
    cholm::T
    x::Vector{Float64}
end

function FixedEffectProblem(fes::Vector{FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:cholesky}})
    m = sparse(fes, sqrtw)
    cholm = cholesky(Symmetric(m' * m))
    total_len = sum(length(unique(fe.refs)) for fe in fes)
    CholeskyFixedEffectProblem(fes, m, cholm, Array{Float64}(undef, total_len))
end

function solve!(fep::CholeskyFixedEffectProblem, r::AbstractVector{Float64} ; kwargs...)
    fep.cholm \ mul!(fep.x, fep.m', r)
end

##############################################################################
##
## Least Squares via QR Factorization
##
##############################################################################

struct QRFixedEffectProblem{T} <: FixedEffectProblem
    fes::Vector{FixedEffect}
    m::SparseMatrixCSC{Float64,Int}
    qrm::T
    b::Vector{Float64}
end

function FixedEffectProblem(fes::Vector{FixedEffect}, sqrtw::AbstractVector, ::Type{Val{:qr}})
    m = sparse(fes, sqrtw)
    qrm = qr(m)
    b = Array{Float64}(undef, length(fes[1].refs))
    QRFixedEffectProblem(fes, m, qrm, b)
end

function solve!(fep::QRFixedEffectProblem, r::AbstractVector{Float64} ; kwargs...) 
    # since \ needs a vector
    copyto!(fep.b, r)
    fep.qrm \ fep.b
end

##############################################################################
##
## Methods used by all matrix factorization
##
##############################################################################

# construct the sparse matrix of fixed effects A in  A'Ax = A'r
function sparse(fes::Vector{FixedEffect}, sqrtw::AbstractVector)
    # construct model matrix A constituted by fixed effects
    nobs = length(fes[1].refs)
    N = length(fes) * nobs
    I = Array{Int}(undef, N)
    J = similar(I)
    V = Array{Float64}(undef, N)
    start = 0
    idx = 0
    for fe in fes
       for i in 1:length(fe.refs)
           idx += 1
           I[idx] = i
           J[idx] = start + fe.refs[i]
           V[idx] = fe.interaction[i] * sqrtw[i]
       end
       start += length(unique(fe.refs))
    end
    sparse(I, J, V)
end

# updates r as the residual of the projection of r on A
function solve_residuals!(fep::Union{CholeskyFixedEffectProblem, QRFixedEffectProblem}, r::AbstractVector{Float64}; kwargs...)
    x = solve!(fep, r; kwargs...)
    mul!(r, fep.m, x, -1.0, 1.0)
    return r, 1, true
end

# solves A'Ax = A'r
# transform x from Vector{Float64} (stacked vector of coefficients) 
# to Vector{Vector{Float64}} (vector of coefficients for each categorical variable)
function solve_coefficients!(fep::Union{CholeskyFixedEffectProblem, QRFixedEffectProblem}, r::AbstractVector{Float64}; kwargs...)
    x = solve!(fep, r; kwargs...)
    out = Vector{Float64}[]
    iend = 0
    for fe in fep.fes
        istart = iend + 1
        iend = istart + length(unique(fe.refs)) - 1
        push!(out, x[istart:iend])
    end
    return out, 1, true
end

get_fes(fep::Union{CholeskyFixedEffectProblem, QRFixedEffectProblem}) = fep.fes

