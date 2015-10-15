##############################################################################
##
## Least Squares via Cholesky Factorization
##
##############################################################################

type CholfactFixedEffectProblem <: FixedEffectProblem
    fes::Vector{FixedEffect}
    m::SparseMatrixCSC{Float64,Int}
    chol::Base.SparseMatrix.CHOLMOD.Factor{Float64}
    x::Vector{Float64}
end

function FixedEffectProblem(fes::Vector{FixedEffect}, ::Type{Val{:cholesky}})
    m = sparse(fes)
    chol = cholfact(At_mul_B(m, m))
    total_len = reduce(+, map(fe -> sum(fe.scale .!= 0), fes))
    x = Array(Float64, total_len)
    return CholfactFixedEffectProblem(fes, m, chol, x)
end

function solve!(fep::CholfactFixedEffectProblem, r::AbstractVector{Float64} ; kwargs...) 
    fep.chol \ At_mul_B!(fep.x, fep.m, r)
end

##############################################################################
##
## Least Squares via QR Factorization
##
##############################################################################

type QRfactFixedEffectProblem <: FixedEffectProblem
    fes::Vector{FixedEffect}
    m::SparseMatrixCSC{Float64,Int}
    qr::Base.SparseMatrix.SPQR.Factorization{Float64}
    b::Vector{Float64}
end

function FixedEffectProblem(fes::Vector{FixedEffect}, ::Type{Val{:qr}})
    m = sparse(fes)
    qr = qrfact(m)
    b = Array(Float64, length(fes[1].refs))
    return QRfactFixedEffectProblem(fes, m, qr, b)
end

function solve!(fep::QRfactFixedEffectProblem, r::AbstractVector{Float64} ; kwargs...) 
    # since \ needs a vector
    copy!(fep.b, r)
    fep.qr \ fep.b
end

##############################################################################
##
## Methods used by all matrix factorization
##
##############################################################################

function Base.sparse(fes::Vector{FixedEffect})
    # construct model matrix A constituted by fixed effects
    nobs = length(fes[1].refs)
    N = length(fes) * nobs
    I = Array(Int, N)
    J = similar(I)
    V = Array(Float64, N)
    start = 0
    idx = 0
    for fe in fes
       for i in 1:length(fe.refs)
           idx += 1
           I[idx] = i
           J[idx] = start + fe.refs[i]
           V[idx] = fe.interaction[i] * fe.sqrtw[i]
       end
       start += sum(fe.scale .!= 0)
    end
    return sparse(I, J, V)
end

# updates r as the residual of the projection of r on A
function solve_residuals!(fep::Union{CholfactFixedEffectProblem, QRfactFixedEffectProblem}, r::AbstractVector{Float64}; kwargs...)
    x = solve!(fep, r; kwargs...)
    A_mul_B!(-1.0, fep.m, x, 1.0, r)
    return r, 1, true
end

# solves A'Ax = A'r
# transform x from Vector{Float64} (stacked vector of coefficients) 
# to Vector{Vector{Float64}} (vector of coefficients for each categorical variable)
function solve_coefficients!(fep::Union{CholfactFixedEffectProblem, QRfactFixedEffectProblem}, r::AbstractVector{Float64}; kwargs...)
    x = solve!(fep, r; kwargs...)
    out = Vector{Float64}[]
    iend = 0
    for fe in get_fes(fep)
        istart = iend + 1
        iend = istart + sum(fe.scale .!= 0) - 1
        push!(out, x[istart:iend])
    end
    return out, 1, true
end

get_fes(fep::Union{CholfactFixedEffectProblem, QRfactFixedEffectProblem}) = fep.fes

