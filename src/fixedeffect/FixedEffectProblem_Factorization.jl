##############################################################################
##
## Least Squares via Cholesky Factorization
##
##############################################################################

struct CholeskyFixedEffectProblem <: FixedEffectProblem
    fes::Vector{FixedEffect}
    m::SparseMatrixCSC{Float64,Int}
    chol::Factor{Float64}
    x::Vector{Float64}
end

function FixedEffectProblem(fes::Vector{FixedEffect}, ::Type{Val{:cholesky}})
    m = sparse(fes)
    chol = cholesky!(Symmetric(m' * m))
    total_len = reduce(+, map(fe -> sum(fe.scale .!= 0), fes))
    CholeskyFixedEffectProblem(fes, m, chol, Array{Float64}(undef, total_len))
end

function solve!(fep::CholeskyFixedEffectProblem, r::AbstractVector{Float64} ; kwargs...) 
    fep.chol \ mul!(fep.x, fep.m', r)
end

##############################################################################
##
## Least Squares via QR Factorization
##
##############################################################################

struct QRFixedEffectProblem <: FixedEffectProblem
    fes::Vector{FixedEffect}
    m::SparseMatrixCSC{Float64,Int}
    qr::QRSparse{Float64, Int}
    b::Vector{Float64}
end

function FixedEffectProblem(fes::Vector{FixedEffect}, ::Type{Val{:qr}})
    m = sparse(fes)
    qr = qr(m)
    b = Array{Float64}(undef, length(fes[1].refs))
    QRFixedEffectProblem(fes, m, qr, b)
end

function solve!(fep::QRFixedEffectProblem, r::AbstractVector{Float64} ; kwargs...) 
    # since \ needs a vector
    copy!(fep.b, r)
    fep.qr \ fep.b
end

##############################################################################
##
## Methods used by all matrix factorization
##
##############################################################################

# construct the sparse matrix of fixed effects A in  A'Ax = A'r
function sparse(fes::Vector{FixedEffect})
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
           V[idx] = fe.interaction[i] * fe.sqrtw[i]
       end
       start += sum(fe.scale .!= 0)
    end
    sparse(I, J, V)
end

# updates r as the residual of the projection of r on A
function solve_residuals!(fep::Union{CholeskyFixedEffectProblem, QRFixedEffectProblem}, r::AbstractVector{Float64}; kwargs...)
    x = solve!(fep, r; kwargs...)
    gemm!('N', 'N', -1.0, fep.m, x, 1.0, r)
    return r, 1, true
end

# solves A'Ax = A'r
# transform x from Vector{Float64} (stacked vector of coefficients) 
# to Vector{Vector{Float64}} (vector of coefficients for each categorical variable)
function solve_coefficients!(fep::Union{CholeskyFixedEffectProblem, QRFixedEffectProblem}, r::AbstractVector{Float64}; kwargs...)
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

get_fes(fep::Union{CholeskyFixedEffectProblem, QRFixedEffectProblem}) = fep.fes

