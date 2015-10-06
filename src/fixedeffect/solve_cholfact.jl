type CholfactFixedEffectProblem{Ta, Tchol} <: FixedEffectProblem
    fes::Vector{FixedEffect}
    A::Ta
    chol::Tchol
    x::Vector{Float64}
end

function FixedEffectProblem(fes::Vector{FixedEffect}, ::Type{Val{:cholfact}})
    nobs = length(fes[1].refs)
    total_len = reduce(+, map(fe -> sum(fe.scale .!= 0), fes))

    # construct model matrix A constituted by fixed effects
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

    A = sparse(I, J, V)

    # compute cholesky factorization once and for all
    chol = cholfact(At_mul_B(A, A))
    x = Array(Float64, total_len)
    return CholfactFixedEffectProblem(fes, A, chol, x)
end

get_fes(fep::CholfactFixedEffectProblem) = fep.fes


# solves A'Ax = A'r
function solve!(fep::CholfactFixedEffectProblem, r::AbstractVector{Float64} ; kwargs...) 
    fep.chol \ At_mul_B!(fep.x, fep.A, r)
end

# updates r as the residual of the projection of r on A
function solve_residuals!(fep::CholfactFixedEffectProblem, r::AbstractVector{Float64}; kwargs...)
    x = solve!(fep, r; kwargs...)
    A_mul_B!(-1.0, fep.A, x, 1.0, r)
    return r, 1, true
end

# solves A'Ax = A'r
# transform x from Vector{Float64} (stacked vector of coefficients) 
# to Vector{Vector{Float64}} (vector of coefficients for each categorical variable)
function solve_coefficients!(fep::CholfactFixedEffectProblem, r::AbstractVector{Float64}; kwargs...)
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


