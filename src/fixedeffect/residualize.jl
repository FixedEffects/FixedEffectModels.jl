##############################################################################
##
## get residuals
##
##############################################################################

function residualize!(x::AbstractVector{Float64}, iterationsv::Vector{Int}, convergedv::Vector{Bool}, pfe::FixedEffectProblem ; maxiter::Int = 1000, tol::Float64 = 1e-8)
    iterations, converged = cgls!(nothing, x, pfe; tol = tol, maxiter = maxiter)
    push!(iterationsv, iterations)
    push!(convergedv, converged)
end

function residualize!(X::Matrix{Float64}, iterationsv::Vector{Int}, convergedv::Vector{Bool}, 
                 pfe::FixedEffectProblem ; maxiter::Int = 1000, tol::Float64 = 1e-8)
    for j in 1:size(X, 2)
        residualize!(slice(X, :, j), iterationsv, convergedv, pfe, maxiter = maxiter, tol = tol)
    end
end

function residualize!(::Array, ::Vector{Int}, ::Vector{Bool}, ::Nothing; 
                 maxiter::Int = 1000, tol::Float64 = 1e-8)
    nothing
end



