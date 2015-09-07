##############################################################################
##
## get residuals
##
##############################################################################

function residualize!(x::AbstractVector{Float64}, pfe::FixedEffectProblem, 
	                  iterationsv::Vector{Int}, convergedv::Vector{Bool}; 
	                  maxiter::Int = 1000, tol::Float64 = 1e-8)
    iterations, converged = cgls!(nothing, x, pfe;  maxiter = maxiter, tol = tol)
    push!(iterationsv, iterations)
    push!(convergedv, converged)
end

function residualize!(X::Matrix{Float64}, pfe::FixedEffectProblem, 
	                  iterationsv::Vector{Int}, convergedv::Vector{Bool}; 
	                  maxiter::Int = 1000, tol::Float64 = 1e-8)
    for j in 1:size(X, 2)
        residualize!(slice(X, :, j), pfe, iterationsv, convergedv, maxiter = maxiter, tol = tol)
    end
end

function residualize!(::Array, ::Nothing, 
	                  ::Vector{Int}, ::Vector{Bool}; 
                      maxiter::Int = 1000, tol::Float64 = 1e-8)
    nothing
end



