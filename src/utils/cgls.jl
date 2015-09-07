# r = b0 - Ax0 contains all initial condition and is replaced by the residual b - Ax
# x is used to store the solution of Ax = b
# s, p, q are used for storage. s, p should have dimension size(A, 2). q should have simension size(A, 1). 


# TODO. Follow LMQR for (i) better stopping rule (ii) better projection on zero in case x non identified
function cgls!(x::Union(AbstractVector{Float64}, Nothing), 
               r::AbstractVector{Float64}, A::AbstractMatrix{Float64}, 
               s::Vector{Float64}, p::Vector{Float64}, q::Vector{Float64}; 
               tol::Real=1e-10, maxiter::Int=1000)

    # Initialization.
    converged = false
    iterations = maxiter 

    Ac_mul_B!(p, A, r)
    copy!(s, p)

    normS0 = sumabs2(s)
    normSold = normS0  

    iter = 0
    while iter < maxiter
        iter += 1
        A_mul_B!(q, A, p) 
        α = normSold / sumabs2(q)
        # x = x + αp
        x == nothing || BLAS.axpy!(α, p, x) 
        # r = r - αq
        BLAS.axpy!(-α, q, r) 
        # s = A'r
        Ac_mul_B!(s, A, r) 
        normS = sumabs2(s)
        if ((iter == 1) || α * maxabs(q) <= tol) && normS/normS0 <= tol
            iterations = iter
            converged = true
            break
        end
        beta = normS / normSold
        # p = s + beta p
        scale!(p, beta)
        BLAS.axpy!(1.0, s, p) 
        normSold = normS
    end
    return iterations, converged
end
