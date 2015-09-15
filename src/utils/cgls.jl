##############################################################################
##
## Solve A'Ax = A'b by cgls 
## x is the initial guess for x. It is modified in place
## r equals b - Ax0 where x0 is the initial guess for x. It is modified in place and equals b - Ax
## s, p are used for storage. They have dimension size(A, 2). 
## q is used for storage. It has dimension size(A, 1). 
##
##############################################################################

# TODO. Follow LMQR for better stopping rule
function cgls!(x, r, A, s, p, q; tol::Real=1e-10, maxiter::Int=1000)

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
        x == nothing || axpy!(α, p, x) 
        # r = r - αq
        axpy!(-α, q, r) 
        # s = A'r
        Ac_mul_B!(s, A, r) 
        normS = sumabs2(s)
        if (iter == 1 && normS/normS0 <= tol) || α * maxabs(q) <= tol 
            iterations = iter
            converged = true
            break
        end
        β = normS / normSold
        # p = s + β p
        scale!(p, β)
        axpy!(1.0, s, p) 
        normSold = normS
    end
    return iterations, converged
end
