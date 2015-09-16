##############################################################################
##
## Solve A'Ax = A'b by cgls with Jacobi preconditioner
## x is the initial guess for x. It is modified in place
## r equals b - Ax0 where x0 is the initial guess for x. It is modified in place and equals b - Ax
## s, p are used for storage. They have dimension size(A, 2). 
## q is used for storage. It has dimension size(A, 1). 
##
##############################################################################

# TODO. Follow LMQR for better stopping rule
function cgls!(x, r, A, q, normalization, s, p, z, ptmp; 
               tol::Real=1e-5, maxiter::Int=100)

    # Initialization.
    converged = false
    iterations = maxiter 

    Ac_mul_B!(s, A, r)
    sumabs2!(normalization, A)
    broadcast!(/, z, s, normalization)
    copy!(p, z)
    ssr0 = dot(s, z)
    ssrold = ssr0  

    iter = 0
    while iter < maxiter
        iter += 1
        A_mul_B!(q, A, p) 
        Ac_mul_B!(ptmp, A, q)
        α = ssrold / dot(ptmp, p)
        x == nothing || axpy!(α, p, x) 
        axpy!(-α, ptmp, s)
        axpy!(-α, q, r)
        broadcast!(/, z, s, normalization)
        ssr = dot(s, z)
        if ssr <= tol^2 * ssr0
            iterations = iter
            converged = true
            break
        end
        β = ssr / ssrold
        # p = s + β p
        scale!(p, β)
        axpy!(1.0, z, p) 
        ssrold = ssr
    end
    return iterations, converged
end


