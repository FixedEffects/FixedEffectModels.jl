

##############################################################################
##
## CG on (Id - AA'/diag(AA')X = 0)
## 
##############################################################################

type CimminoProblem
    X::Matrix{Float64}
    invsumabs2::Vector{Float64}
    z::Vector{Float64}
end
function CimminoProblem(X::Matrix{Float64}) 
    return CimminoProblem(X, 1./vec(sumabs2(X, 1)),  Array(Float64, size(X, 1)))
end

function Base.A_mul_B!(y::Vector{Float64}, cp::CimminoProblem, x::Vector{Float64})
	# Multiply by (I - AA'/diag(AA')X)
    At_mul_B(cp.z, cp.X, x) 
    broadcast!(*, cp.z, cp.z, cp.invsumabs2)
    copy!(y, x)
    BLAS.gemm!('N', 'N', -1.0, cp.z, tmp, 1.0, y)
    return y
end

function cg!(x, r, A; tol::Real=1e-8, maxiter::Int=100)

    # Initialization.
    converged = false
    iterations = maxiter 

    p = deepcopy(r)
    q = similar(r)
    tmp = Array(Float64, size(A.X, 2))
    ssr0 = sumabs2(r)
    ssrold = ssr0  
    iter = 0
    while iter < maxiter
        iter += 1
        A_mul_B!(q, A, p) 
        α = ssrold / dot(q, p)
        Base.BLAS.axpy!(α, p, x) 
        Base.BLAS.axpy!(-α, q, r)
        ssr = sumabs2(r)
        error = sumabs2(At_mul_B!(tmp, A.X, x))
        @show error
        if error <= tol^2 
            iterations = iter
            converged = true
            break
        end
        β = ssr / ssrold
        scale!(p, β)
        Base.BLAS.axpy!(1.0, r, p)
        ssrold = ssr
    end
    return iterations, converged
end


function residualize!(y, cp::CimminoProblem)
    # r = b- Ax0
    r = similar(y)
    A_mul_B!(r, cp, y)
    scale!(r, -1.0)
    # start conjugate gradient
    iterations, converged = cg!(y, r, cp)
    @assert sumabs2(At_mul_B(cp.X, cp.X) \ At_mul_B(cp.X, y)) <= 1e-10
    return iterations
end

##############################################################################
##
## CG on AA' u = b with X = A'u (CGNE)
## 
##############################################################################

function cgne!(r, A; tol::Real=1e-8, maxiter::Int=100)

    # Initialization.
    converged = false
    iterations = maxiter 
    invdiag = 1./vec(sumabs2(A, 1))
    p = At_mul_B(A, r)
    broadcast!(*, z, p, invsumabs2)
    ssr0 = dot(r, r)
    ssrold = ssr0  
    iter = 0
    while iter < maxiter
    	α = ssrold / dot(p, p)
    	Base.BLAS.gemv!('N', 1.0, A, p, 1.0, r)
    	ssr = dot(r, r)
        β = ssr / ssrold
        # p = A'r + β p
        Base.BLAS.gemv!('T', 1.0, A, r, β, p)
        ssrold = ssr
    end
    return iterations, converged
end

function residualize!(y, X::Matrix{Float64})
    # start conjugate gradient
    iterations, converged = cgne!(y, X)
    @assert sumabs2(At_mul_B(X, X) \ At_mul_B(X, y)) <= 1e-10
    return iterations
end

residualize(y, X) = residualize!(deepcopy(y), X)



##############################################################################
##
## CG on A'A X = A'y (CGNR)
## 
##############################################################################

function cgnr!(r, A; tol::Real=1e-8, maxiter::Int=100)

    # Initialization.
    converged = false
    iterations = maxiter 
    s = Array(Float64, size(A, 2))
    p = similar(s)
    ptmp = similar(s)
    q = similar(r)
    invdiag = 1./vec(sumabs2(A, 1))

    At_mul_B!(s, A, r)
    broadcast!(*, ptmp, s, invdiag)
    copy!(p, ptmp)
    ssr0 = dot(s, ptmp)
    ssrold = ssr0  
    iter = 0
    tmp = similar(s)
    while iter < maxiter
        iter += 1
        A_mul_B!(q, A, p) 
        At_mul_B!(ptmp, A, q)
        α = ssrold / dot(ptmp, p)
        Base.BLAS.axpy!(-α, q, r)
        Base.BLAS.axpy!(-α, ptmp, s)
        broadcast!(*, ptmp, s, invdiag)
        ssr = dot(s, ptmp)
        error = sumabs2(At_mul_B!(tmp, A, r)) 
        @show error
        if error <= tol^2 
            iterations = iter
            converged = true
            break
        end
        β = ssr / ssrold
        # p = s + β p
        scale!(p, β)
        Base.BLAS.axpy!(1.0, ptmp, p) 
        ssrold = ssr
    end
    return iterations, converged
end

function residualize!(y, X::Matrix{Float64})
    # start conjugate gradient
    iterations, converged = cgnr!(y, X)
    @assert sumabs2(At_mul_B(X, X) \ At_mul_B(X, y)) <= 1e-10
    return iterations
end

residualize(y, X) = residualize!(deepcopy(y), X)


##############################################################################
##
## Tests
## For both methods, error stopped when sumabs2(A' x residual) <= tol
## Both method require the same amount of computations at each iteration
##
##############################################################################
X = randn(500, 2)
y = randn(500)
residualize(y, CimminoProblem(X))
residualize(y, X)


X = randn(5000, 2)
y = randn(5000)
residualize(y, CimminoProblem(X))
residualize(y, X)


X = randn(50000, 2)
y = randn(50000)
residualize(y, CimminoProblem(X))
residualize(y, X)



X = randn(500, 10)
y = randn(500)
residualize(y, CimminoProblem(X))
residualize(y, X)


X = randn(5000, 10)
y = randn(5000)
residualize(y, CimminoProblem(X))
residualize(y, X)

X = randn(50000, 10)
y = randn(50000)
residualize(y, CimminoProblem(X))
residualize(y, X)


X = randn(500, 100)
y = randn(500)
residualize(y, CimminoProblem(X))
residualize(y, X)


X = randn(5000, 100)
y = randn(5000)
residualize(y, CimminoProblem(X))
residualize(y, X)


X = randn(50000, 100)
y = randn(50000)
residualize(y, CimminoProblem(X))
residualize(y, X)



X = randn(500, 1000)
y = randn(500)
residualize(y, CimminoProblem(X))
residualize(y, X)


X = randn(5000, 1000)
y = randn(5000)
residualize(y, CimminoProblem(X))
residualize(y, X)


X = randn(50000, 1000)
y = randn(50000)
residualize(y, CimminoProblem(X))
residualize(y, X)
