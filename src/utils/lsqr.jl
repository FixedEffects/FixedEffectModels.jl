using Base.LinAlg

function cgls!(x::Union{AbstractVector{Float64}, Nothing}, r::AbstractVector{Float64}, A, b::AbstractVector{Float64}, s::Vector{Float64}, p::Vector{Float64}, q::Vector{Float64}; tol::Real=1e-10, maxiter::Int=1000)

    # Initialization.
    m = size(A, 1)
    n = size(A, 2)
    converged = false
    iterations = maxiter 

    if typeof(x) <: Nothing || all(z -> z == 0.0, x)
        copy!(r, b)
    else
        # r = b - Ax
        r = A_mul_B!(r, A, x)
        scale!(r, -1)
        axpy!(1, b, r)
    end
    # s = p = A'r
    Ac_mul_B!(p, A, r)
    copy!(s, p)

    normS0 = norm(s)  
    normSc = normS0
    Arn    = zeros(maxiter)
    eta    = zeros(maxiter) # norm of residuals
    rho    = zeros(maxiter) # norm or current iterate
    
    # Iterate.
    iter = 0
    while iter < maxiter
        iter += 1
        A_mul_B!(q, A, p) 
        α = normSc / norm(q)
        if !(typeof(x) <: Nothing)
            # x = x + αp
            axpy!(α, p, x) 
        end
        # r = r - αq
        axpy!(-α, q, r) 
        # s = Ar
        Ac_mul_B!(s, A, r) 
        normSt = norm(s)
        if (iter>1) && (normSt <= tol * length(y))
            iterations = iter
            converged = true
            break
        end
        beta = normSt/normSc
        # p = s - beta p
        scale!(p, beta)
        axpy!(1, s, p)        
        # store intermediates and report resuls
        normSc = normSt
    end
    @show iterations
    return iterations, converged
end


function cg!(x::Union{AbstractVector{Float64}, Nothing}, r::AbstractVector{Float64}, A, b::AbstractVector{Float64}, s::Vector{Float64}, p::Vector{Float64}, q::Vector{Float64}; tol::Real=1e-10, maxiter::Int=1000)

    # Initialization.
    m = size(A, 1)
    n = size(A, 2)
    converged = false
    iterations = maxiter 

    if typeof(x) <: Nothing || all(z -> z == 0.0, x)
        copy!(r, b)
    else
        # r = b - Ax
        r = A_mul_B!(r, A, x)
        scale!(r, -1)
        axpy!(1, b, r)
    end
    # s = p = A'r
    Ac_mul_B!(p, A, r)

    normS0 = norm(s)  
    normSc = normS0
    Arn    = zeros(maxiter)
    eta    = zeros(maxiter) # norm of residuals
    rho    = zeros(maxiter) # norm or current iterate
    
    # Iterate.
    iter = 0
    while iter < maxiter
        @show normSc
        iter += 1
        A_mul_B!(q, A, p) 
        α = normSc / norm(q)
        if !(typeof(x) <: Nothing)
            # x = x + αp
            axpy!(α, p, x) 
        end
        # r = r - αq
        axpy!(-α, q, r) 
        normSt = norm(r)
        if (iter>1) && (normSt <= tol)
            iterations = iter
            converged = true
            break
        end
        beta = normSt/normSc
        # p = s + beta p
        scale!(p, beta)
        axpy!(1, r, p)        
        # store intermediates and report resuls
        normSc = normSt
    end
    @show iterations
    return iterations, converged
end

