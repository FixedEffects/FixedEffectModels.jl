##############################################################################
## x is initial x0
## r is initilal b - Ax0
## u, utmp are size(A, 1)
## v, h, hbar, vtmp are size(A, 2)
## 
## Adapted from the BSD-licensed Matlab implementation at
##  http://web.stanford.edu/group/SOL/software/lsmr/
##############################################################################

# TODO. Follow LMQR for better stopping rule
function lsmr!(x, r, A, u, utmp, v, h, hbar, vtmp; 
               tol::Real=1e-8, maxiter::Integer=100, λ::Real = zero(Float64))


    # Determine dimensions m and n, and
    # form the first vectors u and v.
    # These satisfy  β*u = b,  α*v = A'u.
    copy!(u, r)
    β = norm(u)
    β > 0 && scale!(u, 1/β)
    Ac_mul_B!(v, A, u)

    λ = zero(Float64)
    atol = tol
    btol = tol
    conlim = 1e8
    localSize = zero(Float64)

    α = norm(v)
    α > 0 && scale!(v, 1/α)

    #  Initialize variables for 1st iteration.
    ζbar = α*β
    αbar = α
    ρ = one(Float64)
    ρbar = one(Float64)
    cbar = one(Float64)
    sbar = zero(Float64)

    copy!(h, v)
    fill!(hbar, zero(Float64))

    #  Initialize variables for estimation of ||r||.

    βdd = β
    βd = zero(Float64)
    ρdold = one(Float64)
    τtildeold = zero(Float64)
    θtilde  = zero(Float64)
    ζ = zero(Float64)
    d = zero(Float64)

    # Initialize variables for estimation of ||A|| and cond(A).

    normA2 = α^2
    maxrbar = zero(Float64)
    minrbar = 1e+100

    # Items for use in stopping rules.
    normb  = β
    istop  = zero(Float64)
    ctol = zero(Float64)         
    if conlim > 0 ctol = one(Float64) / conlim; end
    normr  = β

#  Exit if b=0 or A'b = zero(Float64).
    normAr = α * β
    if normAr == zero(Float64) 
        A_mul_B!(utmp, A, x)
        axpy!(-1.0, utmp, r)
        return 1, true
    end

    iter = zero(Float64)
    while iter < maxiter
        iter += one(Float64)
        scale!(u, -α)
        A_mul_B!(utmp, A, v)
        axpy!(1.0, utmp, u)
        β = norm(u)
        if β > 0
            scale!(u, 1/β)
            scale!(v, -β)
            Ac_mul_B!(vtmp, A, u)
            axpy!(1.0, vtmp, v)
            α = norm(v)
            α > 0 && scale!(v, 1/α)
        end

        # Construct rotation Qhat_{k,2k+1}.

        αhat = sqrt(αbar^2 + λ^2)
        chat = αbar/αhat
        shat = λ/αhat

        # Use a plane rotation (Q_i) to turn B_i to R_i.

        ρold = ρ
        ρ = sqrt(αhat^2 + β^2)
        c = αhat/ρ
        s = β/ρ
        θnew = s*α
        αbar = c*α

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar.

        ρbarold = ρbar
        ζold = ζ
        θbar = sbar * ρ
        ρtemp = cbar * ρ
        ρbar = sqrt(cbar^2 * ρ^2 + θnew^2)
        cbar = cbar * ρ / ρbar
        sbar = θnew / ρbar
        ζ = cbar * ζbar
        ζbar = - sbar * ζbar

        # Update h, h_hat, x.
        scale!(hbar, -θbar * ρ / (ρold * ρbarold))
        axpy!(1.0, h, hbar)
        axpy!(ζ/(ρ*ρbar), hbar, x)
        scale!(h, -θnew/ρ)
        axpy!(1.0, v, h)

        # Estimate of ||r||.

        # Apply rotation Qhat_{k,2k+1}.
        βacute = chat * βdd
        βcheck = - shat * βdd

        # Apply rotation Q_{k,k+1}.
        βhat = c * βacute
        βdd = - s * βacute
          
        # Apply rotation Qtilde_{k-1}.
        # βd = βd_{k-1} here.

        θtildeold = θtilde
        ρtildeold = sqrt(ρdold^2 + θbar^2)
        ctildeold = ρdold/ρtildeold
        stildeold = θbar/ρtildeold
        θtilde = stildeold * ρbar
        ρdold = ctildeold * ρbar
        βd = - stildeold * βd + ctildeold * βhat

        # βd = βd_k here.
        # ρdold = ρd_k  here.

        τtildeold = (ζold - θtildeold*τtildeold)/ρtildeold
        τd = (ζ - θtilde*τtildeold)/ρdold
        d  = d + βcheck^2
        normr = sqrt(d + (βd - τd)^2 + βdd^2)

        # Estimate ||A||.
        normA2 = normA2 + β^2
        normA  = sqrt(normA2)
        normA2 = normA2 + α^2

        # Estimate cond(A).
        maxrbar = max(maxrbar, ρbarold)
        if iter>1 
            minrbar = min(minrbar, ρbarold)
        end
        condA = max(maxrbar, ρtemp) / min(minrbar, ρtemp)

        # Test for convergence.

        # Compute norms for convergence testing.
        normAr  = abs(ζbar)
        normx = norm(x)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1 = normr / normb
        test2 = normAr / (normA * normr)
        test3 = one(Float64) / condA
        t1 =  test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = one(Float64)/eps.

        if iter >= maxiter  istop = 7; end
        if 1 + test3 <= one(Float64)  istop = 6; end
        if 1 + test2 <= one(Float64)  istop = 5; end
        if 1 + t1 <= one(Float64)  istop = 4; end

        # Allow for tolerances set by the user.

        if test3 <= ctol   istop = 3; end
        if test2 <= atol   istop = 2; end
        if test1 <= rtol   istop = one(Float64); end
        if istop > 0 break end
    end
    A_mul_B!(utmp, A, x)
    axpy!(-1.0, utmp, r)
    return iter, istop > 0
end
    