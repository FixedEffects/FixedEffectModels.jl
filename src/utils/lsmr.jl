##############################################################################
## LSMR
##
## Minimize ||Ax-b||^2 + λ^2 ||x||^2
##
## Arguments:
## x is initial x0. Will equal the solution.
## r is initial b - Ax0. Will equal the residual b - Ax.
## u, utmp are storage arrays of length size(A, 1)
## v, h, hbar, vtmp are storage arrays of length size(A, 2)
## 
## Adapted from the BSD-licensed Matlab implementation at
##  http://web.stanford.edu/group/SOL/software/lsmr/
##
## A is anything such that
## A_mul_B!(α, A, b, β, c) updates c -> α Ab + βc
## Ac_mul_B!(α, A, b, β, c) updates c -> α A'b + βc
##############################################################################

function lsmr!(x, r, A, u, v, h, hbar; 
    atol = 1e-8, btol = 1e-8, conlim = 1e8, maxiter::Integer=100, λ::Real = zero(Float64))

    conlim > 0.0 ? ctol = 1 / conlim : ctol = zero(Float64)

    # form the first vectors u and v (satisfy  β*u = b,  α*v = A'u)
    copy!(u, r)
    β = norm(u)
    β > 0 && scale!(u, 1/β)
    Ac_mul_B!(1.0, A, u, 0.0, v)
    α = norm(v)
    α > 0 && scale!(v, 1/α)

    # Initialize variables for 1st iteration.
    ζbar = α * β
    αbar = α
    ρ = one(Float64)
    ρbar = one(Float64)
    cbar = one(Float64)
    sbar = zero(Float64)

    copy!(h, v)
    fill!(hbar, zero(Float64))

    # Initialize variables for estimation of ||r||.
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
    minrbar = 1e100

    # Items for use in stopping rules.
    normb = β
    istop = 7
    normr = β

    # Exit if b = 0 or A'b = zero(Float64).
    normAr = α * β
    if normAr == zero(Float64) 
        A_mul_B!(-1.0, A, x, 1.0, r)
        return 1, true
    end

    iter = 0
    while iter < maxiter
        iter += 1
        A_mul_B!(1.0, A, v, -α, u)
        β = norm(u)
        if β > 0
            scale!(u, 1/β)
            Ac_mul_B!(1.0, A, u, -β, v)
            α = norm(v)
            α > 0 && scale!(v, 1/α)
        end

        # Construct rotation Qhat_{k,2k+1}.
        αhat = sqrt(αbar^2 + λ^2)
        chat = αbar / αhat
        shat = λ / αhat

        # Use a plane rotation (Q_i) to turn B_i to R_i.
        ρold = ρ
        ρ = sqrt(αhat^2 + β^2)
        c = αhat / ρ
        s = β / ρ
        θnew = s * α
        αbar = c * α

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
        scale!(hbar, - θbar * ρ / (ρold * ρbarold))
        axpy!(1.0, h, hbar)
        axpy!(ζ / (ρ * ρbar), hbar, x)
        scale!(h, - θnew / ρ)
        axpy!(1.0, v, h)

        ##############################################################################
        ##
        ## Estimate of ||r||
        ##
        ##############################################################################

        # Apply rotation Qhat_{k,2k+1}.
        βacute = chat * βdd
        βcheck = - shat * βdd

        # Apply rotation Q_{k,k+1}.
        βhat = c * βacute
        βdd = - s * βacute
          
        # Apply rotation Qtilde_{k-1}.
        θtildeold = θtilde
        ρtildeold = sqrt(ρdold^2 + θbar^2)
        ctildeold = ρdold / ρtildeold
        stildeold = θbar / ρtildeold
        θtilde = stildeold * ρbar
        ρdold = ctildeold * ρbar
        βd = - stildeold * βd + ctildeold * βhat

        τtildeold = (ζold - θtildeold * τtildeold) / ρtildeold
        τd = (ζ - θtilde * τtildeold) / ρdold
        d  = d + βcheck^2
        normr = sqrt(d + (βd - τd)^2 + βdd^2)

        # Estimate ||A||.
        normA2 = normA2 + β^2
        normA  = sqrt(normA2)
        normA2 = normA2 + α^2

        # Estimate cond(A).
        maxrbar = max(maxrbar, ρbarold)
        if iter > 1 
            minrbar = min(minrbar, ρbarold)
        end
        condA = max(maxrbar, ρtemp) / min(minrbar, ρtemp)
        ##############################################################################
        ##
        ## Test for convergence
        ##
        ##############################################################################

        # Compute norms for convergence testing.
        normAr  = abs(ζbar)
        normx = norm(x)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1 = normr / normb
        test2 = normAr / (normA * normr)
        test3 = 1 / condA
        t1 =  test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = one(Float64)/eps.
        if 1 + test3 <= one(Float64) istop = 6; break end
        if 1 + test2 <= one(Float64) istop = 5; break end
        if 1 + t1 <= one(Float64) istop = 4; break end

        # Allow for tolerances set by the user.
        if test3 <= ctol istop = 3; break end
        if test2 <= atol istop = 2; break end
        if test1 <= rtol  istop = 1; break end
    end
    A_mul_B!(-1.0, A, x, 1.0, r)
    return iter, istop < 7
end
    