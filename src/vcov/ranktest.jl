
##############################################################################
##
## The following function follows the command ranktest (called by ivreg2)
## RANKTEST: Stata module to test the rank of a matrix using the Kleibergen-Paap rk statistic
## Authors: Frank Kleibergen, Mark E Schaffer
## IVREG2: Stata module for extended instrumental variables/2SLS and GMM estimation
## Authors: Christopher F Baum, Mark E Schaffer, Steven Stillman
## More precisely, it corresponds to the Stata command:  ranktest  (X) (Z), wald full
##############################################################################

function ranktest!(X::Matrix{Float64}, 
                    Z::Matrix{Float64}, 
                    Pi::Matrix{Float64}, 
                    vcov_method::AbstractVcovMethod, 
                    df_small::Int, 
                    df_absorb::Int)

    K = size(X, 2) 
    L = size(Z, 2) 

    crossz = cholfact!(At_mul_B(Z, Z), :U)
    crossx = cholfact!(At_mul_B(X, X), :U)

    Fmatrix = crossz[:U]
    Gmatrix = crossx[:U]
    theta = Fmatrix * (Gmatrix' \ Pi')'

    svddecomposition = svdfact(theta, thin = false) 
    u = svddecomposition.U
    vt = svddecomposition.Vt

    # compute lambda
    if K == 1
        a_qq = sqrtm(A_mul_Bt(u, u))
        b_qq = sqrtm(A_mul_Bt(vt, vt)) 
    else
        u_12 = u[1:(K-1),(K:L)]
        v_12 = vt[1:(K-1),K]
        u_22 = u[(K:L),(K:L)]
        v_22 = vt[K,K]
        a_qq = vcat(u_12, u_22) * (u_22 \ sqrtm(A_mul_Bt(u_22, u_22)))
        b_qq = sqrtm(A_mul_Bt(v_22, v_22)) * (v_22' \ vcat(v_12, v_22)')
    end
    kronv = kron(b_qq, a_qq')
    lambda = kronv * vec(theta)

    # compute vhat
    if typeof(vcov_method) == VcovSimpleMethod
        vhat= eye(L * K) / size(X, 1)
    else
        temp1 = convert(Matrix{eltype(Gmatrix)}, Gmatrix)
        temp2 = convert(Matrix{eltype(Fmatrix)}, Fmatrix)
        k = kron(temp1, temp2)'
        vcovmodel = VcovData(Z, k, X, size(Z, 1) - df_small - df_absorb) 
        matrix_vcov2 = shat!(vcov_method, vcovmodel)
        vhat = k \ (k \ matrix_vcov2)'
    end

    # return statistics
    # why do I need to add Hermitian? (since 0.5)
    vlab = cholfact!(Hermitian(A_mul_Bt(kronv * vhat, kronv)))
    r_kp = lambda' * (vlab \ lambda)
    p_kp = ccdf(Chisq((L-K+1 )), r_kp[1])
    F_kp = r_kp[1] / size(Z, 2)
    return F_kp, p_kp
end
