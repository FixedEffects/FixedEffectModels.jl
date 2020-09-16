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
                    vcov_method::CovarianceEstimator, 
                    df_small::Int, 
                    df_absorb::Int)

    # Compute theta
    Fmatrix = cholesky!(Symmetric(Z' * Z)).U
    Gmatrix = cholesky!(Symmetric(X' * X)).U
    theta = Fmatrix * (Gmatrix' \ Pi')'

    # compute lambda
    svddecomposition = svd(theta, full = true) 
    u = svddecomposition.U
    vt = svddecomposition.Vt

    k = size(X, 2) 
    l = size(Z, 2) 

    u_sub = u[k:l, k:l]
    a_qq = u[1:l, k:l] * (u_sub \ sqrt(u_sub * u_sub'))
        
    vt_sub = vt[k,k]
    b_qq = sqrt(vt_sub * vt_sub') * (vt_sub' \ vt[1:k, k]')

    kronv = kron(b_qq, a_qq')
    lambda = kronv * vec(theta)

    # compute vhat
    if vcov_method isa Vcov.SimpleCovariance
        vlab = cholesky!(Hermitian((kronv * kronv') ./ size(X, 1)))
    else
        K = kron(Gmatrix, Fmatrix)'
        vcovmodel = Vcov.VcovData(Z, K, X, size(Z, 1) - df_small - df_absorb) 
        matrix_vcov2 = Vcov.S_hat(vcovmodel, vcov_method)
        vhat = K \ (K \ matrix_vcov2)'
        vlab = cholesky!(Hermitian(kronv * vhat * kronv'))
    end
    r_kp = lambda' * (vlab \ lambda)
    return r_kp[1]
end
