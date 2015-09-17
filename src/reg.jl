# TODO: minimize memory. For now two factorizations (qr and cholfact)
function reg(f::Formula, df::AbstractDataFrame, 
             vcov_method::AbstractVcovMethod = VcovSimple(); 
             weight::Union{Symbol, Void} = nothing, 
             subset::Union{AbstractVector{Bool}, Void} = nothing, 
             maxiter::Integer = 10000, tol::Real= 1e-10, df_add::Integer = 0, 
             save::Bool = false)

    ##############################################################################
    ##
    ## Parse formula
    ##
    ##############################################################################

    rf = deepcopy(f)
    (has_absorb, absorb_formula, absorb_terms,
        has_iv,iv_formula,iv_terms,endo_formula,endo_terms) = decompose!(rf)
    rt = Terms(rf)
    has_weight = weight != nothing

    ##############################################################################
    ##
    ## Construct new dataframe
    ##
    ##############################################################################

    # create a dataframe without missing values & negative weights
    vars = allvars(rf)
    absorb_vars = allvars(absorb_formula)
    iv_vars = allvars(iv_formula)
    endo_vars = allvars(endo_formula)
    vcov_vars = allvars(vcov_method)

    # create a dataframe without missing values & negative weights
    all_vars = vcat(vars, vcov_vars, absorb_vars, endo_vars, iv_vars)
    all_vars = unique(convert(Vector{Symbol}, all_vars))
    esample = complete_cases(df[all_vars])
    if has_weight
        esample &= isnaorneg(df[weight])
        all_vars = unique(vcat(all_vars, weight))
    end
    if subset != nothing
        if length(subset) != size(df, 1)
            error("df has $(size(df, 1)) rows but the subset vector has $(length(subset)) elements")
        end
        esample &= convert(BitArray, subset)
    end
    subdf = df[esample, all_vars]
    (size(subdf, 1) > 0) || error("sample is empty")

    # remove unusued levels
    main_vars = unique(convert(Vector{Symbol}, vcat(vars, endo_vars, iv_vars)))
    for v in main_vars
        # in case subdataframe, don't construct subdf[v] if you dont need to do it
        if typeof(df[v]) <: PooledDataArray
            dropUnusedLevels!(subdf[v])
        end
    end

    # Compute weight
    sqrtw = get_weight(subdf, weight)

    # Compute pfe, a FixedEffectProblem
    has_intercept = rt.intercept
    if has_absorb
        fixedeffects = FixedEffect[FixedEffect(subdf, a, sqrtw) for a in absorb_terms.terms]
        # in case some FixedEffect does not have interaction, remove the intercept
        if any([typeof(f.interaction) <: Ones for f in fixedeffects]) 
            rt.intercept = false
            has_intercept = true
        end
        pfe = FixedEffectProblem(fixedeffects)
    else
        pfe = nothing
    end

    # Compute data for std errors
    vcov_method_data = VcovMethodData(vcov_method, subdf)

    # initialize iterations and converged
    iterations = Int[]
    converged = Bool[]

    ##############################################################################
    ##
    ## Dataframe --> Matrix
    ##
    ##############################################################################

    # Compute X
    mf = simpleModelFrame(subdf, rt, esample)
    coef_names = coefnames(mf)
    Xexo = ModelMatrix(mf).m
    broadcast!(*, Xexo, Xexo, sqrtw)
    residualize!(Xexo, pfe, iterations, converged; maxiter = maxiter, tol = tol)

    # Compute y
    py = model_response(mf)[:]
    if eltype(py) != Float64
        y = convert(py, Float64)
    else
        y = py
    end
    yname = rt.eterms[1]
    broadcast!(*, y, y, sqrtw)
    # old y will be used if fixed effects
    if has_absorb
        oldy = deepcopy(y)
    end
    residualize!(y, pfe, iterations, converged; maxiter = maxiter, tol = tol)

    # Compute Xendo and Z
    if has_iv
        mf = simpleModelFrame(subdf, endo_terms, esample)
        coef_names = vcat(coef_names, coefnames(mf))
        Xendo = ModelMatrix(mf).m
        broadcast!(*, Xendo, Xendo, sqrtw)
        residualize!(Xendo, pfe, iterations, converged; maxiter = maxiter, tol = tol)
        
        mf = simpleModelFrame(subdf, iv_terms, esample)
        Z = ModelMatrix(mf).m
        if size(Z, 2) < size(Xendo, 2)
            error("Model not identified. There must be at least as many ivs as endogeneneous variables")
        end
        broadcast!(*, Z, Z, sqrtw)
        residualize!(Z, pfe, iterations, converged; maxiter = maxiter, tol = tol)
    end

    # Compute Xhat
    if has_iv
        # get linearly independent columns
        X = hcat(Xexo, Xendo)
        Xqr = qrfact!(X)
        basecolX = basecol(Xqr)
        basecolXexo = basecolX[1:size(Xexo, 2)]
        basecolXendo = basecolX[(size(Xexo, 2)+1):end]
        X = getcols(Xqr, basecolX)
        Xexo = getcols(Xexo, basecolXexo)
        Xendo = getcols(Xendo, basecolXendo)
        newZ = hcat(Xexo, Z)
        newZqr = qrfact!(newZ)
        basecolnewZ = basecol(newZqr)
        newZ = getcols(newZqr, basecolnewZ)
        crossz = cholfact!(At_mul_B(newZ, newZ))
        Pi = crossz \ At_mul_B(newZ, Xendo)
        Xhat = hcat(Xexo, newZ * Pi)
        X = hcat(Xexo, Xendo)
        basecoef = basecolX

        # prepare residuals used for first stage F statistic
        ## partial out Xendo in place wrt (Xexo, Z)
        Xendo_res = BLAS.gemm!('N', 'N', -1.0, newZ, Pi, 1.0, Xendo)
        ## partial out Z in place wrt Xexo
        Pi2 = cholfact!(At_mul_B(Xexo, Xexo)) \ At_mul_B(Xexo, Z)
        Z_res = BLAS.gemm!('N', 'N', -1.0, Xexo, Pi2, 1.0, Z)
    else
        # get linearly independent columns
        Xexoqr = qrfact!(Xexo)
        basecolXexo = basecol(Xexoqr)
        Xexo = getcols(Xexoqr, basecolXexo)
        Xhat = Xexo
        X = Xexo
        basecoef = basecolXexo
    end

    # iter and convergence
    if has_absorb
        iterations = maximum(iterations)
        converged = all(converged)
    end

    ##############################################################################
    ##
    ## Regression
    ##
    ##############################################################################

    # Compute coef and residuals
    crossx =  cholfact!(At_mul_B(Xhat, Xhat))
    coef = crossx \ At_mul_B(Xhat, y)

    ##############################################################################
    ##
    ## Statistics
    ##
    ##############################################################################

    residuals = y - X * coef

    # Compute degrees of freedom
    df_intercept = 0
    if has_absorb || rt.intercept
        df_intercept = 1
    end
    df_absorb = 0
    if has_absorb 
        ## poor man adjustement of df for clustedered errors + fe: only if fe name != cluster name
        for fe in fixedeffects
            if typeof(vcov_method) == VcovCluster && in(fe.factorname, vcov_vars)
                df_absorb += 0
                else
                df_absorb += sum(fe.scale .!= zero(Float64))
            end
        end
    end
    nobs = size(X, 1)
    df_residual = max(1, size(X, 1) - size(X, 2) - df_absorb - df_add)

    # Compute ess, tss, r2, r2 adjusted
    ess = sumabs2(residuals)
    if has_absorb
        tss = compute_tss(y, rt.intercept, sqrtw)
        r2_within = 1 - ess / tss 
        tss = compute_tss(oldy, has_intercept, sqrtw)
        r2 = 1 - ess / tss 
        r2_a = 1 - ess / tss * (nobs - has_intercept) / df_residual 
    else    
        tss = compute_tss(y, has_intercept, sqrtw)
        r2 = 1 - ess / tss 
        r2_a = 1 - ess / tss * (nobs - has_intercept) / df_residual 
    end

    # Compute standard error
    vcov_data = VcovData(Xhat, crossx, residuals, df_residual)
    matrix_vcov = vcov!(vcov_method_data, vcov_data)

    # Compute Fstat
    coefF = deepcopy(coef)
    matrix_vcovF = matrix_vcov
    if rt.intercept && length(coef)==1 
        # TODO: check I can't do better
        F = NaN
        p = NaN
    else
        if rt.intercept && length(coef) > 1
            coefF = coefF[2:end]
            matrix_vcovF = matrix_vcovF[2:end, 2:end]
        end
        F = diagm(coefF)' * (matrix_vcovF \ diagm(coefF))
        F = F[1] 
        if typeof(vcov_method) == VcovCluster 
            nclust = minimum(values(vcov_method_data.size))
            p = ccdf(FDist(size(X, 1) - df_intercept, nclust - 1), F)
        else
            p = ccdf(FDist(size(X, 1) - df_intercept, max(df_residual - df_intercept, 1)), F)
        end    
    end

    # save residuals in a new dataframe
    augmentdf = DataFrame()
    if save
        broadcast!(/, residuals, residuals, sqrtw)
        if all(esample)
            augmentdf[:residuals] = residuals
        else
            augmentdf[:residuals] =  DataArray(Float64, length(esample))
            augmentdf[esample, :residuals] = residuals
        end
        if has_absorb
            mf = simpleModelFrame(subdf, rt, esample)
            oldX = ModelMatrix(mf).m
            if !all(basecoef)
                oldX = oldX[:, basecoef]
            end
            broadcast!(*, oldX, oldX, sqrtw)
            oldresiduals = oldy - oldX * coef
            diffres = oldresiduals - residuals
            augmentdf = hcat(augmentdf, getfe!(pfe, diffres, esample; tol = tol, maxiter = maxiter))
        end
    end

    # Compute Fstat first stage based on Kleibergen-Paap
    if has_iv
        Pip = Pi[(size(Pi, 1) - size(Z_res, 2) + 1):end, :]
        (F_kp, p_kp) = ranktest!(Xendo_res, Z_res, Pip, 
                                  vcov_method_data, size(X, 2), df_absorb)
    end

    # add omitted variables
    if !all(basecoef) 
        newcoef = fill(zero(Float64), length(basecoef))
        newmatrix_vcov = fill(NaN, (length(basecoef), length(basecoef)))
        newindex = [searchsortedfirst(cumsum(basecoef), i) for i in 1:length(coef)]
        for i in 1:length(coef)
            newcoef[newindex[i]] = coef[i]
            for j in 1:length(coef)
                newmatrix_vcov[newindex[i], newindex[j]] = matrix_vcov[i, j]
            end
        end
        coef = newcoef
        matrix_vcov = newmatrix_vcov
    end

    # return
    if !has_iv && !has_absorb 
        return RegressionResult(coef, matrix_vcov, esample, augmentdf, 
                                coef_names, yname, f, nobs, df_residual, 
                                r2, r2_a, F, p)
    elseif has_iv && !has_absorb
        return RegressionResultIV(coef, matrix_vcov, esample, augmentdf, 
                                  coef_names, yname, f, nobs, df_residual, 
                                  r2, r2_a, F, p, F_kp, p_kp)
    elseif !has_iv && has_absorb
        return RegressionResultFE(coef, matrix_vcov, esample, augmentdf, 
                                  coef_names, yname, f, nobs, df_residual, 
                                  r2, r2_a, r2_within, F, p, iterations, converged)
    elseif has_iv && has_absorb 
        return RegressionResultFEIV(coef, matrix_vcov, esample, augmentdf, 
                                   coef_names, yname, f, nobs, df_residual, 
                                   r2, r2_a, r2_within, F, p, F_kp, p_kp, 
                                   iterations, converged)
    end
end


function basecol(QR::Base.LinAlg.QRCompactWY{Float64,Array{Float64,2}})
    R = QR[:R]
    out = Array(Bool, size(QR, 2))
    out[1] = true
    for i in 2:size(QR, 2)
        out[i] = abs(R[i, i]) >= (abs(R[1, 1]) * 1e-10)
    end
    return out
end

function getcols(X::Matrix{Float64},  basecolX::Vector{Bool})
    if sum(basecolX) == size(X, 2)
        return X
    else
        return X[:, basecolX]
    end
end

function getcols(QR::Base.LinAlg.QRCompactWY{Float64,Array{Float64,2}}, basecolX::Vector{Bool})
    Xnew = Array(Float64, size(QR, 1), sum(basecolX))
    idx = 0
    for i in 1:size(QR, 2)
        if basecolX[i]
            idx += 1
            Xnew[:, idx] =  QR[:Q] * slice(QR[:R], :, i)
        end
    end
    return Xnew
end
    
