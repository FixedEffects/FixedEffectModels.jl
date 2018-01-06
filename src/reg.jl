"""
Estimate a linear model with high dimensional categorical variables / instrumental variables

### Arguments
* `df` : AbstractDataFrame
* `f` : Formula, 
* `fe` : Fixed effect formula.
* `vcov` : Vcov formula. Default to `simple`. `robust` and `cluster()` are also implemented
* `weights`: Weights formula. Corresponds to analytical weights
* `subset` : Expression of the form State .>= 30
* `save` : Should residuals and eventual estimated fixed effects saved in a dataframe?
* `maxiter` : Maximum number of iterations
* `tol` : tolerance
* `method` : Default is lsmr (akin to conjugate gradient descent). Other choices are qr and cholesky (factorization methods)


### Returns
* `::AbstractRegressionResult` : a regression results

### Details
A typical formula is composed of one dependent variable, exogeneous variables, endogeneous variables, and instruments
```
depvar ~ exogeneousvars + (endogeneousvars ~ instrumentvars
```
Categorical variable should be of type PooledDataArray.  Use the function `pool` to create PooledDataArray.
Models with instruments variables are estimated using 2SLS. `reg` tests for weak instruments by computing the Kleibergen-Paap rk Wald F statistic, a generalization of the Cragg-Donald Wald F statistic for non i.i.d. errors. The statistic is similar to the one returned by the Stata command `ivreg2`.

### Examples
```julia
using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
df[:StatePooled] =  pool(df[:State])
df[:YearPooled] =  pool(df[:Year])
reg(df, @model(Sales ~ Price, fe = StatePooled + YearPooled))
reg(df, @model(Sales ~ NDI, fe = StatePooled + StatePooled&Year))
reg(df, @model(Sales ~ NDI, fe = StatePooled*Year))
reg(df, @model(Sales ~ (Price ~ Pimin)))
reg(df, @model(Sales ~ Price, weights = Pop))
reg(df, @model(Sales ~ NDI, subset = State .< 30))
reg(df, @model(Sales ~ NDI, vcov = robust))
reg(df, @model(Sales ~ NDI, vcov = cluster(StatePooled)))
reg(df, @model(Sales ~ NDI, vcov = cluster(StatePooled + YearPooled)))
```
"""



# TODO: minimize memory
function reg(df::AbstractDataFrame, m::Model)
    reg(df, m.f; m.dict...)
end

function reg(df::AbstractDataFrame, f::Formula; 
    fe::Union{Symbol, Expr, Void} = nothing, 
    vcov::Union{Symbol, Expr, Void} = :(simple()), 
    weights::Union{Symbol, Expr, Void} = nothing, 
    subset::Union{Symbol, Expr, Void} = nothing, 
    maxiter::Integer = 10000, tol::Real= 1e-8, df_add::Integer = 0, 
    save::Bool = false,
    method::Symbol = :lsmr)
    feformula = fe
    if isa(vcov, Symbol)
        vcovformula = VcovFormula(Val{vcov})
    else 
        vcovformula = VcovFormula(Val{vcov.args[1]}, (vcov.args[i] for i in 2:length(vcov.args))...)
    end


    ##############################################################################
    ##
    ## Parse formula
    ##
    ##############################################################################
    rf = deepcopy(f)
    (has_iv, iv_formula, iv_terms, endo_formula, endo_terms) = decompose_iv!(rf)
    rt = Terms(rf)
    has_absorb = feformula != nothing
    if has_absorb
        # check depth 1 symbols in original formula are all PooledDataArray
        if isa(feformula, Symbol)
            x = feformula
            !isa(df[x], PooledDataArray) && error("$x should be PooledDataArray")
        elseif feformula.args[1] == :+
            x = feformula.args
            for i in 2:length(x)
                isa(x[i], Symbol) && !isa(df[x[i]], PooledDataArray) && error("$(x[i]) should be PooledDataArray")
            end
        end
    end
    has_weights = (weights != nothing)


    ##############################################################################
    ##
    ## Construct new dataframe after removing missing values
    ##
    ##############################################################################

    # create a dataframe without missing values & negative weights
    vars = allvars(rf)
    iv_vars = allvars(iv_formula)
    endo_vars = allvars(endo_formula)
    absorb_vars = allvars(feformula)
    vcov_vars = allvars(vcovformula)

    # create a dataframe without missing values & negative weights
    all_vars = vcat(vars, vcov_vars, absorb_vars, endo_vars, iv_vars)
    all_vars = unique(convert(Vector{Symbol}, all_vars))
    esample = completecases(df[all_vars])

    if has_weights
        esample .&= isnaorneg(df[weights])
    end
    if subset != nothing
        subset = eval(evaluate_subset(df, subset))
        if length(subset) != size(df, 1)
            error("df has $(size(df, 1)) rows but the subset vector has $(length(subset)) elements")
        end
        esample .&= convert(BitArray, subset)
    end
    nobs = sum(esample)
    (nobs > 0) || error("sample is empty")

    # Compute weights
    sqrtw = get_weights(df, esample, weights)

    # remove unusused levels
    subdf = df[esample, all_vars]
    main_vars = unique(convert(Vector{Symbol}, vcat(vars, endo_vars, iv_vars)))
    for v in main_vars
        # in case subdataframe, don't construct subdf[v] if you dont need to do it
        if typeof(df[v]) <: PooledDataArray
            dropUnusedLevels!(subdf[v])
        end
    end

    # Compute pfe, a FixedEffectProblem
    has_intercept = rt.intercept
    if has_absorb
        fixedeffects = FixedEffect(subdf, feformula, sqrtw)
        # in case some FixedEffect does not have interaction, remove the intercept
        if any([typeof(f.interaction) <: Ones for f in fixedeffects]) 
            rt.intercept = false
            has_intercept = true
        end
        pfe = FixedEffectProblem(fixedeffects, Val{method})
    else
        pfe = nothing
    end


    # Compute data for std errors
    vcov_method_data = VcovMethod(subdf, vcovformula)


    ##############################################################################
    ##
    ## Dataframe --> Matrix
    ##
    ##############################################################################

    # initialize iterations and converged
    iterations = Int[]
    converged = Bool[]

    mf = ModelFrame2(rt, subdf, esample)

    # Obtain y
    py = model_response(mf)[:]
    if eltype(py) != Float64
        y = convert(Vector{Float64}, py)
    else
        y = py
    end
    yname = rt.eterms[1]
    y .= y .* sqrtw
    # old y will be used if fixed effects
    if has_absorb
        oldy = deepcopy(y)
    else
        oldy = y
    end
    residualize!(y, pfe, iterations, converged; maxiter = maxiter, tol = tol)


    # Obtain X
    coef_names = coefnames(mf)
    if isempty(mf.terms.terms) && mf.terms.intercept == false
        Xexo = Matrix{Float64}(sum(mf.msng), 0)
    else    
        Xexo = ModelMatrix(mf).m
        if size(Xexo, 2) == 1
            # See pull request #1017 in DataFrames Package
            Xexo = deepcopy(Xexo)
        end
    end
    Xexo .= Xexo .* sqrtw
    norm_Xexo =  sum(abs2, Xexo, 1)
    residualize!(Xexo, pfe, iterations, converged; maxiter = maxiter, tol = tol)

    
    # Obtain Xendo and Z
    if has_iv
        mf = ModelFrame2(endo_terms, subdf, esample)
        coef_names = vcat(coef_names, coefnames(mf))
        Xendo = ModelMatrix(mf).m
        Xendo .= Xendo .* sqrtw

        norm_Xendo =  sum(abs2, Xendo, 1)
        residualize!(Xendo, pfe, iterations, converged; maxiter = maxiter, tol = tol)
        
        mf = ModelFrame2(iv_terms, subdf, esample)
        Z = ModelMatrix(mf).m
        Z .= Z .* sqrtw
        norm_Z =  sum(abs2, Z, 1)
        residualize!(Z, pfe, iterations, converged; maxiter = maxiter, tol = tol)
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

    # Compute Xhat
    if has_iv
        if size(Z, 2) < size(Xendo, 2)
            error("Model not identified. There must be at least as many ivs as endogeneneous variables")
        end
        # get linearly independent columns
        # special case for variables demeaned by fixed effects
        baseall= basecol(Z, Xexo, Xendo)
        basecolZ = baseall[1:size(Z, 2)] .& vec(sum(abs2, Z, 1) .> tol * norm_Z)
        basecolXexo = baseall[(size(Z, 2)+1):(size(Z, 2) + size(Xexo, 2))] .& vec(sum(abs2, Xexo, 1) .> tol * norm_Xexo)
        basecolXendo = baseall[(size(Z, 2) + size(Xexo, 2) + 1):end] .& vec(sum(abs2, Xendo, 1) .> tol * norm_Xendo)
        Z = getcols(Z, basecolZ)
        Xexo = getcols(Xexo, basecolXexo)
        Xendo = getcols(Xendo, basecolXendo)
        basecoef = vcat(basecolXexo, basecolXendo)

        # Build
        X = hcat(Xexo, Xendo)
        newZ = hcat(Xexo, Z)
        crossz = cholfact!(At_mul_B(newZ, newZ))
        Pi = crossz \ At_mul_B(newZ, Xendo)
        Xhat = hcat(Xexo, newZ * Pi)
        X = hcat(Xexo, Xendo)

        # prepare residuals used for first stage F statistic
        ## partial out Xendo in place wrt (Xexo, Z)
        Xendo_res = BLAS.gemm!('N', 'N', -1.0, newZ, Pi, 1.0, Xendo)
        ## partial out Z in place wrt Xexo
        Pi2 = cholfact!(At_mul_B(Xexo, Xexo)) \ At_mul_B(Xexo, Z)
        Z_res = BLAS.gemm!('N', 'N', -1.0, Xexo, Pi2, 1.0, Z)

        # free memory (not sure it helps)
        Xexo = nothing
    else
        # get linearly independent columns
        # special case for variables demeaned by fixed effects
        basecolXexo = basecol(Xexo) .& vec(sum(abs2, Xexo, 1) .> tol * norm_Xexo)
        Xexo = getcols(Xexo, basecolXexo)
        Xhat = Xexo
        X = Xexo
        basecoef = basecolXexo
    end

    # Compute coef and residuals
    crossx =  cholfact!(At_mul_B(Xhat, Xhat))
    coef = crossx \ At_mul_B(Xhat, y)
    residuals = y - X * coef


    ##############################################################################
    ##
    ## Optionally save some vectors in a new dataframe
    ##
    ##############################################################################

    # save residuals in a new dataframe
    augmentdf = DataFrame()
    if save
        residuals .= residuals ./ sqrtw
        if all(esample)
            augmentdf[:residuals] = residuals
        else
            augmentdf[:residuals] =  DataArray(Float64, length(esample))
            augmentdf[esample, :residuals] = residuals
        end
        if has_absorb
            mf = ModelFrame2(rt, subdf, esample)
            oldX = ModelMatrix(mf).m
            if !all(basecoef)
                oldX = oldX[:, basecoef]
            end
            oldX .= oldX .* sqrtw
            BLAS.gemm!('N', 'N', -1.0, oldX, coef, 1.0, oldy)
            axpy!(-1.0, residuals, oldy)


            augmentdf = hcat(augmentdf, getfe!(pfe, oldy, esample; tol = tol, maxiter = maxiter))
        end
    end

    ##############################################################################
    ##
    ## Test Statistics
    ##
    ##############################################################################

    # Compute degrees of freedom
    df_intercept = 0
    if has_absorb || rt.intercept
        df_intercept = 1
    end
    df_absorb = 0
    if has_absorb 
        ## poor man adjustement of df for clustedered errors + fe: only if fe name != cluster name
        #for fe in fixedeffects
        #    if typeof(vcovformula) == VcovClusterFormula && in(fe.factorname, vcov_vars)
        #        df_absorb += 0
        #    else
        #        df_absorb += sum(fe.scale .!= zero(Float64))
        #    end
        #end
        ## better adjustment of df for clustered errors + fe: adjust only if fe is not fully nested in a cluster variable:
        for fe in fixedeffects
            if typeof(vcovformula) == VcovClusterFormula && any([isnested(fe.refs,vcov_method_data.clusters[clustervar].refs) for clustervar in vcov_vars])
                println("$(fe.factorname) is nested in one of the cluster variables")
                df_absorb += 0
            else
                println("$(fe.factorname) is not nested in one of the cluster variables")
                df_absorb += sum(fe.scale .!= zero(Float64))
            end
        end
    end
    nvars = size(X, 2)
    df_residual = max(1, nobs - nvars - df_absorb - df_add)

    # Compute ess, tss, r2, r2 adjusted
    ess = sum(abs2, residuals)
    if has_absorb
        tss = compute_tss(y, rt.intercept, sqrtw)
        r2_within = 1 - ess / tss 
    end
    tss = compute_tss(oldy, has_intercept, sqrtw)
    r2 = 1 - ess / tss 
    r2_a = 1 - ess / tss * (nobs - has_intercept) / df_residual 

    # Compute standard error
    vcov_data = VcovData(Xhat, crossx, residuals, df_residual)
    matrix_vcov = vcov!(vcov_method_data, vcov_data)

    # Compute Fstat
    (F, p) = compute_Fstat(coef, matrix_vcov, nobs, rt.intercept, vcov_method_data, vcov_data)

    # Compute Fstat of First Stage
    if has_iv
        Pip = Pi[(size(Pi, 1) - size(Z_res, 2) + 1):end, :]
        (F_kp, p_kp) = ranktest!(Xendo_res, Z_res, Pip, 
                                  vcov_method_data, nvars, df_absorb)
    end

    ##############################################################################
    ##
    ## Return regression result
    ##
    ##############################################################################

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
                                  coef_names, yname, f, feformula, nobs, df_residual, 
                                  r2, r2_a, r2_within, F, p, iterations, converged)
    elseif has_iv && has_absorb 
        return RegressionResultFEIV(coef, matrix_vcov, esample, augmentdf, 
                                   coef_names, yname, f, feformula, nobs, df_residual, 
                                   r2, r2_a, r2_within, F, p, F_kp, p_kp, 
                                   iterations, converged)
    end
end







##############################################################################
##
## Fstat
##
##############################################################################

function compute_Fstat(coef::Vector{Float64}, matrix_vcov::Matrix{Float64}, 
    nobs::Int, hasintercept::Bool, 
    vcov_method_data::AbstractVcovMethod, vcov_data::VcovData)
    coefF = deepcopy(coef)
    # TODO: check I can't do better
    length(coef) == hasintercept && return NaN, NaN
    if hasintercept && length(coef) > 1
        coefF = coefF[2:end]
        matrix_vcov = matrix_vcov[2:end, 2:end]
    end
    F = (diagm(coefF)' * (matrix_vcov \ diagm(coefF)))[1]
    df_ans = df_FStat(vcov_method_data, vcov_data, hasintercept)
    dist = FDist(nobs - hasintercept, max(df_ans, 1))
    return F, ccdf(dist, F)
end



function compute_tss(y::Vector{Float64}, hasintercept::Bool, ::Ones)
    if hasintercept
        tss = zero(Float64)
        m = mean(y)::Float64
        @inbounds @simd  for i in 1:length(y)
            tss += abs2((y[i] - m))
        end
    else
        tss = sum(abs2, y)
    end
    return tss
end

function compute_tss(y::Vector{Float64}, hasintercept::Bool, sqrtw::Vector{Float64})
    if hasintercept
        m = (mean(y) / sum(sqrtw) * length(y))::Float64
        tss = zero(Float64)
        @inbounds @simd  for i in 1:length(y)
            tss += abs2(y[i] - sqrtw[i] * m)
        end
    else
        tss = sum(abs2, y)
    end
    return tss
end


##############################################################################
##
## Syntax without keywords
##
##############################################################################
function evaluate_subset(df, ex::Expr)
    if ex.head == :call
        return Expr(ex.head, ex.args[1], (evaluate_subset(df, ex.args[i]) for i in 2:length(ex.args))...)
    else
        return Expr(ex.head, (evaluate_subset(df, ex.args[i]) for i in 1:length(ex.args))...)
    end
end
evaluate_subset(df, ex::Symbol) = df[ex]
evaluate_subset(df, ex)  = ex



