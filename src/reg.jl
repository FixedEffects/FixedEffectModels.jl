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
Categorical variable should be of type CategoricalVector.  Use the function `categorical` to create CategoricalVector.
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
        # check depth 1 symbols in original formula are all CategoricalVector
        if isa(feformula, Symbol)
            x = feformula
            !isa(df[x], CategoricalVector) && error("$x should be CategoricalVector")
        elseif feformula.args[1] == :+
            x = feformula.args
            for i in 2:length(x)
                isa(x[i], Symbol) && !isa(df[x[i]], CategoricalVector) && error("$(x[i]) should be CategoricalVector")
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
    if has_absorb
        # remove singletons
        remove_singletons!(esample, df, feformula)
    end
    nobs = sum(esample)
    (nobs > 0) || error("sample is empty")

    # Compute weights
    sqrtw = get_weights(df, esample, weights)

    # Compute pfe, a FixedEffectProblem
    has_intercept = rt.intercept
    if has_absorb
        # slow in 0.6 due to any. Is it improved in 0.7?
        subdf = df[esample, unique(convert(Vector{Symbol}, absorb_vars))]
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
    vcov_method_data = VcovMethod(df[esample, unique(convert(Vector{Symbol}, vcov_vars))], vcovformula)

    ##############################################################################
    ##
    ## Dataframe --> Matrix
    ##
    ##############################################################################

    # initialize iterations and converged
    iterations = Int[]
    converged = Bool[]


    mf = ModelFrame2(rt, df, esample)

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
    end
    Xexo .= Xexo .* sqrtw
    if save & has_absorb
        oldX = deepcopy(Xexo)
    end
    residualize!(Xexo, pfe, iterations, converged; maxiter = maxiter, tol = tol)

    
    # Obtain Xendo and Z
    if has_iv
        mf = ModelFrame2(endo_terms, df, esample)
        coef_names = vcat(coef_names, coefnames(mf))
        Xendo = ModelMatrix(mf).m
        Xendo .= Xendo .* sqrtw
        if save & has_absorb
            oldX = hcat(Xexo, Xendo)
        end
        residualize!(Xendo, pfe, iterations, converged; maxiter = maxiter, tol = tol)
        
        mf = ModelFrame2(iv_terms, df, esample)
        Z = ModelMatrix(mf).m
        Z .= Z .* sqrtw
        residualize!(Z, pfe, iterations, converged; maxiter = maxiter, tol = tol)   
    end

    # iter and convergence
    if has_absorb
        iterations = maximum(iterations)
        converged = all(converged)
        if converged == false
            error("convergence not achieved in $(iterations) iterations; try increasing maxiter or decreasing tol.")
        end
    end


    ##############################################################################
    ##
    ## Get Linearly Independent Components of Matrix
    ##
    ##############################################################################

    # Compute Xhat
    if has_iv
        if size(Z, 2) < size(Xendo, 2)
            error("Model not identified. There must be at least as many ivs as endogeneneous variables")
        end
        # get linearly independent columns
        baseall = basecol(Z, Xexo, Xendo)
        basecolXexo = baseall[(size(Z, 2)+1):(size(Z, 2) + size(Xexo, 2))]
        basecolXendo = baseall[(size(Z, 2) + size(Xexo, 2) + 1):end]
        Z = getcols(Z, baseall[1:size(Z, 2)])
        Xexo = getcols(Xexo, basecolXexo)
        Xendo = getcols(Xendo, basecolXendo)
        basecoef = vcat(basecolXexo, basecolXendo)
        # Build
        X = hcat(Xexo, Xendo)
        newZ = hcat(Xexo, Z)
        crossz = cholfact!(At_mul_B(newZ, newZ))
        Pi = crossz \ At_mul_B(newZ, Xendo)
        Xhat = hcat(Xexo, newZ * Pi)


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
        basecolXexo = basecol(Xexo)
        Xexo = getcols(Xexo, basecolXexo)
        Xhat = Xexo
        X = Xexo
        basecoef = basecolXexo
    end


    ##############################################################################
    ##
    ## Do the regression
    ##
    ##############################################################################

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
        if all(esample)
            augmentdf[:residuals] = residuals ./ sqrtw
        else
            augmentdf[:residuals] =  DataArray(Float64, length(esample))
            augmentdf[esample, :residuals] = residuals ./ sqrtw 
        end
        if has_absorb
            if !all(basecoef)
                oldX = oldX[:, basecoef]
            end
            augmentdf = hcat(augmentdf, getfe!(pfe, oldy - oldX * coef, esample; tol = tol, maxiter = maxiter))
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
        # better adjustment of df for clustered errors + fe: adjust only if fe is not fully nested in a cluster variable:
        for fe in fixedeffects
            if typeof(vcovformula) == VcovClusterFormula && any([isnested(fe.refs,vcov_method_data.clusters[clustervar].refs) for clustervar in names(vcov_method_data.clusters)])
                df_absorb += 0
            else
                #only count groups that exists
                df_absorb += sum(fe.scale .!= zero(Float64))
            end
        end
    end
    nvars = size(X, 2)
    df_residual = max(1, nobs - nvars - df_absorb - df_add)

    # Compute ess, tss, r2, r2 adjusted
    ess = sumabs2_precision(residuals)
    if has_absorb
        tss = compute_tss(y, rt.intercept, sqrtw)
        r2_within = convert(Float64, 1 - ess / tss)
    end
    tss = compute_tss(oldy, has_intercept, sqrtw)
    r2 = convert(Float64, 1 - ess / tss)
    r2_a = convert(Float64, 1 - ess / tss * (nobs - has_intercept) / df_residual)

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


function sumabs2_precision(y)
    out = zero(BigFloat)
    @inbounds @simd for i in 1:length(y)
        out += abs2(convert(BigFloat, y[i]))
    end
    return out
end

function compute_tss(y::Vector{Float64}, hasintercept::Bool, ::Ones)
    if hasintercept
        tss = zero(BigFloat)
        m = convert(BigFloat, mean(y))
        @inbounds @simd for i in 1:length(y)
            tss += abs2(convert(BigFloat,y[i]) - m)
        end
    else
        tss = sumabs2_precision(y)
    end
    return tss
end

function compute_tss(y::Vector{Float64}, hasintercept::Bool, sqrtw::Vector{Float64})
    if hasintercept
        m = convert(BigFloat, (mean(y) / sum(sqrtw) * length(y)))
        tss = zero(BigFloat)
        @inbounds @simd for i in 1:length(y)
            tss += abs2(convert(BigFloat, y[i]) - convert(BigFloat, sqrtw[i]) * m)
        end
    else
        tss = sumabs2_precision(y)
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


##############################################################################
##
## Remove singletons
##
##############################################################################
function remove_singletons!(esample, df, feformula)
    for term in Terms(@eval(@formula(nothing ~ $(feformula)))).terms
        if isa(term, Symbol) && isa(df[term], CategoricalVector)
            remove_singletons!(esample, df[term])
        end
    end
end

function remove_singletons!(esample, v)
    cache = zeros(Int, length(v.pool))
    for i in 1:length(esample)
        if esample[i]
            cache[v.refs[i]] += 1
        end
    end
    for i in 1:length(esample)
        if esample[i] && cache[v.refs[i]] <= 1
            esample[i] = false
        end
    end
end



