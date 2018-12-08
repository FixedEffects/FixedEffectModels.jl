"""
Estimate a linear model with high dimensional categorical variables / instrumental variables

### Arguments
* `df::AbstractDataFrame`
* `model::Model`: A model created using [`@model`](@ref)
* `save::Union{Bool, Symbol} = false`: Should residuals and eventual estimated fixed effects saved in a dataframe? Use `save = :residuals` to only save residuals. Use `save = :fe` to only save fixed effects.
* `method::Symbol = :lsmr`: Method to deman regressors. `:lsmr` is akin to conjugate gradient descent.  With parallel use `:lsmr_parallel`. To use multi threaded use `lsmr_threads`. Other choices are `:qr` and `:cholesky` (factorization methods)
* `contrasts::Dict = Dict()` An optional Dict of contrast codings for each categorical variable in the `formula`.  Any unspecified variables will have `DummyCoding`.
* `maxiter::Integer = 10000`: Maximum number of iterations
* `tol::Real =1e-8`: Tolerance


### Details
Models with instruments variables are estimated using 2SLS. `reg` tests for weak instruments by computing the Kleibergen-Paap rk Wald F statistic, a generalization of the Cragg-Donald Wald F statistic for non i.i.d. errors. The statistic is similar to the one returned by the Stata command `ivreg2`.

### Examples
```julia
using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
df[:StateC] =  categorical(df[:State])
df[:YearC] =  categorical(df[:Year])
reg(df, @model(Sales ~ Price, fe = StateC + YearC))
reg(df, @model(Sales ~ NDI, fe = StateC + StateC&Year))
reg(df, @model(Sales ~ NDI, fe = StateC*Year))
reg(df, @model(Sales ~ (Price ~ Pimin)))
reg(df, @model(Sales ~ Price, weights = Pop))
reg(df, @model(Sales ~ NDI, subset = State .< 30))
reg(df, @model(Sales ~ NDI, vcov = robust))
reg(df, @model(Sales ~ NDI, vcov = cluster(StateC)))
reg(df, @model(Sales ~ NDI, vcov = cluster(StateC + YearC)))
reg(df, @model(Sales ~ YearC), contrasts = Dict(:YearC => DummyCoding(base = 80)))
```
"""
function reg(df::AbstractDataFrame, m::Model; kwargs...)
    reg(df, m.f; m.dict..., kwargs...)
end

function reg(df::AbstractDataFrame, f::Formula; 
    fe::Union{Symbol, Expr, Nothing} = nothing, 
    vcov::Union{Symbol, Expr, Nothing} = :(simple()), 
    weights::Union{Symbol, Expr, Nothing} = nothing, 
    subset::Union{Symbol, Expr, Nothing} = nothing, 
    maxiter::Integer = 10000, contrasts::Dict = Dict(), 
    tol::Real= 1e-8, df_add::Integer = 0, 
    save::Union{Bool, Symbol} = false,  method::Symbol = :lsmr
   )
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
    ## Save keyword argument
    ##
    ##############################################################################
    if !isa(save, Bool)
        if save ∉ (:residuals, :fe)
            error("the save keyword argument must be a Bool or a Symbol equal to :residuals or :fe")
        end
    end
    save_residuals = (save == :residuals) | (save == true)
    save_fe = (save == :fe) | ((save == true) & has_absorb)


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
    all_vars = unique(Symbol.(all_vars))


    
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
        # slow in 0.6 due to any. Is it improved in 0.7?
        fes, ids = parse_fixedeffect(df, Terms(@eval(@formula(nothing ~ $(feformula)))))
        for fe in fes
            remove_singletons!(esample, fe)
        end
    end

    nobs = sum(esample)
    (nobs > 0) || error("sample is empty")

    # Compute weights
    sqrtw = get_weights(df, esample, weights)

    # Compute pfe, a FixedEffectMatrix
    has_intercept = rt.intercept
    if has_absorb
        # in case some FixedEffect does not have interaction, remove the intercept
        if any([isa(fe.interaction, Ones) for fe in fes]) 
            rt.intercept = false
            has_intercept = true
        end
        fes = FixedEffect[_subset(fe, esample) for fe in fes]
        pfe = FixedEffectMatrix(fes, sqrtw, Val{method})
    else
        pfe = nothing
    end


    # Compute data for std errors
    vcov_method_data = VcovMethod(df[esample, unique(Symbol.(vcov_vars))], vcovformula)

    ##############################################################################
    ##
    ## Dataframe --> Matrix
    ##
    ##############################################################################

    # initialize iterations and converged
    iterations = Int[]
    convergeds = Bool[]


    mf = ModelFrame2(rt, df, esample; contrasts = contrasts)

    # Obtain y
    # for a Vector{Float64}, conver(Vector{Float64}, y) aliases y
    y = convert(Vector{Float64}, model_response(mf)[:])
    yname = rt.eterms[1]
    y .= y .* sqrtw

    # Obtain X
    coef_names = coefnames(mf)
    has_exo = !isempty(mf.terms.terms) | mf.terms.intercept
    if has_exo
        Xexo = ModelMatrix(mf).m
        Xexo .= Xexo .* sqrtw
    else
        Xexo = Matrix{Float64}(undef, nobs, 0)
    end

    if has_iv
        mf = ModelFrame2(endo_terms, df, esample)
        coef_names = vcat(coef_names, coefnames(mf))
        Xendo = ModelMatrix(mf).m
        Xendo .= Xendo .* sqrtw

        mf = ModelFrame2(iv_terms, df, esample)
        Z = ModelMatrix(mf).m
        Z .= Z .* sqrtw
    else
        Xendo = Matrix{Float64}(undef, nobs, 0)
        Z = Matrix{Float64}(undef, nobs, 0)
    end

    # compute tss now before potentially demeaning y
    tss = compute_tss(y, has_intercept, sqrtw)

    if has_absorb
        # used to compute tss even without save_fe
        if save_fe
            oldy = deepcopy(y)
            oldX = hcat(Xexo, Xendo)
        end

        y, b, c = solve_residuals!(y, pfe; maxiter = maxiter, tol = tol)
        append!(iterations, b)
        append!(convergeds, c)

        Xexo, b, c = solve_residuals!(Xexo, pfe; maxiter = maxiter, tol = tol)
        append!(iterations, b)
        append!(convergeds, c)

        Xendo, b, c = solve_residuals!(Xendo, pfe; maxiter = maxiter, tol = tol)
        append!(iterations, b)
        append!(convergeds, c)

        Z, b, c = solve_residuals!(Z, pfe; maxiter = maxiter, tol = tol)
        append!(iterations, b)
        append!(convergeds, c)

        iterations = maximum(iterations)
        converged = all(convergeds)
        if converged == false
            @warn "convergence not achieved in $(iterations) iterations; try increasing maxiter or decreasing tol."
        end
    end


    ##############################################################################
    ##
    ## Get Linearly Independent Components of Matrix
    ##
    ##############################################################################

    # Compute linearly independent columns + create the Xhat matrix
    if has_iv
        if size(Z, 2) < size(Xendo, 2)
            error("Model not identified. There must be at least as many ivs as endogeneneous variables")
        end
        # get linearly independent columns
        # note that I do it after residualizing
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
        crossz = cholesky!(Symmetric(newZ' * newZ))
        Pi = crossz \ (newZ' * Xendo)
        Xhat = hcat(Xexo, newZ * Pi)


        # prepare residuals used for first stage F statistic
        ## partial out Xendo in place wrt (Xexo, Z)
        Xendo_res = gemm!('N', 'N', -1.0, newZ, Pi, 1.0, Xendo)
        ## partial out Z in place wrt Xexo
        Pi2 = cholesky!(Symmetric(Xexo' * Xexo)) \ (Xexo' * Z)
        Z_res = gemm!('N', 'N', -1.0, Xexo, Pi2, 1.0, Z)

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

    crossx =  cholesky!(Symmetric(Xhat' * Xhat))
    coef = crossx \ (Xhat' * y)
    residuals = y - X * coef

    ##############################################################################
    ##
    ## Optionally save some vectors in a new dataframe
    ##
    ##############################################################################

    # save residuals in a new dataframe
    augmentdf = DataFrame()
    if save_residuals
        augmentdf[:residuals] =  Vector{Union{Missing, Float64}}(missing, length(esample))
        augmentdf[esample, :residuals] = residuals ./ sqrtw 
    end
    if save_fe
        if !all(basecoef)
            oldX = oldX[:, basecoef]
        end
        newfes, b, c = solve_coefficients!(oldy - oldX * coef, pfe; tol = tol, maxiter = maxiter)
        for j in 1:length(fes)
            augmentdf[ids[j]] = Vector{Union{Float64, Missing}}(missing, length(esample))
            augmentdf[esample, ids[j]] = newfes[j]
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
        for fe in fes
            # adjust degree of freedom only if fe is not fully nested in a cluster variable:
            if isa(vcovformula, VcovClusterFormula)
                if any(isnested(fe, vcov_method_data.clusters[v]) for v in names(vcov_method_data.clusters))
                    break
                end
            end
            #only count groups that exists
            df_absorb += length(Set(fe.refs))
        end
    end
    nvars = size(X, 2)
    dof_residual = max(1, nobs - nvars - df_absorb - df_add)

    # Compute rss, tss, r2, r2 adjusted
    rss = sum(abs2, residuals)
    mss = tss - rss
    r2 = 1 - rss / tss
    adjr2 = 1 - rss / tss * (nobs - has_intercept) / dof_residual
    if has_absorb
        r2_within = 1 - rss / compute_tss(y, rt.intercept, sqrtw)
    end

    # Compute standard error
    vcov_data = VcovData(Xhat, crossx, residuals, dof_residual)
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
                                coef_names, yname, f, nobs, dof_residual, 
                                rss, tss, r2, adjr2, F, p)
    elseif has_iv && !has_absorb
        return RegressionResultIV(coef, matrix_vcov, esample, augmentdf, 
                                  coef_names, yname, f, nobs, dof_residual,
                                  rss, tss,  r2, adjr2, F, p, F_kp, p_kp)
    elseif !has_iv && has_absorb
        return RegressionResultFE(coef, matrix_vcov, esample, augmentdf, 
                                  coef_names, yname, f, feformula, nobs, dof_residual, 
                                  rss, tss, r2, adjr2, r2_within, F, p, iterations, converged)
    elseif has_iv && has_absorb 
        return RegressionResultFEIV(coef, matrix_vcov, esample, augmentdf, 
                                   coef_names, yname, f, feformula, nobs, dof_residual, 
                                   rss, tss, r2, adjr2, r2_within, F, p, F_kp, p_kp, 
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
    coefF = copy(coef)
    # TODO: check I can't do better
    length(coef) == hasintercept && return NaN, NaN
    if hasintercept && length(coef) > 1
        coefF = coefF[2:end]
        matrix_vcov = matrix_vcov[2:end, 2:end]
    end
    F = (Diagonal(coefF)' * (matrix_vcov \ Diagonal(coefF)))[1]
    df_ans = df_FStat(vcov_method_data, vcov_data, hasintercept)
    dist = FDist(nobs - hasintercept, max(df_ans, 1))
    return F, ccdf(dist, F)
end

function compute_tss(y::Vector{Float64}, hasintercept::Bool, ::Ones)
    if hasintercept
        tss = zero(Float64)
        m = mean(y)::Float64
        @inbounds @simd for i in 1:length(y)
            tss += abs2(y[i] - m)
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
        @inbounds @simd for i in 1:length(y)
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



