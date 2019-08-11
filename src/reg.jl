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
df.StateC =  categorical(df.State)
df.YearC =  categorical(df.Year)
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

function reg(df::AbstractDataFrame, f::FormulaTerm;
    fe::Union{Symbol, Expr, Nothing} = nothing,
    vcov::Union{Symbol, Expr, Nothing} = :(simple()),
    weights::Union{Symbol, Expr, Nothing} = nothing,
    subset::Union{Symbol, Expr, Nothing} = nothing,
    maxiter::Integer = 10000, contrasts::Dict = Dict{Symbol, Any}(),
    tol::Real= 1e-8, df_add::Integer = 0,
    save::Union{Bool, Symbol} = false,  method::Symbol = :lsmr, drop_singletons = true
   )

    if isa(vcov, Symbol)
        vcovformula = VcovFormula(Val{vcov})
    else
        vcovformula = VcovFormula(Val{vcov.args[1]}, (vcov.args[i] for i in 2:length(vcov.args))...)
    end

    if  (ConstantTerm(0) ∉ eachterm(f.rhs)) & (ConstantTerm(1) ∉ eachterm(f.rhs))
        f = FormulaTerm(f.lhs, tuple(ConstantTerm(1), eachterm(f.rhs)...))
    end

    ##############################################################################
    ##
    ## Parse formula
    ##
    ##############################################################################
    formula, formula_endo, formula_iv = decompose_iv(f)
    has_iv = formula_iv != nothing
    has_absorb = fe != nothing
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
    vars = allvars(formula)
    iv_vars = allvars(formula_iv)
    endo_vars = allvars(formula_endo)
    absorb_vars = allvars(fe)
    vcov_vars = allvars(vcovformula)
    # create a dataframe without missing values & negative weights
    all_vars = vcat(vars, vcov_vars, absorb_vars, endo_vars, iv_vars)
    all_vars = unique(Symbol.(all_vars))



    esample = completecases(df[!, all_vars])

    if has_weights
        esample .&= isnaorneg(df[!, weights])
    end
    if subset != nothing
        subset = eval(evaluate_subset(df, subset))
        if length(subset) != size(df, 1)
            error("df has $(size(df, 1)) rows but the subset vector has $(length(subset)) elements")
        end
        esample .&= convert(BitArray, subset)
    end

    if has_absorb
        feformula = @eval(@formula(nothing ~ $(fe)))
        fes, ids = parse_fixedeffect(df, feformula)
        if drop_singletons
            for fe in fes
                drop_singletons!(esample, fe)
            end
        end
    end

    nobs = sum(esample)
    (nobs > 0) || error("sample is empty")

    # Compute weights
    sqrtw = get_weights(df, esample, weights)


    # Compute pfe, a FixedEffectMatrix
    has_absorb_intercept = false
    if has_absorb
        # in case some FixedEffect does not have interaction, remove the intercept
        if any([isa(fe.interaction, Ones) for fe in fes])
            formula = FormulaTerm(formula.lhs, tuple(ConstantTerm(0), (t for t in eachterm(formula.rhs) if t!= ConstantTerm(1))...))
            has_absorb_intercept = true
        end
        fes = FixedEffect[_subset(fe, esample) for fe in fes]
        pfe = FixedEffectMatrix(fes, sqrtw, Val{method})
    end

    has_intercept = ConstantTerm(1) ∈ eachterm(formula.rhs)
    
    # Compute data for std errors
    vcov_method_data = VcovMethod(df[esample, unique(Symbol.(vcov_vars))], vcovformula)

    ##############################################################################
    ##
    ## Dataframe --> Matrix
    ##
    ##############################################################################
    subdf = columntable(df[esample, unique(vcat(vars, iv_vars, endo_vars))])

    formula_schema = apply_schema(formula, schema(formula, subdf, contrasts), StatisticalModel)
    # Obtain y
    # for a Vector{Float64}, conver(Vector{Float64}, y) aliases y
    y = convert(Vector{Float64}, response(formula_schema, subdf))
    y .= y .* sqrtw

    # Obtain X
    Xexo = convert(Matrix{Float64}, modelmatrix(formula_schema, subdf))
    Xexo .= Xexo .* sqrtw


    yname, coef_names = coefnames(formula_schema)
    if !isa(coef_names, Vector)
        coef_names = [coef_names]
    end
    yname = Symbol(yname)
    coef_names = Symbol.(coef_names)

    if has_iv
        formula_endo_schema = apply_schema(formula_endo, schema(formula_endo, subdf, contrasts), StatisticalModel)
        Xendo = convert(Matrix{Float64}, modelmatrix(formula_endo_schema, subdf))
        Xendo .= Xendo .* sqrtw
        coefendo_names = coefnames(formula_endo_schema)[2]
        if !isa(coefendo_names, Vector)
            coefendo_names = [coefendo_names]
        end
        append!(coef_names, Symbol.(coefendo_names))

        formula = FormulaTerm(formula.lhs, (tuple(eachterm(formula.rhs)..., eachterm(formula_endo.rhs)...)))
        formula_schema = apply_schema(formula, schema(formula, subdf, contrasts), StatisticalModel)


        formula_iv_schema = apply_schema(formula_iv, schema(formula_iv, subdf, contrasts),StatisticalModel)
        Z = convert(Matrix{Float64}, modelmatrix(formula_iv_schema, subdf))
        Z .= Z .* sqrtw
    else
        Xendo = Matrix{Float64}(undef, nobs, 0)
        Z = Matrix{Float64}(undef, nobs, 0)
    end

    # compute tss now before potentially demeaning y
    tss = compute_tss(y, has_intercept | has_absorb_intercept, sqrtw)

    if has_absorb
        # used to compute tss even without save_fe
        if save_fe
            oldy = deepcopy(y)
            oldX = hcat(Xexo, Xendo)
        end

        # initialize iterations and converged
        iterations = Int[]
        convergeds = Bool[]

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
    ## Optionally save objects in a new dataframe
    ##
    ##############################################################################

    augmentdf = DataFrame()
    if save_residuals
        augmentdf.residuals =  Vector{Union{Missing, Float64}}(missing, length(esample))
        augmentdf[esample, :residuals] = residuals ./ sqrtw
    end
    if save_fe
        oldX = getcols(oldX, basecoef)
        newfes, b, c = solve_coefficients!(oldy - oldX * coef, pfe; tol = tol, maxiter = maxiter)
        for j in 1:length(fes)
            augmentdf[!, ids[j]] = Vector{Union{Float64, Missing}}(missing, length(esample))
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
    if has_intercept | has_absorb_intercept
        df_intercept = 1
    end
    df_absorb = 0
    if has_absorb
        for fe in fes
            # adjust degree of freedom only if fe is not fully nested in a cluster variable:
            if isa(vcovformula, VcovClusterFormula)
                if any(isnested(fe, v) for v in eachcol(vcov_method_data.clusters))
                    df_absorb = 1 # if fe is nested you still lose 1 degree of freedom 
                    break
                end
            end
            #only count groups that exists
            df_absorb += length(Set(fe.refs))
        end
    end
    dof_residual = max(1, nobs - size(X, 2) - df_absorb - df_add)

    # Compute rss, tss, r2, r2 adjusted
    rss = sum(abs2, residuals)
    mss = tss - rss
    r2 = 1 - rss / tss
    adjr2 = 1 - rss / tss * (nobs - (has_intercept | has_absorb_intercept)) / dof_residual
    if has_absorb
        r2_within = 1 - rss / compute_tss(y, (has_intercept | has_absorb_intercept), sqrtw)
    end

    # Compute standard error
    vcov_data = VcovData(Xhat, crossx, residuals, dof_residual)
    matrix_vcov = vcov!(vcov_method_data, vcov_data)

    # Compute Fstat
    (F, p) = compute_Fstat(coef, matrix_vcov, nobs, has_intercept, vcov_method_data, vcov_data)

    # Compute Fstat of First Stage
    if has_iv
        Pip = Pi[(size(Pi, 1) - size(Z_res, 2) + 1):end, :]
        (F_kp, p_kp) = ranktest!(Xendo_res, Z_res, Pip,
                                  vcov_method_data, size(X, 2), df_absorb)
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
                                coef_names, yname, f, formula_schema, nobs, dof_residual,
                                rss, tss, r2, adjr2, F, p)
    elseif has_iv && !has_absorb
        return RegressionResultIV(coef, matrix_vcov, esample, augmentdf,
                                  coef_names, yname, f, formula_schema, nobs, dof_residual,
                                  rss, tss,  r2, adjr2, F, p, F_kp, p_kp)
    elseif !has_iv && has_absorb
        return RegressionResultFE(coef, matrix_vcov, esample, augmentdf,
                                  coef_names, yname, f, formula_schema, fe, nobs, dof_residual,
                                  rss, tss, r2, adjr2, r2_within, F, p, iterations, converged)
    elseif has_iv && has_absorb
        return RegressionResultFEIV(coef, matrix_vcov, esample, augmentdf,
                                   coef_names, yname, f, formula_schema, fe, nobs, dof_residual,
                                   rss, tss, r2, adjr2, r2_within, F, p, F_kp, p_kp,
                                   iterations, converged)
    end
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
evaluate_subset(df, ex::Symbol) = df[!, ex]
evaluate_subset(df, ex)  = ex
