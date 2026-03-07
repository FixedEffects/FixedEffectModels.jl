"""
Estimate a linear model with high dimensional categorical variables / instrumental variables

### Arguments
* `df`: a Table
* `FormulaTerm`: A formula created using [`@formula`](@ref)
* `CovarianceEstimator`: A method to compute the variance-covariance matrix

### Keyword arguments
* `contrasts::Dict = Dict()` An optional Dict of contrast codings for each categorical variable in the `formula`.  Any unspecified variables will have `DummyCoding`.
* `weights::Union{Nothing, Symbol}` A symbol to refer to a columns for weights
* `save::Symbol`: Should residuals and eventual estimated fixed effects saved in a dataframe? Default to `:none` Use `save = :residuals` to only save residuals, `save = :fe` to only save fixed effects, `save = :all` for both. Once saved, they can then be accessed using `residuals(m)` or `fe(m)` where `m` is the object returned by the estimation. The returned DataFrame is automatically aligned with the original DataFrame.
* `method::Symbol`: A symbol for the method. Default is :cpu. Alternatively,  use :CUDA or :Metal  (in this case, you need to import the respective package before importing FixedEffectModels)
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true if `method =:cpu' and false if `method = :CUDA` or `method = :Metal`.
* `tol::Real` Tolerance. Default to 1e-6.
* `maxiter::Integer = 10000`: Maximum number of iterations
* `drop_singletons::Bool = true`: Should singletons be dropped?
* `progress_bar::Bool = true`: Should the regression show a progressbar?
* `first_stage::Bool = true`: Should the first-stage F-stat and p-value be computed?
* `subset::Union{Nothing, AbstractVector} = nothing`: select specific rows. 


### Details
Models with instruments variables are estimated using 2SLS. `reg` tests for weak instruments by computing the Kleibergen-Paap rk Wald F statistic, a generalization of the Cragg-Donald Wald F statistic for non i.i.d. errors. The statistic is similar to the one returned by the Stata command `ivreg2`.

### Examples
```julia
using RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
reg(df, @formula(Sales ~ NDI + fe(State) + fe(State)&Year))
reg(df, @formula(Sales ~ NDI + fe(State)*Year))
reg(df, @formula(Sales ~ (Price ~ Pimin)))
reg(df, @formula(Sales ~ NDI), Vcov.robust())
reg(df, @formula(Sales ~ NDI), Vcov.cluster(:State))
reg(df, @formula(Sales ~ NDI), Vcov.cluster(:State , :Year))
df.YearC = categorical(df.Year)
reg(df, @formula(Sales ~ YearC), contrasts = Dict(:YearC => DummyCoding(base = 80)))
```

### Alias
`reg` is an alias for the more typical StatsAPI `fit`
```julia
using RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
fit(FixedEffectModel, @formula(Sales ~ NDI + fe(State) + fe(State)&Year), df)
```
"""
function reg(df,     
    formula::FormulaTerm,
    vcov::CovarianceEstimator = Vcov.simple();
    contrasts::Dict = Dict{Symbol, Any}(),
    weights::Union{Symbol, Nothing} = nothing,
    save::Union{Bool, Symbol} = :none,
    method::Symbol = :cpu,
    nthreads::Union{Integer, Nothing} = nothing,
    double_precision::Bool = method == :cpu,
    tol::Real = 1e-6,
    maxiter::Integer = 10000,
    drop_singletons::Bool = true,
    progress_bar::Bool = true,
    subset::Union{Nothing, AbstractVector} = nothing, 
    first_stage::Bool = true)
    StatsAPI.fit(FixedEffectModel, formula, df, vcov; contrasts = contrasts, weights = weights, save = save, method = method, double_precision = double_precision, tol = tol, maxiter = maxiter, drop_singletons = drop_singletons, progress_bar = progress_bar, subset = subset, first_stage = first_stage)
end
    
function StatsAPI.fit(::Type{FixedEffectModel},     
    @nospecialize(formula::FormulaTerm),
    @nospecialize(df),
    @nospecialize(vcov::CovarianceEstimator = Vcov.simple());
    @nospecialize(contrasts::Dict = Dict{Symbol, Any}()),
    @nospecialize(weights::Union{Symbol, Nothing} = nothing),
    @nospecialize(save::Union{Bool, Symbol} = :none),
    @nospecialize(method::Symbol = :cpu),
    @nospecialize(nthreads::Union{Integer, Nothing} = nothing),
    @nospecialize(double_precision::Bool = method == :cpu),
    @nospecialize(tol::Real = 1e-6),
    @nospecialize(maxiter::Integer = 10000),
    @nospecialize(drop_singletons::Bool = true),
    @nospecialize(progress_bar::Bool = true),
    @nospecialize(subset::Union{Nothing, AbstractVector} = nothing), 
    @nospecialize(first_stage::Bool = true))

    df = DataFrame(df; copycols = false)
    nrows = size(df, 1)

    #========================================================
    Keyword Arguments
    ========================================================#

    if method == :gpu
        @info "method = :gpu is deprecated. Use method = :CUDA or method = :Metal"
        method = :CUDA
    end
    if nthreads !== nothing
        @info "The keyword argument nthreads is deprecated. Multiple threads are now used by default."
    end
    if save == true
        save = :all
    elseif save == false
        save = :none
    end
    if save ∉ (:all, :residuals, :fe, :none)
            throw(ArgumentError("the save keyword argument must be a Symbol equal to :all, :none, :residuals or :fe"))
    end
    save_residuals = (save == :residuals) || (save == :all)

    #========================================================
    Parse formula
    ========================================================#

    formula_origin = formula
    if !omitsintercept(formula) && !hasintercept(formula)
        formula = FormulaTerm(formula.lhs, InterceptTerm{true}() + formula.rhs)
    end
    formula, formula_endo, formula_iv = parse_iv(formula)
    has_iv = formula_iv != FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    formula, formula_fes = parse_fe(formula)
    has_fes = formula_fes != FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    # when save = :fe but there are no fixed effects in the formula, don't save fixed effects
    save_fes = save ∈ (:fe, :all) && has_fes
    has_weights = weights !== nothing

    # Compute feM, an AbstractFixedEffectSolver
    fes, feids, fekeys = parse_fixedeffect(df, formula_fes)
    has_fe_intercept = any(fe.interaction isa UnitWeights for fe in fes)

    # remove intercept if absorbed by fixed effects
    if has_fe_intercept
        formula = FormulaTerm(formula.lhs, tuple(InterceptTerm{false}(), (term for term in eachterm(formula.rhs) if !isa(term, Union{ConstantTerm,InterceptTerm}))...))
    end
    has_intercept = hasintercept(formula)

    #========================================================
    Create boolean vector esample that is true for observations used in estimation
    ========================================================#

    # Collect all variable names needed to detect missing values and build model matrices
    exo_vars = unique(StatsModels.termvars(formula))
    iv_vars = unique(StatsModels.termvars(formula_iv))
    endo_vars = unique(StatsModels.termvars(formula_endo))
    fe_vars = unique(StatsModels.termvars(formula_fes))
    all_vars = unique(vcat(exo_vars, endo_vars, iv_vars, fe_vars))

    # Create esample that returns obs used in estimation
    esample = completecases(df, all_vars)
    if has_weights
        esample .&= BitArray(!ismissing(x) && (x > 0) for x in df[!, weights])
    end
    if subset !== nothing
        if length(subset) != nrows
            throw(DimensionMismatch("df has $(nrows) rows but the subset vector has $(length(subset)) elements"))
        end
        esample .&= BitArray(!ismissing(x) && x for x in subset)
    end
    esample .&= Vcov.completecases(df, vcov)

    n_singletons = 0
    if drop_singletons
        n_singletons = drop_singletons!(esample, fes)
    end

    nobs = sum(esample)
    (nobs > 0) || throw(ArgumentError("sample is empty"))
    # If all rows are used, replace BitVector with Colon() for faster indexing
    (nobs < nrows) || (esample = Colon())

    # Materialize weights, subdataframe, and vcov data on the estimation sample
    if has_weights
        weights = Weights(disallowmissing(view(df[!, weights], esample)))
    else
        weights = uweights(nobs)
    end
    subdf = DataFrame((; (x => disallowmissing(view(df[!, x], esample)) for x in all_vars)...))
    subfes = FixedEffect[fe[esample] for fe in fes]
    vcov_method = Vcov.materialize(view(df, esample, :), vcov)

    #========================================================
    Dataframe --> Matrix
    ========================================================#

    s = schema(formula, subdf, contrasts)
    
    formula_schema = apply_schema(formula, s, FixedEffectModel, has_fe_intercept)

    # for a Vector{Float64}, convert(Vector{Float64}, y) aliases y
    y = convert(Vector{Float64}, response(formula_schema, subdf))
    Xexo = convert(Matrix{Float64}, modelmatrix(formula_schema, subdf))
    response_name, coefnames_exo = coefnames(formula_schema)

    Xendo = Array{Float64}(undef, nobs, 0)
    Z = Array{Float64}(undef, nobs, 0)
    coefnames_endo = typeof(coefnames_exo)[]
    coefnames_iv = typeof(coefnames_exo)[]
    if has_iv
        formula_endo_schema = apply_schema(formula_endo, schema(formula_endo, subdf, contrasts), StatisticalModel)
        Xendo = convert(Matrix{Float64}, modelmatrix(formula_endo_schema, subdf))
        _, coefnames_endo = coefnames(formula_endo_schema)

        formula_iv_schema = apply_schema(formula_iv, schema(formula_iv, subdf, contrasts), StatisticalModel)
        _, coefnames_iv = coefnames(formula_iv_schema)
    
        Z = convert(Matrix{Float64}, modelmatrix(formula_iv_schema, subdf))

        # modify formula to use in predict
        formula_schema = FormulaTerm(formula_schema.lhs, MatrixTerm(tuple(eachterm(formula_schema.rhs)..., (term for term in eachterm(formula_endo_schema.rhs) if term != ConstantTerm(0))...)))
    end
    coef_names = vcat(coefnames_exo, coefnames_endo)
    # compute tss now before potentially demeaning y
    tss_total = tss(y, has_intercept || has_fe_intercept, weights)

    all(isfinite, weights) || throw(ArgumentError("Weights are not finite"))
    all(isfinite, y) || throw(ArgumentError("Some observations for the dependent variable are infinite"))
    all(isfinite, Xexo) || throw(ArgumentError("Some observations for the exogeneous variables are infinite"))
    all(isfinite, Xendo) || throw(ArgumentError("Some observations for the endogenous variables are infinite"))
    all(isfinite, Z) || throw(ArgumentError("Some observations for the instrumental variables are infinite"))

    iterations, converged = 0, true
    if has_fes
        # Save pre-demeaned data for FE recovery after estimation
        if save_fes
            oldy = copy(y)
            oldX = hcat(Xexo, Xendo)
        end

        cols = vcat(eachcol(y), eachcol(Xexo), eachcol(Xendo), eachcol(Z))
        colnames = vcat(response_name, coefnames_exo, coefnames_endo, coefnames_iv)
        # compute 2-norm (sum of squares) for each variable 
        # (to see if they are collinear with the fixed effects)
        sumsquares_pre = [sum(abs2, x) for x in cols]

        feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(subfes, weights, Val{method})

        # Partial out fixed effects from y, Xexo, Xendo, Z
        _, iterations, convergeds = solve_residuals!(cols, feM; maxiter = maxiter, tol = tol, progress_bar = progress_bar)

        # set variables that are likely to be collinear with the fixed effects to zero
        for i in 1:length(cols)
            if sum(abs2, cols[i]) < tol * sumsquares_pre[i]
                if i == 1
                    @info "Dependent variable $(colnames[1]) is probably perfectly explained by fixed effects."
                else
                    @info "RHS-variable $(colnames[i]) is collinear with the fixed effects."
                    # set to zero so that removed when taking basis
                    cols[i] .= 0.0
                end
            end
        end

        # convergence info
        iterations = maximum(iterations)
        converged = all(convergeds)
        if converged == false
            @info "Convergence not achieved in $(iterations) iterations; try increasing maxiter or decreasing tol."
        end
        tss_within = tss(y, has_intercept || has_fe_intercept, weights)
    end

    if has_weights
        y .= y .*  sqrt.(weights)
        Xexo .= Xexo .*  sqrt.(weights)
        Xendo .= Xendo .*  sqrt.(weights)
        Z .= Z .*  sqrt.(weights)
    end
    
    #========================================================
    Get Linearly Independent Components of Matrix + Create the Xhat matrix
    ========================================================#

    Xexo, Xendo, Z, X, Xhat, XhatXhat, basis_coef, perm, Xendo_res, Z_res, Pi = collinearity!(Xexo, Xendo, Z, has_intercept, has_iv, coefnames_endo)

    #========================================================
    Do the regression: solve Xhat'Xhat \ Xhat'y via sweep operator
    
    Build augmented matrix [Xhat'Xhat  Xhat'y; y'Xhat  0] and sweep on
    the first k diagonal entries. After sweeping, the top-right block gives
    coef = (Xhat'Xhat)^{-1} Xhat'y and the top-left block gives -(Xhat'Xhat)^{-1}.
    Uses pre-computed cross-products rather than X'y to avoid numerical issues
    (see https://github.com/FixedEffects/FixedEffectModels.jl/issues/249).
    ========================================================#

    Xy = Symmetric(hvcat(2, XhatXhat, Xhat'reshape(y, length(y), 1),
                         zeros(size(Xhat, 2))', [0.0]))
    invsym!(Xy; diagonal = 1:size(Xhat, 2))
    invXhatXhat = Symmetric(.- Xy[1:(end-1),1:(end-1)])
    coef = Xy[1:(end-1),end]

    #========================================================
    Test Statistics
    ========================================================#

    mul!(y, X, coef, -1.0, 1.0)
    residuals = y
    residuals2 = nothing
    if save_residuals
        residuals2 = Vector{Union{Float64, Missing}}(missing, nrows)
        if has_weights
            residuals2[esample] .= residuals ./ sqrt.(weights)
        else
            residuals2[esample] .= residuals
        end
    end

    # Compute degrees of freedom absorbed by fixed effects.
    # When an FE is nested within a cluster variable, it only absorbs 1 dof
    # (its mean is not identified separately from the cluster effect),
    # rather than the full number of groups.
    ngroups_fes = [nunique(fe) for fe in subfes]
    dof_fes = sum(ngroups_fes)
    if vcov isa Vcov.ClusterCovariance
        dof_fes = 0
        for i in 1:length(subfes)
            if any(isnested(subfes[i], v.groups) for v in values(vcov_method.clusters))
                dof_fes += 1
            else
                dof_fes += ngroups_fes[i]
            end
        end
    end

    # Compute standard error
    nclusters = vcov isa Vcov.ClusterCovariance ?  Vcov.nclusters(vcov_method) : nothing
    vcov_data = Vcov.VcovData(Xhat, XhatXhat, invXhatXhat, residuals, nobs - size(X, 2) - dof_fes)
    matrix_vcov = StatsAPI.vcov(vcov_data, vcov_method)
   
    # Compute Fstat
    F = Fstat(coef, matrix_vcov, has_intercept)
   
    # dof_ is the number of estimated coefficients beyond the intercept.
    dof_ = size(X, 2) - has_intercept
    dof_tstat_ = max(1, Vcov.dof_residual(vcov_data, vcov_method) - (has_intercept || has_fe_intercept))
    p = fdistccdf(dof_, dof_tstat_, F)
    
    # Compute Fstat of First Stage
    F_kp, p_kp = NaN, NaN
    if has_iv && first_stage
        Pip = Pi[(size(Pi, 1) - size(Z_res, 2) + 1):end, :]
        try 
            r_kp = Vcov.ranktest!(Xendo_res, Z_res, Pip,
                              vcov_method, size(X, 2), dof_fes)
            p_kp = chisqccdf(size(Z_res, 2) - size(Xendo_res, 2) + 1, r_kp)
            F_kp = r_kp / size(Z_res, 2)
        catch
            @info "ranktest failed ; first-stage statistics not estimated"
        end
    end

    rss = sum(abs2, residuals)
    r2_within = has_fes ? 1 - rss / tss_within : 1 - rss / tss_total

    #========================================================
    Return regression result
    ========================================================#

    # add omitted variables and reorder IV-reclassified variables
    coef, matrix_vcov = reinsert_omitted!(coef, matrix_vcov, basis_coef, perm)

    # Recover FE estimates by projecting (y - Xβ) onto the FE structure.
    # Uses pre-demeaned oldy/oldX (saved before weighting and demeaning).
    # coef has been expanded by reinsert_omitted!: omitted entries are zero
    # (so extra columns in oldX contribute nothing) and reclassified IV
    # variables are permuted back to the original column order of oldX.
    augmentdf = DataFrame()
    if save_fes
        newfes, b, c = solve_coefficients!(oldy - oldX * coef, feM; tol = tol, maxiter = maxiter)
        for fekey in fekeys
            augmentdf[!, fekey] = df[:, fekey]
        end
        for j in eachindex(subfes)
            augmentdf[!, feids[j]] = Vector{Union{Float64, Missing}}(missing, nrows)
            augmentdf[esample, feids[j]] = newfes[j]
        end
    end

    if esample == Colon()
        esample = trues(nrows)
    end

    return FixedEffectModel(coef, matrix_vcov, vcov, nclusters, esample, residuals2, augmentdf, fekeys, coef_names, response_name, formula_origin, formula_schema, contrasts, nobs, dof_, sum(ngroups_fes), dof_tstat_, rss, tss_total, F, p, iterations, converged, r2_within, F_kp, p_kp)
end
