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
* `nthreads::Integer` Number of threads to use in the estimation. If `method = :cpu`, defaults to `Threads.nthreads()`. Otherwise, defaults to 256.
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
    nthreads::Integer = method == :cpu ? Threads.nthreads() : 256,
    double_precision::Bool = method == :cpu,
    tol::Real = 1e-6,
    maxiter::Integer = 10000,
    drop_singletons::Bool = true,
    progress_bar::Bool = true,
    dof_add::Integer = 0,
    subset::Union{Nothing, AbstractVector} = nothing, 
    first_stage::Bool = true)
    StatsAPI.fit(FixedEffectModel, formula, df, vcov; contrasts = contrasts, weights = weights, save = save, method = method, nthreads = nthreads, double_precision = double_precision, tol = tol, maxiter = maxiter, drop_singletons = drop_singletons, progress_bar = progress_bar, dof_add = dof_add, subset = subset, first_stage = first_stage)
end
    
function StatsAPI.fit(::Type{FixedEffectModel},     
    @nospecialize(formula::FormulaTerm),
    @nospecialize(df),
    @nospecialize(vcov::CovarianceEstimator = Vcov.simple());
    @nospecialize(contrasts::Dict = Dict{Symbol, Any}()),
    @nospecialize(weights::Union{Symbol, Nothing} = nothing),
    @nospecialize(save::Union{Bool, Symbol} = :none),
    @nospecialize(method::Symbol = :cpu),
    @nospecialize(nthreads::Integer = method == :cpu ? Threads.nthreads() : 256),
    @nospecialize(double_precision::Bool = true),
    @nospecialize(tol::Real = 1e-6),
    @nospecialize(maxiter::Integer = 10000),
    @nospecialize(drop_singletons::Bool = true),
    @nospecialize(progress_bar::Bool = true),
    @nospecialize(dof_add::Integer = 0),
    @nospecialize(subset::Union{Nothing, AbstractVector} = nothing), 
    @nospecialize(first_stage::Bool = true))

    df = DataFrame(df; copycols = false)
    nrows = size(df, 1)

    ##############################################################################
    ##
    ## Keyword Arguments
    ##
    ##############################################################################

    if method == :gpu
        info("method = :gpu is deprecated. Use method = :CUDA or method = :Metal")
        method = :CUDA
    end

    if save == true
        save = :all
    elseif save == false
        save = :none
    end
    if save ∉ (:all, :residuals, :fe, :none)
            throw("the save keyword argument must be a Symbol equal to :all, :none, :residuals or :fe")
    end
    save_residuals = (save == :residuals) | (save == :all)

    if method == :cpu && nthreads > Threads.nthreads()
        @warn "Keyword argument nthreads = $(nthreads) is ignored (Julia was started with only $(Threads.nthreads()) threads)."
        nthreads = Threads.nthreads()
    end

    ##############################################################################
    ##
    ## Parse formula
    ##
    ##############################################################################

    formula_origin = formula
    if !omitsintercept(formula) & !hasintercept(formula)
        formula = FormulaTerm(formula.lhs, InterceptTerm{true}() + formula.rhs)
    end
    formula, formula_endo, formula_iv = parse_iv(formula)
    has_iv = formula_iv != FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    formula, formula_fes = parse_fe(formula)
    has_fes = formula_fes != FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    save_fes = (save == :fe) | ((save == :all) & has_fes)
    has_weights = weights !== nothing

    # Compute feM, an AbstractFixedEffectSolver
    fes, feids, fekeys = parse_fixedeffect(df, formula_fes)

    # Change Intercept of Formula
    has_fe_intercept = any(fe.interaction isa UnitWeights for fe in fes)
    # remove intercept if absrobed by fixed effects
    if has_fe_intercept
        formula = FormulaTerm(formula.lhs, tuple(InterceptTerm{false}(), (term for term in eachterm(formula.rhs) if !isa(term, Union{ConstantTerm,InterceptTerm}))...))
    end
    has_intercept = hasintercept(formula)
    ##############################################################################
    ##
    ## Construct new dataframe after removing missing values & negative weights
    ##
    ##############################################################################

    # Get all variables
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
            throw("df has $(nrows) rows but the subset vector has $(length(subset)) elements")
        end
        esample .&= BitArray(!ismissing(x) && x for x in subset)
    end
    esample .&= Vcov.completecases(df, vcov)
    #TODO: add tests + return n_singletons
    n_singletons = 0
    if drop_singletons
        n_singletons = drop_singletons!(esample, fes)
    end
    nobs = sum(esample)
    (nobs > 0) || throw("sample is empty")
    (nobs < nrows) || (esample = Colon())

    # Compute weights
    if has_weights
        weights = Weights(convert(Vector{Float64}, view(df, esample, weights)))
        all(isfinite, weights) || throw("Weights are not finite")
    else
        weights = uweights(nobs)
    end
    if has_fes
        fes = FixedEffect[fe[esample] for fe in fes]
        feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, weights, Val{method}, nthreads)
    end
    # Compute data for std errors
    vcov_method = Vcov.materialize(view(df, esample, :), vcov)
    ##############################################################################
    ##
    ## Dataframe --> Matrix
    ##
    ##############################################################################
    subdf = Tables.columntable((; (x => disallowmissing(view(df[!, x], esample)) for x in exo_vars)...))
    s = schema(formula, subdf, contrasts)
    formula_schema = apply_schema(formula, s, FixedEffectModel, has_fe_intercept)

    # Obtain y
    # for a Vector{Float64}, convert(Vector{Float64}, y) aliases y
    y = convert(Vector{Float64}, response(formula_schema, subdf))
    all(isfinite, y) || throw("Some observations for the dependent variable are infinite")

    # Obtain X
    Xexo = convert(Matrix{Float64}, modelmatrix(formula_schema, subdf))
    all(isfinite, Xexo) || throw("Some observations for the exogeneous variables are infinite")

    response_name, coef_names = coefnames(formula_schema)
    if !(coef_names isa Vector)
        coef_names = typeof(coef_names)[coef_names]
    end

    # collect all variable names (outcome, exo [, endo, iv])
    var_names_all = vcat(response_name, coef_names)

    if has_iv
        subdf = Tables.columntable((; (x => disallowmissing(view(df[!, x], esample)) for x in endo_vars)...))
        formula_endo_schema = apply_schema(formula_endo, schema(formula_endo, subdf, contrasts), StatisticalModel)
        Xendo = convert(Matrix{Float64}, modelmatrix(formula_endo_schema, subdf))
        all(isfinite, Xendo) || throw("Some observations for the endogenous variables are infinite")
        _, coefendo_names = coefnames(formula_endo_schema)
        append!(coef_names, coefendo_names)
        append!(var_names_all, coefendo_names)

        subdf = Tables.columntable((; (x => disallowmissing(view(df[!, x], esample)) for x in iv_vars)...))
        formula_iv_schema = apply_schema(formula_iv, schema(formula_iv, subdf, contrasts), StatisticalModel)
        _, coefnames_iv = coefnames(formula_iv_schema)
        append!(var_names_all, coefnames_iv)
    
        Z = convert(Matrix{Float64}, modelmatrix(formula_iv_schema, subdf))
        all(isfinite, Z) || throw("Some observations for the instrumental variables are infinite")

        # modify formula to use in predict
        formula_schema = FormulaTerm(formula_schema.lhs, MatrixTerm(tuple(eachterm(formula_schema.rhs)..., (term for term in eachterm(formula_endo_schema.rhs) if term != ConstantTerm(0))...)))
    end

    # compute tss now before potentially demeaning y
    tss_total = tss(y, has_intercept | has_fe_intercept, weights)
    # initalize fields
    iterations, converged, r2_within = 0, true, nothing
    F_kp, p_kp = nothing, nothing
    collinear_fe = falses(length(var_names_all))

    if has_fes
        # used to compute tss even without save_fes
        if save_fes
            oldy = deepcopy(y)
            if has_iv
                oldX = hcat(Xexo, Xendo)
            else
                oldX = deepcopy(Xexo)
            end
        end

        # initialize iterations and converged
        iterations = Int[]
        convergeds = Bool[]
        cols = vcat(eachcol(y), eachcol(Xexo))
        if has_iv
            append!(cols, eachcol(Xendo), eachcol(Z))
        end

        # compute 2-norm (sum of squares) for each variable 
        # (to see if they are collinear with the fixed effects)
        sumsquares_pre = [sum(abs2, x) for x in cols]

        # partial out fixed effects
        _, iterations, convergeds = solve_residuals!(cols, feM; maxiter = maxiter, tol = tol, progress_bar = progress_bar)

        # re-compute 2-norm (sum of squares) for each variable
        sumsquares_post = [sum(abs2, x) for x in cols]

        # mark variables that are likely to be collinear with the fixed effects
        collinear_tol = min(1e-6, tol / 10)
        collinear_fe = (sumsquares_post ./ sumsquares_pre) .< tol
        for i in findall(collinear_fe)
            if i == 1
                @info "Dependent variable $(var_names_all[1]) is probably perfectly explained by fixed effects (tol = $collinear_tol)."
            else
                @info "RHS-variable $(var_names_all[i]) is collinear with the fixed effects (tol = $collinear_tol)."
                # set to zero so that removed when taking basis
                cols[i] .= 0.0
            end
        end

        # convergence info
        iterations = maximum(iterations)
        converged = all(convergeds)
        if converged == false
            @info "Convergence not achieved in $(iterations) iterations; try increasing maxiter or decreasing tol."
        end

        tss_partial = tss(y, has_intercept | has_fe_intercept, weights)
    end

    if has_weights
        sqrtw = sqrt.(weights)
        y .= y .* sqrtw
        Xexo .= Xexo .* sqrtw
        if has_iv
            Xendo .= Xendo .* sqrtw
            Z .= Z .* sqrtw
        end
    end
    
    ##############################################################################
    ##
    ## Get Linearly Independent Components of Matrix
    ##
    ##############################################################################
    # Compute linearly independent columns + create the Xhat matrix
    if has_iv    	
        perm = 1:(size(Xexo, 2) +size(Xendo, 2))

        # first pass: remove collinear variables in Xendo
        XendoXendo = Xendo' * Xendo
    	basis_endo = basis!(Symmetric(deepcopy(XendoXendo)); has_intercept = false)
        if !all(basis_endo)
        	Xendo = Xendo[:, basis_endo]
            XendoXendo = XendoXendo[basis_endo, basis_endo]
        end

    	# second pass: remove collinear variable in Xexo, Z, and Xendo
        XexoXexo = Xexo'Xexo
        XexoZ = Xexo'Z
        XexoXendo = Xexo'Xendo
        ZZ = Z'Z
        ZXendo = Z'Xendo
        XexoZXendo = Symmetric(hvcat(3, XexoXexo, XexoZ, XexoXendo, 
                           zeros(size(Z, 2), size(Xexo, 2)), ZZ, ZXendo, 
                           zeros(size(Xendo, 2), size(Xexo, 2)), zeros(size(Xendo, 2), size(Z, 2)), XendoXendo))
    	basis_all = basis!(XexoZXendo; has_intercept = has_intercept)
        basis_Xexo, basis_Z, basis_endo_small = basis_all[1:size(Xexo, 2)], basis_all[(size(Xexo, 2) +1):(size(Xexo, 2) + size(Z, 2))], basis_all[(size(Xexo, 2) + size(Z, 2) + 1):end]
        # basis_endo_small has same length as number of basis_endo who are true
        if !all(basis_endo_small)
            # if adding Xexo and Z makes Xendo collinear, consider these variables are exogeneous
            Xexo = hcat(Xexo, Xendo[:, .!basis_endo_small])
            Xendo = Xendo[:, basis_endo_small]
            XexoXexo = Xexo'Xexo
            XexoZ = Xexo'Z
            XexoXendo = Xexo'Xendo
            ZXendo = Z'Xendo
            XendoXendo = Xendo'Xendo

            # out returns false for endo collinear with instruments
            basis_endo2 = trues(length(basis_endo))
            basis_endo2[basis_endo] = basis_endo_small
            # TODO: I should probably also change formula in this case so that predict still works 
            ans = 1:length(basis_endo)
            ans = vcat(ans[.!basis_endo2], ans[basis_endo2])
            perm = vcat(1:length(basis_Xexo), length(basis_Xexo) .+ ans)
            # there are basis_endo - basis_endo_small in endo
            out = join(coefendo_names[.!basis_endo2], " ")
            @info "Endogenous vars collinear with ivs. Recategorized as exogenous: $(out)"
            
            # third pass
            XexoZXendo = Symmetric(hvcat(3, XexoXexo, XexoZ, XexoXendo, 
                               zeros(size(Z, 2), size(Xexo, 2)), ZZ, ZXendo, 
                               zeros(size(Xendo, 2), size(Xexo, 2)), zeros(size(Xendo, 2), size(Z, 2)), XendoXendo))
            basis_all = basis!(XexoZXendo; has_intercept = has_intercept)
            basis_Xexo, basis_Z, basis_endo_small2 = basis_all[1:size(Xexo, 2)], basis_all[(size(Xexo, 2) +1):(size(Xexo, 2) + size(Z, 2))], basis_all[(size(Xexo, 2) + size(Z, 2) + 1):end]
        end
        if !all(basis_Xexo)
        	Xexo = Xexo[:, basis_Xexo]
            XexoXexo = XexoXexo[basis_Xexo, basis_Xexo]
            XexoXendo = XexoXendo[basis_Xexo, :]
        end
        if !all(basis_Z)
        	Z = Z[:, basis_Z]
            ZZ = ZZ[basis_Z, basis_Z]
            ZXendo = ZXendo[basis_Z, :]
        end
        XexoZ = XexoZ[basis_Xexo, basis_Z]
        size(ZXendo, 1) >= size(ZXendo, 2) || throw("Model not identified. There must be at least as many ivs as endogeneous variables")
        # basis_endo is true for stuff non colinear
        # I need to have same vector but removeing the true that have been reclassified as exo and replace them by nothing.  so i need to create a vector equal to false if non endo and non basis_endo_small, which is basis_endo2
        basis_endo2 = trues(length(basis_endo))
        basis_endo2[basis_endo] = basis_endo_small
        basis_coef = vcat(basis_Xexo, basis_endo[basis_endo2])

        # Build
        newZ = hcat(Xexo, Z)
        # now create Pi = newZ \ Xendo
        newZnewZ = hvcat(2,  XexoXexo, XexoZ, 
                             XexoZ', ZZ)
        newZXendo = vcat(XexoXendo, ZXendo)
        Pi = ls_solve!(Symmetric(hvcat(2, newZnewZ, newZXendo,
                                zeros(size(newZXendo')), zeros(size(Xendo, 2), size(Xendo, 2)))), 
                       size(newZnewZ, 2))
        newnewZ = newZ * Pi
        Xhat = hcat(Xexo, newnewZ)
        XhatXhat = Symmetric(hvcat(2,  XexoXexo, Xexo'newnewZ, 
                           zeros(size(newnewZ, 2), size(Xexo, 2)), newnewZ'newnewZ))
        X = hcat(Xexo, Xendo)
        # prepare residuals used for first stage F statistic
        ## partial out Xendo in place wrt (Xexo, Z)
        Xendo_res = BLAS.gemm!('N', 'N', -1.0, newZ, Pi, 1.0, Xendo)
        ## partial out Z in place wrt Xexo
        # Now create Pi2 = Xexo \ Z
        Pi2 = ls_solve!(Symmetric(hvcat(2, XexoXexo, XexoZ,
                                zeros(size(Z, 2), size(Xexo, 2)), ZZ)), size(Xexo, 2))
        Z_res = BLAS.gemm!('N', 'N', -1.0, Xexo, Pi2, 1.0, Z)
    else
        # get linearly independent columns
        perm = 1:size(Xexo, 2)
        XexoXexo = Xexo'Xexo
        basis_Xexo = basis!(Symmetric(deepcopy(XexoXexo)); has_intercept = has_intercept)
        if !all(basis_Xexo)
            Xexo = Xexo[:, basis_Xexo]
            XexoXexo = XexoXexo[basis_Xexo, basis_Xexo]
        end
        Xhat = Xexo
        XhatXhat = Symmetric(XexoXexo)
        X = Xexo
        basis_coef = basis_Xexo
    end

    ##############################################################################
    ##
    ## Do the regression
    ##
    ##############################################################################
    # use X'Y instead of X'y because of https://github.com/FixedEffects/FixedEffectModels.jl/issues/249
    Xy = Symmetric(hvcat(2, XhatXhat, Xhat'reshape(y, length(y), 1), 
                         zeros(size(Xhat, 2))', [0.0]))
    invsym!(Xy; diagonal = 1:size(Xhat, 2))
    invXhatXhat = Symmetric(.- Xy[1:(end-1),1:(end-1)])
    coef = Xy[1:(end-1),end]

    ##############################################################################
    ##
    ## Optionally save objects in a new dataframe
    ##
    ##############################################################################
    residuals = y - X * coef
    residuals2 = nothing
    if save_residuals
        residuals2 = Vector{Union{Float64, Missing}}(missing, nrows)
        if has_weights
            residuals2[esample] .= residuals ./ sqrt.(weights)
        else
            residuals2[esample] .= residuals
        end
    end

    augmentdf = DataFrame()
    if save_fes
        oldX = oldX[:, perm]
        if !all(basis_coef)
            oldX = oldX[:, basis_coef]
        end
        newfes, b, c = solve_coefficients!(oldy - oldX * coef, feM; tol = tol, maxiter = maxiter)
        for fekey in fekeys
            augmentdf[!, fekey] = df[:, fekey]
        end
        for j in eachindex(fes)
            augmentdf[!, feids[j]] = Vector{Union{Float64, Missing}}(missing, nrows)
            augmentdf[esample, feids[j]] = newfes[j]
        end
    end

    ##############################################################################
    ##
    ## Test Statistics
    ##
    ##############################################################################
    # Compute degrees of freedom
    dof_fes_total = 0
    dof_fes = 0
    if has_fes
        for fe in fes
            # adjust degree of freedom only if fe is not fully nested in a cluster variable:
            if (vcov isa Vcov.ClusterCovariance) && any(isnested(fe, v.groups) for v in values(vcov_method.clusters))
                dof_fes += 1 # if fe is nested you still lose 1 degree of freedom
            else
                #only count groups that exists
                dof_fes += nunique(fe)
            end
            dof_fes_total += nunique(fe)
        end
    end

    nclusters = nothing
    if vcov isa Vcov.ClusterCovariance
        nclusters = Vcov.nclusters(vcov_method)
    end

    # Compute standard error
    vcov_data = Vcov.VcovData(Xhat, XhatXhat, invXhatXhat, residuals, nobs - size(X, 2) - dof_fes)
    matrix_vcov = StatsAPI.vcov(vcov_data, vcov_method)
    # Compute Fstat
    F = Fstat(coef, matrix_vcov, has_intercept)
    # dof_ is the number of estimated coefficients beyond the intercept.
    dof_ = size(X, 2) - has_intercept
    dof_tstat_ = max(1, Vcov.dof_residual(vcov_data, vcov_method) - has_intercept | has_fe_intercept)
    p = fdistccdf(dof_, dof_tstat_, F)
    # Compute Fstat of First Stage
    if has_iv && first_stage
        Pip = Pi[(size(Pi, 1) - size(Z_res, 2) + 1):end, :]
        try 
            r_kp = Vcov.ranktest!(Xendo_res, Z_res, Pip,
                              vcov_method, size(X, 2), dof_fes)
            p_kp = chisqccdf(size(Z_res, 2) - size(Xendo_res, 2) + 1, r_kp)
            F_kp = r_kp / size(Z_res, 2)
        catch
            @info "ranktest failed ; first-stage statistics not estimated"
            p_kp, F_kp = NaN, NaN
        end
    end

    # Compute rss, tss
    rss = sum(abs2, residuals)
    mss = tss_total - rss
    if has_fes
        r2_within = 1 - rss / tss_partial
    end

    ##############################################################################
    ##
    ## Return regression result
    ##
    ##############################################################################
    # add omitted variables
    if !all(basis_coef)
        newcoef = zeros(length(basis_coef))
        newmatrix_vcov = fill(NaN, (length(basis_coef), length(basis_coef)))
        newindex = [searchsortedfirst(cumsum(basis_coef), i) for i in 1:length(coef)]
        for i in eachindex(newindex)
            newcoef[newindex[i]] = coef[i]
            for j in eachindex(newindex)
                newmatrix_vcov[newindex[i], newindex[j]] = matrix_vcov[i, j]
            end
        end
        coef = newcoef
        matrix_vcov = Symmetric(newmatrix_vcov)
    end
    if any(perm[i] != i for i in perm)
        _invperm = invperm(perm)
        coef = coef[_invperm]
        newmatrix_vcov = zeros(size(matrix_vcov))
        for i in 1:size(newmatrix_vcov, 1)
            for j in 1:size(newmatrix_vcov, 1)
                newmatrix_vcov[i, j] = matrix_vcov[_invperm[i], _invperm[j]]
            end
        end
        matrix_vcov = Symmetric(newmatrix_vcov)
    end


    if esample == Colon()
        esample = trues(nrows)
    end
    return FixedEffectModel(coef, matrix_vcov, vcov, nclusters, esample, residuals2, augmentdf, fekeys, coef_names, response_name, formula_origin, formula_schema, contrasts, nobs, dof_, dof_fes_total, dof_tstat_, rss, tss_total, F, p, iterations, converged, r2_within, F_kp, p_kp)
end
