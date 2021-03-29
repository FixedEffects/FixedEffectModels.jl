"""
Estimate a linear model with high dimensional categorical variables / instrumental variables

### Arguments
* `df`: a Table
* `FormulaTerm`: A formula created using [`@formula`](@ref)
* `CovarianceEstimator`: A method to compute the variance-covariance matrix

### Keyword arguments
* `contrasts::Dict = Dict()` An optional Dict of contrast codings for each categorical variable in the `formula`.  Any unspecified variables will have `DummyCoding`.
* `weights::Union{Nothing, Symbol}` A symbol to refer to a columns for weights
* `save::Symbol`: Should residuals and eventual estimated fixed effects saved in a dataframe? Default to `:none` Use `save = :residuals` to only save residuals, `save = :fe` to only save fixed effects, `save = :all` for both. Once saved, they can then be accessed using `residuals()` or `fe()`. The returned DataFrame is automatically aligned with the original DataFrame.
* `method::Symbol`: A symbol for the method. Default is :cpu. Alternatively,  :gpu requires `CuArrays`. In this case, use the option `double_precision = false` to use `Float32`.
* `nthreads::Integer` Number of threads used in estimated. Default to `Threads.nthreads()` if `method = :cpu`, else to 256 if `method = :gpu`.
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true.
* `tol::Real` Tolerance. Default to 1e-8 if `double_precision = true`, 1e-6 otherwise.
* `maxiter::Integer = 10000`: Maximum number of iterations
* `drop_singletons::Bool = true`: Should singletons be dropped?
* `progress_bar::Bool = true`: Should the regression show a progressbar
* `first_stage::Bool = true`: Should the first-stage F-stat and p-value be computed?
* `dof_add::Integer = 0`: 
* `subset::Union{Nothing, AbstractVector} = nothing`: select specific rows 


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
"""

function reg(
    @nospecialize(df),
    @nospecialize(formula::FormulaTerm),
    @nospecialize(vcov::CovarianceEstimator = Vcov.simple());
    @nospecialize(contrasts::Dict = Dict{Symbol, Any}()),
    @nospecialize(weights::Union{Symbol, Nothing} = nothing),
    @nospecialize(save::Union{Bool, Symbol} = :none),
    @nospecialize(method::Symbol = :cpu),
    @nospecialize(nthreads::Integer = method == :cpu ? Threads.nthreads() : 256),
    @nospecialize(double_precision::Bool = true),
    @nospecialize(tol::Real = double_precision ? 1e-8 : 1e-6),
    @nospecialize(maxiter::Integer = 10000),
    @nospecialize(drop_singletons::Bool = true),
    @nospecialize(progress_bar::Bool = true),
    @nospecialize(dof_add::Integer = 0),
    @nospecialize(subset::Union{Nothing, AbstractVector} = nothing), 
    @nospecialize(first_stage::Bool = true))

    df = DataFrame(df; copycols = false)
    N = size(df, 1)

    ##############################################################################
    ##
    ## Parse formula
    ##
    ##############################################################################

    formula_origin = formula
    if  !omitsintercept(formula) & !hasintercept(formula)
        formula = FormulaTerm(formula.lhs, InterceptTerm{true}() + formula.rhs)
    end
    formula, formula_endo, formula_iv = parse_iv(formula)
    has_iv = formula_iv !== nothing
    has_weights = weights !== nothing

    ##############################################################################
    ##
    ## Save keyword argument
    ##
    ##############################################################################
    if save == true
        save = :all
    elseif save == false
        save = :none
    end
    if save ∉ (:all, :residuals, :fe, :none)
            throw("the save keyword argument must be a Symbol equal to :all, :none, :residuals or :fe")
    end
    save_residuals = (save == :residuals) | (save == :all)

    ##############################################################################
    ##
    ## Construct new dataframe after removing missing values
    ##
    ##############################################################################

    # create a dataframe without missing values & negative weights
    vars = StatsModels.termvars(formula)
    iv_vars = Symbol[]
    endo_vars = Symbol[]
    if has_iv
        iv_vars = StatsModels.termvars(formula_iv)
        endo_vars = StatsModels.termvars(formula_endo)
    end
    # create a dataframe without missing values & negative weights
    all_vars = unique(vcat(vars, endo_vars, iv_vars))

    esample = completecases(df, all_vars)
    if has_weights
        esample .&= BitArray(!ismissing(x) && (x > 0) for x in df[!, weights])
    end
    if subset !== nothing
        if length(subset) != N
            throw("df has $(N) rows but the subset vector has $(length(subset)) elements")
        end
        esample .&= BitArray(!ismissing(x) && x for x in subset)
    end
    esample .&= Vcov.completecases(df, vcov)
    fes, ids, formula = parse_fixedeffect(df, formula)
    has_fes = !isempty(fes)
    if has_fes
        if drop_singletons
            for fe in fes
                drop_singletons!(esample, fe)
            end
        end
    end
    save_fe = (save == :fe) | ((save == :all) & has_fes)

    nobs = sum(esample)
    (nobs > 0) || throw("sample is empty")

    if nobs == N
        esample = Colon()
    end

    # Compute weights
    if has_weights
        weights = Weights(convert(Vector{Float64}, view(df, esample, weights)))
        all(isfinite, weights) || throw("Weights are not finite")
    else
        weights = uweights(nobs)
    end

    # Compute feM, an AbstractFixedEffectSolver
    has_intercept = hasintercept(formula)
    has_fe_intercept = false
    if has_fes
        if any(fe.interaction isa UnitWeights for fe in fes)
            has_fe_intercept = true
        end
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
    exo_vars = unique(StatsModels.termvars(formula))
    subdf = Tables.columntable((; (x => disallowmissing(view(df[!, x], esample)) for x in exo_vars)...))
    formula_schema = apply_schema(formula, schema(formula, subdf, contrasts), FixedEffectModel, has_fe_intercept)

    # Obtain y
    # for a Vector{Float64}, conver(Vector{Float64}, y) aliases y
    y = convert(Vector{Float64}, response(formula_schema, subdf))
    all(isfinite, y) || throw("Some observations for the dependent variable are infinite")

    # Obtain X
    Xexo = convert(Matrix{Float64}, modelmatrix(formula_schema, subdf))
    all(isfinite, Xexo) || throw("Some observations for the exogeneous variables are infinite")

    response_name, coef_names = coefnames(formula_schema)
    if !(coef_names isa Vector)
        coef_names = typeof(coef_names)[coef_names]
    end

    if has_iv
        subdf = Tables.columntable((; (x => disallowmissing(view(df[!, x], esample)) for x in endo_vars)...))
        formula_endo_schema = apply_schema(formula_endo, schema(formula_endo, subdf, contrasts), StatisticalModel)
        Xendo = convert(Matrix{Float64}, modelmatrix(formula_endo_schema, subdf))
        all(isfinite, Xendo) || throw("Some observations for the endogenous variables are infinite")
        _, coefendo_names = coefnames(formula_endo_schema)
        append!(coef_names, coefendo_names)

        subdf = Tables.columntable((; (x => disallowmissing(view(df[!, x], esample)) for x in iv_vars)...))
        formula_iv_schema = apply_schema(formula_iv, schema(formula_iv, subdf, contrasts), StatisticalModel)
        Z = convert(Matrix{Float64}, modelmatrix(formula_iv_schema, subdf))
        all(isfinite, Z) || throw("Some observations for the instrumental variables are infinite")

        # modify formula to use in predict
        formula = FormulaTerm(formula.lhs, (tuple(eachterm(formula.rhs)..., (term for term in eachterm(formula_endo.rhs) if term != ConstantTerm(0))...)))
    end
    # compute tss now before potentially demeaning y
    tss_total = tss(y, has_intercept | has_fe_intercept, weights)
    # create unitilaized
    iterations, converged, r2_within = nothing, nothing, nothing
    F_kp, p_kp = nothing, nothing

    if has_fes
        # used to compute tss even without save_fe
        if save_fe
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
        if has_iv
            Xall = Combination(y, Xexo, Xendo, Z)
        else
            Xall = Combination(y, Xexo)
        end

        _, iterations, convergeds = solve_residuals!(Xall, feM; maxiter = maxiter, tol = tol, progress_bar = progress_bar)

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

    if has_iv
        # Compute linearly independent columns + create the Xhat matrix
    	
        # first pass: remove colinear variables in Xendo
    	basis_endo = basecol(Xendo)
    	Xendo = getcols(Xendo, basis_endo)

    	# second pass: remove colinear variable in Xexo, Z, and Xendo
    	basis = basecol(Xexo, Z, Xendo)
        basis_Xexo = basis[1:size(Xexo, 2)]
        basis_Z = basis[(size(Xexo, 2) +1):(size(Xexo, 2) + size(Z, 2))]
        basis_endo_small = basis[(size(Xexo, 2) + size(Z, 2) + 1):end]
        if !all(basis_endo_small)
            # if adding Xexo and Z makes Xendo collinar
            # consider these variables are exogeneous
            Xexo = hcat(Xexo, getcols(Xendo, .!basis_endo_small))
            Xendo = getcols(Xendo, basis_endo_small)

            # out returns false for endo collinear with instruments
            basis_endo2 = trues(length(basis_endo))
            basis_endo2[basis_endo] = basis_endo_small

            # TODO: I should probably also change formula in this case so that predict still works 
            coef_names = vcat(coef_names[1:length(basis_Xexo)], coefendo_names[.!basis_endo2], coefendo_names[basis_endo2])
            if savefe
                oldX = hcat(oldX[1:length(basis_Xexo)], oldX[:, (length(basis_Xexo)+1):end][.!basisendo2], oldX[:, (length(basis_Xexo)+1):end][!basisendo2])
            end
            out = join(coefendo_names[.!basis_endo2], " ")
            @info "Endogeneous var(s) are collinear with instruments. Var(s) recategorized as exogenous: $(out)"

            # third pass
            basis = basecol(Xexo, Z, Xendo)
            basis_Xexo = basis[1:size(Xexo, 2)]
            basis_Z = basis[(size(Xexo, 2) +1):(size(Xexo, 2) + size(Z, 2))]
        end

    	Xexo = getcols(Xexo, basis_Xexo)
    	Z = getcols(Z, basis_Z)
        size(Z, 2) >= size(Xendo, 2) || throw("Model not identified. There must be at least as many ivs as endogeneneous variables")
        basis_coef = vcat(basis_Xexo, basis_endo[basis_endo_small])

        # Build
        newZ = hcat(Xexo, Z)
        Pi = cholesky!(Symmetric(newZ' * newZ)) \ (newZ' * Xendo)
        Xhat = hcat(Xexo, newZ * Pi)
        X = hcat(Xexo, Xendo)

        # prepare residuals used for first stage F statistic
        ## partial out Xendo in place wrt (Xexo, Z)
        Xendo_res = BLAS.gemm!('N', 'N', -1.0, newZ, Pi, 1.0, Xendo)
        ## partial out Z in place wrt Xexo
        Pi2 = cholesky!(Symmetric(Xexo' * Xexo)) \ (Xexo' * Z)
        Z_res = BLAS.gemm!('N', 'N', -1.0, Xexo, Pi2, 1.0, Z)
    else
        # get linearly independent columns
        basis_Xexo = basecol(Xexo)
        Xexo = getcols(Xexo, basis_Xexo)
        Xhat = Xexo
        X = Xexo
        basis_coef = basis_Xexo
    end

    if !all(basis_coef)
        out = join(coef_names[.!basis_coef], " ")
        @info "Collinearities detected. Var(s) dropped: $(out)"
    end


    ##############################################################################
    ##
    ## Do the regression
    ##
    ##############################################################################
   
    crossx = cholesky!(Symmetric(Xhat' * Xhat))
    coef = crossx \ (Xhat' * y)

    ##############################################################################
    ##
    ## Optionally save objects in a new dataframe
    ##
    ##############################################################################
    residuals = y - X * coef
    residuals2 = nothing
    if save_residuals
        residuals2 = Vector{Union{Float64, Missing}}(missing, N)
        if has_weights
            residuals .= residuals ./ sqrt.(weights)
        end
        residuals2[esample] = residuals
    end

    augmentdf = DataFrame()
    if save_fe
        oldX = getcols(oldX, basis_coef)
        newfes, b, c = solve_coefficients!(oldy - oldX * coef, feM; tol = tol, maxiter = maxiter)
        for j in eachindex(fes)
            augmentdf[!, ids[j]] = Vector{Union{Float64, Missing}}(missing, N)
            augmentdf[esample, ids[j]] = newfes[j]
        end
    end

    ##############################################################################
    ##
    ## Test Statistics
    ##
    ##############################################################################
    # Compute degrees of freedom
    dof_fes = 0
    if has_fes
        for fe in fes
            # adjust degree of freedom only if fe is not fully nested in a cluster variable:
            if (vcov isa Vcov.ClusterCovariance) && any(isnested(fe, v.refs) for v in values(vcov_method.clusters))
                dof_fes += 1 # if fe is nested you still lose 1 degree of freedom
            else
                #only count groups that exists
                dof_fes += nunique(fe)
            end
        end
    end
    dof_residual_ = max(1, nobs - size(X, 2) - dof_fes - dof_add)

    nclusters = nothing
    if vcov isa Vcov.ClusterCovariance
        nclusters = Vcov.nclusters(vcov_method)
    end


    # Compute standard error
    vcov_data = Vcov.VcovData(Xhat, crossx, residuals, dof_residual_)
    matrix_vcov = StatsBase.vcov(vcov_data, vcov_method)

    # Compute Fstat
    F = Fstat(coef, matrix_vcov, has_intercept)
    df_FStat_ = max(1, Vcov.df_FStat(vcov_data, vcov_method, has_intercept | has_fe_intercept))
    p = fdistccdf(max(length(coef) - (has_intercept | has_fe_intercept), 1), df_FStat_, F)
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

    # Compute rss, tss, r2, r2 adjusted
    rss = sum(abs2, residuals)
    mss = tss_total - rss
    r2 = 1 - rss / tss_total
    adjr2 = 1 - rss / tss_total * (nobs - (has_intercept | has_fe_intercept)) / dof_residual_
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
        matrix_vcov = newmatrix_vcov
    end
    if esample == Colon()
        esample = trues(N)
    end
    return FixedEffectModel(coef, matrix_vcov, vcov, nclusters, esample, residuals2, augmentdf, coef_names, response_name, formula_origin, formula, contrasts, nobs, dof_residual_,  rss, tss_total, r2, adjr2, F, p, iterations, converged, r2_within, F_kp, p_kp)
end
