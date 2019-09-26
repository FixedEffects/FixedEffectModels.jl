function StatsBase.fit(::Type{FixedEffectModel}, m::Model, df::AbstractDataFrame; kwargs...)
    reg(m, df; kwargs...)
end

"""
Estimate a linear model with high dimensional categorical variables / instrumental variables

### Arguments
* `df::AbstractDataFrame`
* `model::Model`: A model created using [`@model`](@ref)
* `save::Union{Bool, Symbol} = false`: Should residuals and eventual estimated fixed effects saved in a dataframe? Use `save = :residuals` to only save residuals. Use `save = :fe` to only save fixed effects.
* `method::Symbol = :lsmr`: Method to deman regressors. `:lsmr` is akin to conjugate gradient descent.  To use LSMR on multiple cores, use `:lsmr_parallel`. To use LSMR with multiple threads,  use `lsmr_threads`. To use LSMR on GPU, use `lsmr_gpu`(requires `CuArrays`. Use the option `double_precision = false` to use `Float32` on the GPU.).
* `contrasts::Dict = Dict()` An optional Dict of contrast codings for each categorical variable in the `formula`.  Any unspecified variables will have `DummyCoding`.
* `maxiter::Integer = 10000`: Maximum number of iterations
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true.
* `tol::Real =1e-8`: Tolerance


### Details
Models with instruments variables are estimated using 2SLS. `reg` tests for weak instruments by computing the Kleibergen-Paap rk Wald F statistic, a generalization of the Cragg-Donald Wald F statistic for non i.i.d. errors. The statistic is similar to the one returned by the Stata command `ivreg2`.

### Examples
```julia
using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
df.StateC = categorical(df.State)
df.YearC = categorical(df.Year)
@time reg(df, @model(Sales ~ Price))
reg(df, @model(Sales ~ Price, fe = StateC + YearC))
reg(df, @model(Sales ~ NDI, fe = StateC + StateC&Year))
reg(df, @model(Sales ~ NDI, fe = StateC*Year))
reg(df, @model(Sales ~ (Price ~ Pimin)))
@time reg(df, @model(Sales ~ Price), weights = :Pop)
reg(df, @model(Sales ~ NDI, subset = State .< 30))
reg(df, @model(Sales ~ NDI, vcov = robust()))
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
    vcov::Union{Symbol, Expr} = :(simple()),
    weights::Union{Symbol, Nothing} = nothing,
    subset::Union{Symbol, Expr, Nothing} = nothing,
    maxiter::Integer = 10000, contrasts::Dict = Dict{Symbol, Any}(),
    dof_add::Integer = 0,
    save::Union{Bool, Symbol} = false,  method::Symbol = :lsmr, drop_singletons = true, 
    double_precision::Bool = true, tol::Real = double_precision ? sqrt(eps(Float64)) : sqrt(eps(Float32))
   )


    if isa(vcov, Symbol)
        @warn "vcov = $vcov is deprecated. Use vcov = $vcov()"
        vcov = Expr(:call, vcov)
    end
    vcovformula = VcovFormula(Val{vcov.args[1]}, (vcov.args[i] for i in 2:length(vcov.args))...)


    ##############################################################################
    ##
    ## Parse formula
    ##
    ##############################################################################
    if  (ConstantTerm(0) ∉ eachterm(f.rhs)) & (ConstantTerm(1) ∉ eachterm(f.rhs))
        f = FormulaTerm(f.lhs, tuple(ConstantTerm(1), eachterm(f.rhs)...))
    end
    formula, formula_endo, formula_iv = decompose_iv(f)
    has_iv = formula_iv != nothing
    has_fe = fe != nothing 
    has_weights = weights != nothing
    has_subset = subset != nothing


    ##############################################################################
    ##
    ## Save keyword argument
    ##
    ##############################################################################
    if !isa(save, Bool)
        if save ∉ (:residuals, :fe)
            throw("the save keyword argument must be a Bool or a Symbol equal to :residuals or :fe")
        end
    end
    save_residuals = (save == :residuals) | (save == true)
    save_fe = (save == :fe) | ((save == true) & has_fe)


    ##############################################################################
    ##
    ## Construct new dataframe after removing missing values
    ##
    ##############################################################################

    # create a dataframe without missing values & negative weights
    vars = unique(allvars(formula))
    iv_vars = unique(allvars(formula_iv))
    endo_vars = unique(allvars(formula_endo))
    absorb_vars = unique(allvars(fe))
    vcov_vars = unique(allvars(vcovformula))
    # create a dataframe without missing values & negative weights
    all_vars = unique(vcat(vars, vcov_vars, absorb_vars, endo_vars, iv_vars))

    esample = completecases(df, all_vars)

    if has_weights
        esample .&= isnaorneg(df[!, weights])
    end
    if subset != nothing
        subset = eval(evaluate_subset(df, subset))
        if length(subset) != size(df, 1)
            throw("df has $(size(df, 1)) rows but the subset vector has $(length(subset)) elements")
        end
        esample .&= subset
    end

    if has_fe
        feformula = @eval(@formula(0 ~ $(fe)))
        fes, ids = parse_fixedeffect(df, feformula)
        if drop_singletons
            for fe in fes
                drop_singletons!(esample, fe)
            end
        end
    end

    nobs = sum(esample)
    (nobs > 0) || throw("sample is empty")


    # Compute weights
    sqrtw = Ones{Float64}(sum(esample))
    if has_weights
        sqrtw = convert(Vector{Float64}, view(df, esample, weights))
        sqrtw .= sqrt.(sqrtw)
    end

    all(isfinite, sqrtw) || throw("Weights are not finite")

    # Compute feM, an AbstractFixedEffectSolver
    has_fe_intercept = false
    if has_fe
        # in case some FixedEffect does not have interaction, remove the intercept
        if any(isa(fe.interaction, Ones) for fe in fes)
            formula = FormulaTerm(formula.lhs, tuple(ConstantTerm(0), (t for t in eachterm(formula.rhs) if t!= ConstantTerm(1))...))
            has_fe_intercept = true
        end
        fes = FixedEffect[_subset(fe, esample) for fe in fes]
        feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, sqrtw, Val{method})
    end

    has_intercept = ConstantTerm(1) ∈ eachterm(formula.rhs)
    
    # Compute data for std errors
    vcov_method = VcovMethod(disallowmissing(view(df, esample, vcov_vars)), vcovformula)

    ##############################################################################
    ##
    ## Dataframe --> Matrix
    ##
    ##############################################################################
    subdf = StatsModels.columntable(disallowmissing(view(df, esample, vars)))
    formula_schema = apply_schema(formula, schema(formula, subdf, contrasts), StatisticalModel)

    # Obtain y
    # for a Vector{Float64}, conver(Vector{Float64}, y) aliases y
    y = convert(Vector{Float64}, response(formula_schema, subdf))
    all(isfinite, y) || throw("Some observations for the dependent variable are infinite")

    # Obtain X
    Xexo = convert(Matrix{Float64}, modelmatrix(formula_schema, subdf))
    all(isfinite, Xexo) || throw("Some observations for the exogeneous variables are infinite")

    yname, coef_names = coefnames(formula_schema)
    if !isa(coef_names, Vector)
        coef_names = [coef_names]
    end

    yname = Symbol(yname)
    coef_names = Symbol.(coef_names)

    if has_iv
        subdf = StatsModels.columntable(disallowmissing!(df[esample, endo_vars]))
        formula_endo_schema = apply_schema(formula_endo, schema(formula_endo, subdf, contrasts), StatisticalModel)
        Xendo = convert(Matrix{Float64}, modelmatrix(formula_endo_schema, subdf))
        all(isfinite, Xendo) || throw("Some observations for the endogenous variables are infinite")

        _, coefendo_names = coefnames(formula_endo_schema)
        if !isa(coefendo_names, Vector)
              coefendo_names = [coefendo_names]
          end
        append!(coef_names, Symbol.(coefendo_names))

 
        subdf = StatsModels.columntable(disallowmissing!(df[esample, iv_vars]))
        formula_iv_schema = apply_schema(formula_iv, schema(formula_iv, subdf, contrasts), StatisticalModel)
        Z = convert(Matrix{Float64}, modelmatrix(formula_iv_schema, subdf))
        all(isfinite, Z) || throw("Some observations for the instrumental variables are infinite")


        if size(Z, 2) < size(Xendo, 2)
            throw("Model not identified. There must be at least as many ivs as endogeneneous variables")
        end

        # modify formula to use in predict
        formula = FormulaTerm(formula.lhs, (tuple(eachterm(formula.rhs)..., eachterm(formula_endo.rhs)...)))
        formula_schema = apply_schema(formula, schema(formula, StatsModels.columntable(df), contrasts), StatisticalModel)
    end

    # compute tss now before potentially demeaning y
    tss_ = tss(y, has_intercept | has_fe_intercept, sqrtw)


    # create unitilaized 
    iterations, converged, r2_within = nothing, nothing, nothing
    F_kp, p_kp = nothing, nothing

    if has_fe
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

        y, b, c = solve_residuals!(y, feM; maxiter = maxiter, tol = tol)
        append!(iterations, b)
        append!(convergeds, c)

        Xexo, b, c = solve_residuals!(Xexo, feM; maxiter = maxiter, tol = tol)
        append!(iterations, b)
        append!(convergeds, c)

        if has_iv
            Xendo, b, c = solve_residuals!(Xendo, feM; maxiter = maxiter, tol = tol)
            append!(iterations, b)
            append!(convergeds, c)

            Z, b, c = solve_residuals!(Z, feM; maxiter = maxiter, tol = tol)
            append!(iterations, b)
            append!(convergeds, c)
        end

        iterations = maximum(iterations)
        converged = all(convergeds)
        if converged == false
            @warn "convergence not achieved in $(iterations) iterations; try increasing maxiter or decreasing tol."
        end
    end

    y .= y .* sqrtw
    Xexo .= Xexo .* sqrtw
    if has_iv
        Xendo .= Xendo .* sqrtw
        Z .= Z .* sqrtw
    end
    ##############################################################################
    ##
    ## Get Linearly Independent Components of Matrix
    ##
    ##############################################################################

    # Compute linearly independent columns + create the Xhat matrix
    if has_iv
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
        Xendo_res = BLAS.gemm!('N', 'N', -1.0, newZ, Pi, 1.0, Xendo)
        ## partial out Z in place wrt Xexo
        Pi2 = cholesky!(Symmetric(Xexo' * Xexo)) \ (Xexo' * Z)
        Z_res = BLAS.gemm!('N', 'N', -1.0, Xexo, Pi2, 1.0, Z)

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

    crossx = cholesky!(Symmetric(Xhat' * Xhat))
    coef = crossx \ (Xhat' * y)
    residuals = y - X * coef

    ##############################################################################
    ##
    ## Optionally save objects in a new dataframe
    ##
    ##############################################################################

    augmentdf = DataFrame()
    if save_residuals
        if nobs < length(esample)
            augmentdf.residuals = Vector{Union{Float64, Missing}}(missing, length(esample))
            augmentdf[esample, :residuals] = residuals ./ sqrtw
        else
            augmentdf[!, :residuals] = residuals ./ sqrtw
        end
    end
    if save_fe
        oldX = getcols(oldX, basecoef)
        newfes, b, c = solve_coefficients!(oldy - oldX * coef, feM; tol = tol, maxiter = maxiter)
        for j in 1:length(fes)
            if nobs < length(esample)
                augmentdf[!, ids[j]] = Vector{Union{Float64, Missing}}(missing, length(esample))
                augmentdf[esample, ids[j]] = newfes[j]
            else
                augmentdf[!, ids[j]] = newfes[j]
            end
        end
    end
    
    ##############################################################################
    ##
    ## Test Statistics
    ##
    ##############################################################################

    # Compute degrees of freedom
    dof_absorb = 0
    if has_fe
        for fe in fes
            # adjust degree of freedom only if fe is not fully nested in a cluster variable:
            if isa(vcovformula, VcovClusterFormula) && any(isnested(fe, v.refs) for v in eachcol(vcov_method.clusters))
                    dof_absorb += 1 # if fe is nested you still lose 1 degree of freedom 
            else
                #only count groups that exists
                dof_absorb += ndistincts(fe)
            end
        end
    end
    dof_residual = max(1, nobs - size(X, 2) - dof_absorb - dof_add)

    # Compute rss, tss, r2, r2 adjusted
    rss = sum(abs2, residuals)
    mss = tss_ - rss
    r2 = 1 - rss / tss_
    adjr2 = 1 - rss / tss_ * (nobs - (has_intercept | has_fe_intercept)) / dof_residual
    if has_fe
        r2_within = 1 - rss / tss(y, (has_intercept | has_fe_intercept), sqrtw)
    end

    # Compute standard error
    vcov_data = VcovData(Xhat, crossx, residuals, dof_residual)
    matrix_vcov = vcov!(vcov_method, vcov_data)

    # Compute Fstat
    F = Fstat(coef, matrix_vcov, nobs, has_intercept, vcov_method, vcov_data)

    dof_residual = max(1, df_FStat(vcov_method, vcov_data, has_intercept))
    p = ccdf(FDist(max(length(coef) - has_intercept, 1), dof_residual), F)

    # Compute Fstat of First Stage
    if has_iv
        Pip = Pi[(size(Pi, 1) - size(Z_res, 2) + 1):end, :]
        (F_kp, p_kp) = ranktest!(Xendo_res, Z_res, Pip,
                                  vcov_method, size(X, 2), dof_absorb)
    end

    ##############################################################################
    ##
    ## Return regression result
    ##
    ##############################################################################

    # add omitted variables
    if !all(basecoef)
        newcoef = zeros(length(basecoef))
        newmatrix_vcov = fill(NaN, (length(basecoef), length(basecoef)))
        newindex = [searchsortedfirst(cumsum(basecoef), i) for i in 1:length(coef)]
        for i in eachindex(newindex)
            newcoef[newindex[i]] = coef[i]
            for j in eachindex(newindex)
                newmatrix_vcov[newindex[i], newindex[j]] = matrix_vcov[i, j]
            end
        end
        coef = newcoef
        matrix_vcov = newmatrix_vcov
    end

    return FixedEffectModel(coef, matrix_vcov, vcov, esample, augmentdf,
                            coef_names, yname, f, formula_schema, nobs, dof_residual,
                            rss, tss_, r2, adjr2, F, p,
                            fe, iterations, converged, r2_within, 
                            F_kp, p_kp)
end


