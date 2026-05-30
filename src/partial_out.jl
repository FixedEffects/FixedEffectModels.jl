"""
Partial out variables in a Dataframe

### Arguments
* `df`: A table
* `formula::FormulaTerm`: A formula created using `@formula`
* `add_mean::Bool`: Should the initial mean added to the returned variable?
* `method::Symbol`: A symbol for the method. Default is `:cpu`. Alternatively, use `:CUDA` or `:Metal` (load the respective package — CUDA.jl or Metal.jl — before `using FixedEffectModels`).
* `maxiter::Integer`: Maximum number of iterations
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true.
* `tol::Real`: Tolerance for the iterative demeaning. Default `1e-6`, matching `reg` (so that `partial_out` and `reg(...; save = :residuals)` return the same residuals).
* `align::Bool`: Should the returned DataFrame align with the original DataFrame in case of missing values? Default to true.
* `drop_singletons::Bool=false`: Should singletons be dropped?

### Returns
* `::DataFrame`: residuals, one column per dependent variable, aligned with the original dataframe.
* `::BitVector`: the estimation-sample mask (the rows actually used).
* `::Vector{Int}`: number of demeaning iterations for each column.
* `::Vector{Bool}`: convergence flag for each column.
* `::Int`: degrees of freedom absorbed by the fixed effects.

### Details
`partial_out` returns the residuals of a set of variables after regressing them on a set of regressors. The syntax is similar to `reg` - but it accepts multiple dependent variables. It returns a dataframe with as many columns as there are dependent variables and as many rows as the original dataframe.
The regression model is estimated on only the rows where *none* of the dependent variables is missing. 
Finally, with the option `add_mean = true`, the mean of the initial variable is added to the residuals.

### Examples
```julia
using  RDatasets, DataFrames, FixedEffectModels, Gadfly
df = dataset("datasets", "iris")
result = partial_out(df, @formula(SepalWidth + SepalLength ~ fe(Species)), add_mean = true)
plot(layer(result[1], x="SepalWidth", y="SepalLength", Stat.binmean(n=10), Geom.point),
   layer(result[1], x="SepalWidth", y="SepalLength", Geom.smooth(method=:lm)))
```
"""
function partial_out(
    @nospecialize(df), 
    @nospecialize(f::FormulaTerm); 
    @nospecialize(weights::Union{Symbol, Expr, Nothing} = nothing),
    @nospecialize(add_mean = false),
    @nospecialize(maxiter::Integer = 10000), 
    @nospecialize(contrasts::Dict = Dict{Symbol, Any}()),
    @nospecialize(method::Symbol = :cpu),
    @nospecialize(double_precision::Bool = true),
    @nospecialize(tol::Real = 1e-6),
    @nospecialize(align = true),
    @nospecialize(drop_singletons = false))

    # Normalize generic Tables.jl input once; the rest of the function uses
    # DataFrame indexing/views, and copycols = false avoids copies when possible.
    df = DataFrame(df; copycols = false)

    if method == :gpu
        @info "method = :gpu is deprecated. Use method = :CUDA or method = :Metal"
        method = :CUDA
    end

    if  (ConstantTerm(0) ∉ eachterm(f.rhs)) && (ConstantTerm(1) ∉ eachterm(f.rhs))
        f = FormulaTerm(f.lhs, tuple(ConstantTerm(1), eachterm(f.rhs)...))
    end
    formula, formula_endo, formula_iv = parse_iv(f)
    has_iv = formula_iv != FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    has_iv && throw(ArgumentError("partial_out does not support instrumental variables"))
    formula, formula_fes = parse_fe(formula)
    has_weights = weights !== nothing


    # create a dataframe without missing values & negative weights
    main_vars = unique(StatsModels.termvars(formula))
    fe_vars = unique(StatsModels.termvars(formula_fes))
    all_vars = unique(vcat(main_vars, fe_vars))
    esample = completecases(df, all_vars)
    if has_weights
        esample .&= BitArray(!ismissing(x) && (x > 0) for x in df[!, weights])
    end

    # initialize iterations & converged
    iterations = Int[]
    convergeds = Bool[]

    # Build fixedeffects, an array of AbtractFixedEffects
    fes, ids, ids_fes = parse_fixedeffect(df, formula_fes)
    has_fes = !isempty(fes)

    drop_singletons && drop_singletons!(esample, fes, Threads.nthreads())

    nobs = sum(esample)
    (nobs > 0) || throw(ArgumentError("sample is empty"))
    # Compute weights
    if has_weights
        weights = Weights(convert(Vector{Float64}, view(df, esample, weights)))
        all(isfinite, weights) || throw(ArgumentError("Weights are not finite"))
    else
        weights = uweights(sum(esample))
    end

    if has_fes
        # in case some FixedEffect does not have interaction, remove the intercept
        if any(isa(fe.interaction, UnitWeights) for fe in fes)
            formula = FormulaTerm(formula.lhs, tuple(ConstantTerm(0), (t for t in eachterm(formula.rhs) if t!= ConstantTerm(1))...))
        end
        fes = FixedEffect[fe[esample] for fe in fes]
        feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, weights, Val{method})
    end

    # Compute residualized Y
    vars = unique(StatsModels.termvars(formula))
    subdf = Tables.columntable(disallowmissing!(df[esample, vars]))
    formula_y = FormulaTerm(ConstantTerm(0), (ConstantTerm(0), eachterm(formula.lhs)...))
    formula_y_schema = apply_schema(formula_y, schema(formula_y, subdf, contrasts), StatisticalModel)
    Y = convert(Matrix{Float64}, modelmatrix(formula_y_schema, subdf))

    ynames = coefnames(formula_y_schema)[2]
    if !isa(ynames, Vector)
        ynames = [ynames]
    end
    if add_mean
        m = mean(Y, weights, dims = 1)
    end
    if has_fes
        _, b, c = solve_residuals!(eachcol(Y), feM; maxiter = maxiter, tol = tol, progress_bar = false)
        append!(iterations, b)
        append!(convergeds, c)
    end

    # Compute residualized X
    formula_x = FormulaTerm(ConstantTerm(0), formula.rhs)
    formula_x_schema = apply_schema(formula_x, schema(formula_x, subdf, contrasts), StatisticalModel)
    X = convert(Matrix{Float64}, modelmatrix(formula_x_schema, subdf))
    if has_fes
        _, b, c = solve_residuals!(eachcol(X), feM; maxiter = maxiter, tol = tol, progress_bar = false)
        append!(iterations, b)
        append!(convergeds, c)
    end
    if has_weights
        Y .= Y .* sqrt.(weights)
        X .= X .* sqrt.(weights)
    end
    # Compute residuals
    if size(X, 2) > 0
        mul!(Y, X, X \ Y, -1.0, 1.0)
    end
    residuals = Y

    # rescale residuals
    if has_weights
        residuals .= residuals ./ sqrt.(weights)
    end
    if add_mean
        residuals .= residuals .+ m
    end

    dof_fes = mapreduce(nunique, +, fes, init=0)

    # Return a dataframe
    out = DataFrame()

    for (j, y) in enumerate(ynames)
        if align && (nobs < length(esample))
            out[!, Symbol(y)] = Vector{Union{Float64, Missing}}(missing, size(df, 1))
            out[esample, Symbol(y)] = residuals[:, j]
        else
            out[!, Symbol(y)] = residuals[:, j]
        end
    end
    return out, esample, iterations, convergeds, dof_fes
end
