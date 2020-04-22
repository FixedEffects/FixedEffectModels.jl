"""
Partial out variables in a Dataframe

### Arguments
* `df`: A table
* `formula::FormulaTerm`: A formula created using `@formula`
* `add_mean::Bool`: Should the initial mean added to the returned variable?
* `method::Symbol`: A symbol for the method. Default is :cpu. Alternatively,  :gpu requires `CuArrays`. In this case, use the option `double_precision = false` to use `Float32`.
* `maxiter::Integer`: Maximum number of iterations
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true.
* `tol::Real`: Tolerance

### Returns
* `::DataFrame`: a dataframe with as many columns as there are dependent variables and as many rows as the original dataframe.
* `::Vector{Int}`: a vector of iterations for each column
* `::Vector{Bool}`: a vector of success for each column

### Details
`partial_out` returns the residuals of a set of variables after regressing them on a set of regressors. The syntax is similar to `reg` - but it accepts multiple dependent variables. It returns a dataframe with as many columns as there are dependent variables and as many rows as the original dataframe.
The regression model is estimated on only the rows where *none* of the dependent variables is missing. With the option `add_mean = true`, the mean of the initial variable is added to the residuals.

### Examples
```julia
using  RDatasets, DataFrames, FixedEffectModels, Gadfly
df = dataset("datasets", "iris")
result = partial_out(df, @formula(SepalWidth + SepalLength ~ fe(Species)), add_mean = true)
plot(layer(result[1], x="SepalWidth", y="SepalLength", Stat.binmean(n=10), Geom.point),
   layer(result[1], x="SepalWidth", y="SepalLength", Geom.smooth(method=:lm)))
```
"""
function partial_out(df::AbstractDataFrame, f::FormulaTerm; 
    weights::Union{Symbol, Expr, Nothing} = nothing,
    add_mean = false,
    maxiter::Integer = 10000, contrasts::Dict = Dict{Symbol, Any}(),
    method::Symbol = :cpu,
    double_precision::Bool = true,
    tol::Real = double_precision ? 1e-8 : 1e-6)

    if  (ConstantTerm(0) ∉ eachterm(f.rhs)) & (ConstantTerm(1) ∉ eachterm(f.rhs))
        f = FormulaTerm(f.lhs, tuple(ConstantTerm(1), eachterm(f.rhs)...))
    end
    formula, formula_endo, formula_iv = parse_iv(f)
    has_iv = formula_iv != nothing
    has_iv && throw("partial_out does not support instrumental variables")
    has_weights = weights != nothing


    # create a dataframe without missing values & negative weights
    all_vars = StatsModels.termvars(formula)
    esample = completecases(df[!, all_vars])
    if has_weights
        esample .&= BitArray(!ismissing(x) & (x > 0) for x in df[!, weights])
    end

    # initialize iterations & converged
    iterations = Int[]
    convergeds = Bool[]

    # Build fixedeffects, an array of AbtractFixedEffects
    fes, ids, formula = parse_fixedeffect(df, formula)
    has_fes = !isempty(fes)


    nobs = sum(esample)
    (nobs > 0) || throw("sample is empty")
    # Compute weights
    if has_weights
        weights = Weights(convert(Vector{Float64}, view(df, esample, weights)))
    else
        weights = Weights(Ones{Float64}(sum(esample)))
    end
    all(isfinite, weights) || throw("Weights are not finite")
    sqrtw = sqrt.(weights)

    if has_fes
        # in case some FixedEffect does not have interaction, remove the intercept
        if any(isa(fe.interaction, Ones) for fe in fes)
            formula = FormulaTerm(formula.lhs, tuple(ConstantTerm(0), (t for t in eachterm(formula.rhs) if t!= ConstantTerm(1))...))
            has_fes_intercept = true
        end
        fes = FixedEffect[_subset(fe, esample) for fe in fes]
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
        m = mean(Y, dims = 1)
    end
    if has_fes
        Y, b, c = solve_residuals!(Y, feM; maxiter = maxiter, tol = tol)
        append!(iterations, b)
        append!(convergeds, c)
    end
    Y .= Y .* sqrtw
    # Compute residualized X
    formula_x = FormulaTerm(ConstantTerm(0), formula.rhs)
    formula_x_schema = apply_schema(formula_x, schema(formula_x, subdf, contrasts), StatisticalModel)
    X = convert(Matrix{Float64}, modelmatrix(formula_x_schema, subdf))
    if has_fes
        X, b, c = solve_residuals!(X, feM; maxiter = maxiter, tol = tol)
        append!(iterations, b)
        append!(convergeds, c)
    end
    X .= X .* sqrtw
    # Compute residuals
    if size(X, 2) > 0
        residuals = Y .- X * (X \ Y)
    else
        residuals = Y
    end

    # rescale residuals
    if add_mean
        residuals .= residuals .+ m
    end
    residuals .= residuals ./ sqrtw

    # Return a dataframe
    out = DataFrame()
    j = 0

    for y in ynames
        j += 1
        if nobs < length(esample)
            out[!, Symbol(y)] = Vector{Union{Float64, Missing}}(missing, size(df, 1))
            out[esample, Symbol(y)] = residuals[:, j]
        else
            out[!, Symbol(y)] = residuals[:, j]
        end
    end
    return out, iterations, convergeds
end