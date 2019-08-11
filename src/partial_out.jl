"""
Partial out variables in a Dataframe

### Arguments
* `df::AbstractDataFrame`
* `model::Model`: A `Model` created using `@model`. See `@model`.
* `add_mean::Bool`: Should the initial mean added to the returned variable?
* `method::Symbol`: A symbol for the method. Default is :lsmr (akin to conjugate gradient descent). Other choices are :qr and :cholesky (factorization methods)
* `maxiter::Integer`: Maximum number of iterations
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
df.SpeciesC =  categorical(df.Species)
result = partial_out(df, @model(SepalWidth + SepalLength ~ 1, fe = SpeciesC), add_mean = true)
plot(layer(result[1], x="SepalWidth", y="SepalLength", Stat.binmean(n=10), Geom.point),
   layer(result[1], x="SepalWidth", y="SepalLength", Geom.smooth(method=:lm)))
```
"""
function partial_out(df::AbstractDataFrame, m::Model; kwargs...)
    partial_out(df, m.f; m.dict..., kwargs...)
end


function partial_out(df::AbstractDataFrame, f::FormulaTerm; 
    fe::Union{Symbol, Expr, Nothing} = nothing, 
    weights::Union{Symbol, Expr, Nothing} = nothing,
    add_mean = false,
    maxiter::Integer = 10000, contrasts::Dict = Dict{Symbol, Any}(),
    tol::Real = 1e-8,
    method::Symbol = :lsmr)
    weightvar = weights

    if  (ConstantTerm(0) ∉ eachterm(f.rhs)) & (ConstantTerm(1) ∉ eachterm(f.rhs))
        f = FormulaTerm(f.lhs, tuple(ConstantTerm(1), eachterm(f.rhs)...))
    end
    formula, formula_endo, formula_iv = decompose_iv(f)
    has_iv = formula_iv != nothing
    has_absorb = fe != nothing
    if has_iv
        error("partial_out does not support instrumental variables")
    end
    has_weights = (weightvar != nothing)


    # create a dataframe without missing values & negative weights
    vars = allvars(formula)
    absorb_vars = allvars(fe)
    all_vars = unique(vcat(vars, absorb_vars))
    esample = completecases(df[!, all_vars])
    if has_weights
        esample .&= isnaorneg(df[!, weightvar])
    end

    # initialize iterations & converged
    iterations = Int[]
    convergeds = Bool[]

    # Build fixedeffects, an array of AbtractFixedEffects
    if has_absorb
        feformula = @eval(@formula(nothing ~ $(fe)))
        fes, ids = parse_fixedeffect(df, feformula)
    end
    nobs = sum(esample)
    (nobs > 0) || error("sample is empty")
    # Compute weight vector
    sqrtw = get_weights(df, esample, weightvar)
    if has_absorb
        # in case some FixedEffect does not have interaction, remove the intercept
        if any([isa(fe.interaction, Ones) for fe in fes])
            formula = FormulaTerm(formula.lhs, tuple(ConstantTerm(0), (t for t in eachterm(formula.rhs) if t!= ConstantTerm(1))...))
            has_absorb_intercept = true
        end
        fes = FixedEffect[_subset(fe, esample) for fe in fes]
        pfe = FixedEffectMatrix(fes, sqrtw, Val{method})
    end

    # Compute residualized Y
    subdf = columntable(df[esample, unique(vcat(vars))])
    formula_y = FormulaTerm(ConstantTerm(0), (ConstantTerm(0), eachterm(formula.lhs)...))
    formula_y_schema = apply_schema(formula_y, schema(formula_y, subdf, contrasts), StatisticalModel)
    Y = convert(Matrix{Float64}, modelmatrix(formula_y_schema, subdf))
    Y .= Y .* sqrtw

    ynames = coefnames(formula_y_schema)[2]
    if !isa(ynames, Vector)
        ynames = [ynames]
    end
    ynames = Symbol.(ynames)
    if add_mean
        m = mean(Y, dims = 1)
    end
    if has_absorb
        Y, b, c = solve_residuals!(Y, pfe; maxiter = maxiter, tol = tol)
        append!(iterations, b)
        append!(convergeds, c)
    end

    # Compute residualized X
    formula_x = FormulaTerm(ConstantTerm(0), formula.rhs)
    formula_x_schema = apply_schema(formula_x, schema(formula_x, subdf, contrasts), StatisticalModel)
    X = convert(Matrix{Float64}, modelmatrix(formula_x_schema, subdf))
    X .= X .* sqrtw
    if has_absorb
        X, b, c = solve_residuals!(X, pfe; maxiter = maxiter, tol = tol)
        append!(iterations, b)
        append!(convergeds, c)
    end
    
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
        out[!, y] = Vector{Union{Float64, Missing}}(missing, size(df, 1))
        out[esample, y] = residuals[:, j]
    end
    return out, iterations, convergeds
end






