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
df[:SpeciesC] =  categorical(df[:Species])
result = partial_out(df, @model(SepalWidth + SepalLength ~ 1, fe = SpeciesC), add_mean = true)
plot(layer(result[1], x="SepalWidth", y="SepalLength", Stat.binmean(n=10), Geom.point),
   layer(result[1], x="SepalWidth", y="SepalLength", Geom.smooth(method=:lm)))
```
"""
function partial_out(df::AbstractDataFrame, m::Model; kwargs...)
    partial_out(df, m.f; m.dict..., kwargs...)
end


function partial_out(df::AbstractDataFrame, f::Formula; 
    fe::Union{Symbol, Expr, Nothing} = nothing, 
    weights::Union{Symbol, Expr, Nothing} = nothing,
    add_mean = false,
    maxiter::Integer = 10000, tol::Real = 1e-8,
    method::Symbol = :lsmr)
    feformula = fe
    weightvar = weights


    rf = deepcopy(f)
    (has_iv, iv_formula, iv_terms, endo_formula, endo_terms) = decompose_iv!(rf)
    has_absorb = feformula != nothing
    if has_iv
        error("partial_out does not support instrumental variables")
    end
    rt = Terms(rf)
    has_weights = (weightvar != nothing)
    xf = @eval(@formula($nothing ~ $(rf.rhs)))
    xt = Terms(xf)

    # create a dataframe without missing values & negative weights
    vars = allvars(rf)
    absorb_vars = allvars(feformula)
    all_vars = vcat(vars, absorb_vars)
    all_vars = unique(convert(Vector{Symbol}, all_vars))
    esample = completecases(df[all_vars])
    if has_weights
        esample .&= isnaorneg(df[weightvar])
    end

    # initialize iterations & converged
    iterations = Int[]
    convergeds = Bool[]

    # Build fixedeffects, an array of AbtractFixedEffects
    if has_absorb
        fes, ids = parse_fixedeffect(df, Terms(@eval(@formula(nothing ~ $(feformula)))))
    end
    nobs = sum(esample)
    (nobs > 0) || error("sample is empty")
    # Compute weight vector
    sqrtw = get_weights(df, esample, weightvar)
    if has_absorb
        # in case there is any intercept fe, remove the intercept
        if any([isa(fe.interaction, Ones) for fe in fes]) 
            xt.intercept = false
        end
        fes = FixedEffect[_subset(fe, esample) for fe in fes]
        pfe = FixedEffectMatrix(fes, sqrtw, Val{method})
    else
        pfe = nothing
    end

    # Compute residualized Y
    yf = @eval(@formula($nothing ~ $(rf.lhs)))
    yt = Terms(yf)
    yt.intercept = false
    mfY = ModelFrame2(yt, df, esample)
    Y = ModelMatrix(mfY).m
    Y .= Y .* sqrtw
    if add_mean
        m = mean(Y, dims = 1)
    end
    if has_absorb
        Y, b, c = solve_residuals!(Y, pfe; maxiter = maxiter, tol = tol)
        append!(iterations, b)
        append!(convergeds, c)
    end

    # Compute residualized X
    xvars = allvars(xf)
    if length(xvars) > 0 || xt.intercept
        if length(xvars) > 0 
            mf = ModelFrame2(xt, df, esample)
            X = ModelMatrix(mf).m
        else
            X = fill(one(Float64), (length(esample), 1))
        end     
        X .= X .* sqrtw
        if has_absorb
            X, b, c = solve_residuals!(X, pfe; maxiter = maxiter, tol = tol)
            append!(iterations, b)
            append!(convergeds, c)
        end
    end
    
    # Compute residuals
    if length(xvars) > 0 || xt.intercept
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
    yvars = convert(Vector{Symbol}, Symbol.(yt.eterms))
    out = DataFrame()
    j = 0
    for y in yvars
        j += 1
        out[y] = Vector{Union{Float64, Missing}}(missing, size(df, 1))
        out[esample, y] = residuals[:, j]
    end
    return out, iterations, convergeds
end






