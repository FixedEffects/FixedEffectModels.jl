"""
Partial out a vector or a matrix

### Arguments
* `X` : a `AbstractVector{Float64}` or an `AbstractMatrix{Float64}`
* `fes`: A Vector of `FixedEffect`
* w: A Vector of weights
* `add_mean`. If true,  the mean of the initial variable is added to the residuals.
* `maxiter` : Maximum number of iterations
* `tol` : tolerance
* `method` : A symbol for the method. Default is :lsmr (akin to conjugate gradient descent). Other choices are :qr and :cholesky (factorization methods)


### Returns
* `X` : a residualized version of `X` 
* `iterations`: a vector of iterations for each column
* `converged`: a vector of success for each column

### Examples
```julia
using  CategoricalArrays, FixedEffectModels
p1 = categorical(repeat(1:5, inner = 2))
p2 = categorical(repeat(1:5, outer = 2))
X = rand(10, 5)
partial_out!(X, [FixedEffect(p1), FixedEffect(p2)])
```
"""
function partial_out!(Y::Union{AbstractVector{Float64}, Matrix{Float64}}, fes::Vector{<: FixedEffect}; w = Ones{Float64}(size(Y, 1)), add_mean::Bool = false, maxiter::Integer = 10000, tol::Real = 1e-8, method::Symbol = :lsmr)

    sqrtw = sqrt.(w)
    fep = FixedEffectProblem(fes, sqrtw, Val{method})

    Y .= Y .* sqrtw
    if add_mean
        m = mean(Y, dims = 1)
    end

    iterations = Int[]
    converged = Bool[]
    residualize!(Y, fep, iterations, converged; maxiter = maxiter, tol = tol)

    if add_mean
        Y .= Y .+ m
    end
    Y .= Y ./ sqrtw

    return Y, iterations, converged
end




"""
Partial out variables in a dataframe

### Arguments
* `df` : AbstractDataFrame
* `f` : Formula,
* `fe` : Fixed effect formula. Default to fe()
* `weights`: Weight formula. Corresponds to analytical weights 
* `add_mean` : should intial mean added to the returned variable
* `maxiter` : Maximum number of iterations
* `tol` : tolerance
* `method` : A symbol for the method. Default is :lsmr (akin to conjugate gradient descent). Other choices are :qr and :cholesky (factorization methods)

### Returns
* `::DataFrame` : a dataframe with as many columns as there are dependent variables and as many rows as the original dataframe.
* `iterations`: a vector of iterations for each column
* `converged`: a vector of success for each column

### Details
`partial_out` returns the residuals of a set of variables after regressing them on a set of regressors. The syntax is similar to `reg` - but it accepts multiple dependent variables. It returns a dataframe with as many columns as there are dependent variables and as many rows as the original dataframe.
The regression model is estimated on only the rows where *none* of the dependent variables is missing. With the option `add_mean = true`, the mean of the initial variable is added to the residuals.

### Examples
```julia
using  RDatasets, DataFrames, FixedEffectModels, Gadfly
df = dataset("datasets", "iris")
df[:SpeciesCategorical] =  categorical(df[:Species])
result = partial_out(df, @model(SepalWidth + SepalLength ~ 1, fe = SpeciesCategorical, add_mean = true))
plot(
   layer(result, x="SepalWidth", y="SepalLength", Stat.binmean(n=10), Geom.point),
   layer(result, x="SepalWidth", y="SepalLength", Geom.smooth(method=:lm))
)
```
"""
function partial_out(df::AbstractDataFrame, m::Model)
    partial_out(df, m.f; m.dict...)
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
    converged = Bool[]

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
        if any([isa(x.interaction, Ones) for x in fes]) 
            xt.intercept = false
        end
        fes = FixedEffect[x[esample] for x in fes]
        pfe = FixedEffectProblem(fes, sqrtw, Val{method})
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
    residualize!(Y, pfe, iterations, converged, maxiter = maxiter, tol = tol)

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
        residualize!(X, pfe, iterations, converged, maxiter = maxiter, tol = tol)
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
    return out, iterations, converged
end






