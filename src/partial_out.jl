
function partial_out(f::Formula,
                     df::AbstractDataFrame; 
                     add_mean = false,
                     weight::Union(Symbol, Nothing) = nothing,
                     maxiter::Integer = 10000,
                     tol::FloatingPoint = 1e-8)


    rf = deepcopy(f)
    (has_absorb, absorb_formula, absorb_terms, has_iv, iv_formula, iv_terms, endo_formula, endo_terms) = decompose!(rf)
    if has_iv
        error("partial_out does not support instrumental variables")
    end
    rt = Terms(rf)
    has_weight = weight != nothing
    xf = Formula(nothing, rf.rhs)
    xt = Terms(xf)

    # create a dataframe without missing values & negative weights
    vars = allvars(rf)
    absorb_vars = allvars(absorb_formula)
    all_vars = vcat(vars, absorb_vars)
    all_vars = unique(convert(Vector{Symbol}, all_vars))
    esample = complete_cases(df[all_vars])
    if weight != nothing
        esample &= isnaorneg(df[weight])
        all_vars = unique(vcat(all_vars, weight))
    end
    subdf = df[esample, all_vars]
    all_except_absorb_vars = unique(convert(Vector{Symbol}, vars))
    for v in all_except_absorb_vars
        dropUnusedLevels!(subdf[v])
    end

    # Compute weight vector
    sqrtw = get_weight(subdf, weight)

    # initialize iterations & converged
    iterations = Int[]
    converged = Bool[]

    # Build fixedeffects, an array of AbtractFixedEffects
    if has_absorb
        fixedeffects = FixedEffect[FixedEffect(subdf, a, sqrtw) for a in absorb_terms.terms]
        # in case there is any intercept fe, remove the intercept
        if any([typeof(f.interaction) <: Ones for f in fixedeffects]) 
            xt.intercept = false
        end
        pfe = FixedEffectProblem(fixedeffects)
    else
        pfe = nothing
    end

    # Compute residualized Y
    yf = Formula(nothing, rf.lhs)
    yt = Terms(yf)
    yt.intercept = false
    mfY = simpleModelFrame(subdf, yt, esample)
    Y = ModelMatrix(mfY).m
    broadcast!(*, Y, Y, sqrtw)
    if add_mean
        m = mean(Y, 1)
    end
    residualize!(Y, iterations, converged, pfe, maxiter = maxiter, tol = tol)

    # Compute residualized X
    xvars = allvars(xf)
    if length(xvars) > 0 || xt.intercept
        if length(xvars) > 0 
            mf = simpleModelFrame(subdf, xt, esample)
            X = ModelMatrix(mf).m
        else
            X = fill(one(Float64), (size(subdf, 1), 1))
        end     
        broadcast!(*, X, X, sqrtw)
        residualize!(X, iterations, converged, pfe, maxiter = maxiter, tol = tol)
    end
    
    # Compute residuals
    if length(xvars) > 0 || xt.intercept
        H = At_mul_B(X, X)
        H = inv(cholfact!(H))
        coef = H * (At_mul_B(X, Y))
        residuals = Y - X * coef
    else
        residuals = Y
    end

    # rescale residuals
    broadcast!(/, residuals,  residuals, sqrtw)
    if add_mean
        broadcast!(+, residuals, m, residuals)
    end

    # Return a dataframe
    yvars = convert(Vector{Symbol}, map(string, yt.eterms))
    out = DataFrame()
    j = 0
    for y in yvars
        j += 1
        out[y] = DataArray(Float64, size(df, 1))
        out[esample, y] = residuals[:, j]
    end
    return(out)
end

