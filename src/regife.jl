##############################################################################
##
## Factor models (Bai 2009)
##
##############################################################################
# Object constructed by the user
type InteractiveFixedEffectModel 
    id::Symbol
    time::Symbol
    dimension::Int64
end

# Object output by idfe
type InteractiveFixedEffectResult 
    id::PooledDataArray
    time::PooledDataArray
    coef::Vector{Float64} 
    loadings::Matrix{Float64}  # N x d
    factors::Matrix{Float64} # T x d
    iterations::Int64
    converged::Bool
end




function reg(f::Formula, df::AbstractDataFrame, m::InteractiveFixedEffectModel; weight = nothing, maxiter::Int64 = 10000, tol::Float64 = 1e-10)

    #################
    # Prepare the data (transform dataframe to matrix, demean if fixed effects, multiply by weight
    #################
    rf = deepcopy(f)

    ## decompose formula into normal  vs absorbpart
    (rf, has_absorb, absorb_formula) = decompose_absorb!(rf)
    if has_absorb
        absorb_vars = allvars(absorb_formula)
        absorb_terms = Terms(absorb_formula)
    else
        absorb_vars = Symbol[]
    end


    rt = Terms(rf)
    if has_absorb
        rt.intercept = false
    end

    ## create a dataframe without missing values & negative weights
    factor_vars = [m.id, m.time]
    vars = allvars(rf)
    all_vars = vcat(vars, absorb_vars, factor_vars)
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

    ## create weight vector
    if weight == nothing
        w = fill(one(Float64), size(subdf, 1))
        sqrtw = w
    else
        w = convert(Vector{Float64}, subdf[weight])
        sqrtw = sqrt(w)
    end

    ## Compute factors, an array of AbtractFixedEffects
    if has_absorb
        factors = construct_fe(subdf, absorb_terms.terms, sqrtw)
    end

    ## Compute demeaned X
    mf = simpleModelFrame(subdf, rt, esample)
    coef_names = coefnames(mf)
    X = ModelMatrix(mf).m
    if weight != nothing
        broadcast!(*, X, X, sqrtw)
    end
    if has_absorb
        for j in 1:size(X, 2)
            (X[:,j], iterations, converged) = demean_vector!(X[:,j], factors)
        end
    end
    


    ## Compute demeaned y
    py = model_response(mf)[:]
    if eltype(py) != Float64
        y = convert(py, Float64)
    else
        y = py
    end
    if weight != nothing
        broadcast!(*, y, y, sqrtw)
    end
    if has_absorb
        (y, iterations, converged) = demean_vector!(y, factors)
    end


    #################
    # Do the loop that estimates jointly (beta, factors, loadings)
    #################
    
    H = At_mul_B(X, X)
    M = A_mul_Bt(inv(cholfact!(H)), X)
    estimate_factor_model(X, M,  y, df[m.id], df[m.time], m.dimension, maxiter = maxiter, tol = tol) 
    
end



##############################################################################
##
## Factor / beta iteration
##
##############################################################################




function fill_matrix!{F <: FloatingPoint}(res_matrix::Matrix{F}, y::Vector{F}, res_vector::Vector{F}, idrefs, timerefs)
    @inbounds @simd for i in 1:length(y)
        res_matrix[idrefs[i], timerefs[i]] = y[i] - res_vector[i]
    end
end

function fill_vector!{F <: FloatingPoint}(res_vector::Vector{F}, y::Vector{F}, res_matrix::Matrix{F}, idrefs, timerefs)
    @inbounds @simd for i in 1:length(y)
        res_vector[i] = y[i] - res_matrix[idrefs[i], timerefs[i]]
    end
end

function estimate_factor_model(X::Matrix{Float64}, M::Matrix{Float64}, y::Vector{Float64}, id::PooledDataArray, time::PooledDataArray, d::Int64; maxiter::Int64 = 10000, tol::Float64 = 1e-10)
    b = M * y
    res_vector = Array(Float64, length(y))
    # initialize at zero for missing values
    res_matrix = fill(zero(Float64), (length(id.pool), length(time.pool)))
    res_matrix2 = fill(zero(Float64), (length(id.pool), length(time.pool)))
    loadings = Array(Float64, (length(id.pool), d))
    factors = Array(Float64, (length(time.pool), d))
    variance = Array(Float64, (length(time.pool), length(time.pool)))

    converged = false
    iterations = maxiter
    tolerance = tol * length(b)
    iter = 0
    while iter < maxiter
        iter += 1
        oldb = deepcopy(b)
        (res_matrix2, res_matrix) = (res_matrix, res_matrix2)
        A_mul_B!(res_vector, X, b)
        # res_vector -> res_matrix
        fill_matrix!(res_matrix, y, res_vector, id.refs, time.refs)
        # create covariance matrix and do PCA
        At_mul_B!(variance, res_matrix, res_matrix)
        F = eigfact!(variance)
        # obtain d larger factors
        factors = sub(F[:vectors], :, (length(time.pool) - d + 1):length(time.pool))
        # compute the low rank approximation of res_matrix
        A_mul_Bt!(variance, factors, factors)
        A_mul_B!(res_matrix2, res_matrix, variance)
        # res_matrix -> res_vector
        fill_vector!(res_vector, y, res_matrix2, id.refs, time.refs)
        b = M * res_vector
        error = euclidean(b, oldb)
        if error < tolerance
            converged = true
            iterations = iter
            factors = deepcopy(factors)
            loadings = res_matrix * factors
            break
        end
    end
    scale!(loadings, 1/sqrt(length(time.pool)))
    scale!(factors, sqrt(length(time.pool)))
    FactorEstimate(id, time, b, loadings, factors, iterations, converged)
end


