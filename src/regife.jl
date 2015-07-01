##############################################################################
##
## Factor models (Bai 2009)
##
##############################################################################

type InteractiveFixedEffectModel 
    id::Symbol
    time::Symbol
    dimension::Int64
end


# Bai 2009
function reg(f::Formula, df::AbstractDataFrame, m::InteractiveFixedEffectModel; weight = nothing)

    rf = deepcopy(f)

    # decompose formula into normal  vs absorbpart
    (rf, has_absorb, absorb_formula) = decompose_absorb!(rf)
    if has_absorb
        absorb_vars = allvars(absorb_formula)
        absorb_terms = Terms(absorb_formula)
    else
        absorb_vars = Symbol[]
    end


    rt = Terms(rf)
    rt.intercept = false

    # create a dataframe without missing values & negative weights
    factor_vars = [m.id, m.time]
    vars = allvars(rf)
    all_vars = vcat(vars, absorb_vars, factor_vars)
    all_vars = unique(convert(Vector{Symbol}, all_vars))
    esample = complete_cases(df[all_vars])
    if weight != nothing
        esample &= isnaorneg(df[weight])
        all_vars = unique(vcat(all_vars, weight))
    end
    #subdf = sub(df[all_vars], esample)
    subdf = df[esample, all_vars]
    all_except_absorb_vars = unique(convert(Vector{Symbol}, vars))
    for v in all_except_absorb_vars
        dropUnusedLevels!(subdf[v])
    end

    # create weight vector
    if weight == nothing
        w = fill(one(Float64), size(subdf, 1))
        sqrtw = w
    else
        w = convert(Vector{Float64}, subdf[weight])
        sqrtw = sqrt(w)
    end

    # Compute factors, an array of AbtractFixedEffects
    if has_absorb
        factors = construct_fe(subdf, absorb_terms.terms, sqrtw)
    end

    # Compute demeaned X
    mf = simpleModelFrame(subdf, rt, esample)
    coef_names = coefnames(mf)
    X = ModelMatrix(mf).m
    if weight != nothing
        broadcast!(*, X, sqrtw, X)
    end
    if has_absorb
        for j in 1:size(X, 2)
            X[:,j] = demean_vector!(X[:,j], factors)
        end
    else
        meanv = - mean(X, 1)
        for j in 1:size(X, 2)
           X[:,j] =  addition_elementwise!(X[:,j], meanv[j])
        end
    end
    


    # Compute demeaned y
    py = model_response(mf)[:]
    if eltype(py) != Float64
        y = convert(py, Float64)
    else
        y = py
    end
    if weight != nothing
        multiplication_elementwise!(y, sqrtw)
    end
    if has_absorb
        y = demean_vector!(y, factors)
    else
        meanv = - mean(y)
        addition_elementwise!(y, meanv)
    end

    H = At_mul_B(X, X)
    M = A_mul_Bt(inv(cholfact!(H)), X)
    # get factors
    estimate_factor_model(X, M,  y, df[m.id], df[m.time], m.dimension) 
end



##############################################################################
##
## Result
##
##############################################################################


type FactorEstimate
    id::PooledDataArray
    time::PooledDataArray
    coef::Vector{Float64} 
    lambda::Matrix{Float64}  # N x d
    ft::Matrix{Float64} # d x T
    iter::Int64
end

function fill_matrix!(res_matrix, y, res_vector, idrefs, timerefs)
    @inbounds @simd for i in 1:length(y)
        res_matrix[idrefs[i], timerefs[i]] = y[i] - res_vector[i]
    end
end

function fill_vector!(res_vector, y, res_matrix, idrefs, timerefs)
    @inbounds @simd for i in 1:length(y)
        res_vector[i] = y[i] - res_matrix[idrefs[i], timerefs[i]]
    end
end

function estimate_factor_model(X::Matrix{Float64}, M::Matrix{Float64}, y::Vector{Float64}, id::PooledDataArray, time::PooledDataArray, d::Int64) 
    b = M * y
    res_vector = Array(Float64, length(y))
    # initialize at zero for missing values
    res_matrix = fill(zero(Float64), (length(id.pool), length(time.pool)))
    Lambda = Array(Float64, (length(id.pool), d))
    Ft = Array(Float64, (length(time.pool), d))
    max_iter = 10000
    tolerance = 1e-9
    iter = 0
    while iter < max_iter
        iter += 1
        oldb = deepcopy(b)
        # Compute predicted(regressor)
        A_mul_B!(res_vector, X, b)
        fill_matrix!(res_matrix, y, res_vector, id.refs, time.refs)
        svdresult = svdfact!(res_matrix) 
        A_mul_B!(Lambda, sub(svdresult.U, :, 1:d), diagm(sub(svdresult.S, 1:d)))
        A_mul_B!(res_matrix, Lambda, sub(svdresult.Vt, 1:d, :))
        fill_vector!(res_vector, y, res_matrix, id.refs, time.refs)
        # regress y - predicted(factor) over X
        b = M * res_vector
        error = euclidean(b, oldb)
        if error < tolerance
            Ft = sub(svdresult.Vt, 1:d, :)
            break
        end
    end
    scale!(Lambda, 1 / sqrt(length(time.pool)))
    scale!(Ft, sqrt(length(time.pool)))
    FactorEstimate(id, time, b, Lambda, Ft, iter)
end


