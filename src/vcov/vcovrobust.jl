VcovFormula(::Type{Val{:robust}}) = VcovRobustFormula()

type VcovRobustFormula <: AbstractVcovFormula  end
allvars(x::VcovRobustFormula) = Symbol[]


type VcovRobustMethod <: AbstractVcovMethod end

VcovMethod(::AbstractDataFrame, ::VcovRobustFormula) = VcovRobustMethod()

function vcov!(v::VcovRobustMethod, x::VcovData) 
    S = shat!(v, x)
    return sandwich(x.crossmatrix, S) 
end

function shat!(::VcovRobustMethod, x::VcovData{T, 1}) where {T}
    X = x.regressors
    res = x.residuals
    Xu = scale!(res, X)
    S = At_mul_B(Xu, Xu)
    scale!(S, size(X, 1) / x.df_residual)
    return S
end

# S_{(l-1) * K + k, (l'-1)*K + k'} = \sum_i X[i, k] res[i, l] X[i, k'] res[i, l']
function shat!(::VcovRobustMethod, x::VcovData{T, 2}) where {T}
    X = x.regressors
    res = x.residuals
    nobs = size(X, 1)
    dim = size(X, 2) * size(res, 2)
    S = fill(zero(Float64), (dim, dim))
    temp = fill(zero(Float64), nobs, dim)
    index = zero(Int)
    for k in 1:size(X, 2), l in 1:size(res, 2)
        index += 1
        for i in 1:nobs
            temp[i, index] = X[i, k]* res[i, l]
        end
    end
    S = At_mul_B(temp, temp)
    scale!(S, nobs / x.df_residual)
    return S
end


