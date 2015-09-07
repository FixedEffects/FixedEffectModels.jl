
type VcovWhite <: AbstractVcovMethod end

type VcovWhiteData <: AbstractVcovMethodData end

VcovMethodData(::VcovWhite, ::AbstractDataFrame) = VcovWhiteData()

function vcov!(v::VcovWhiteData, x::VcovData) 
    S = shat!(v, x)
    return sandwich(x.crossmatrix, S) 
end

function shat!{T}(::VcovWhiteData, x::VcovData{T, 1}) 
    X = x.regressors
    res = x.residuals
    Xu = scale!(res, X)
    S = At_mul_B(Xu, Xu)
    scale!(S, size(X, 1) / x.df_residual)
    return S
end

# S_{(l-1) * K + k, (l'-1)*K + k'} = \sum_i X[i, k] res[i, l] X[i, k'] res[i, l']
function shat!{T}(::VcovWhiteData, x::VcovData{T, 2}) 
    X = x.regressors
    res = x.residuals
    nobs = size(X, 1)
    dim = size(X, 2) * size(res, 2)
    S = fill(zero(Float64), (dim, dim))
    temp = fill(zero(Float64), nobs, dim)
    index = zero(Int)
    @inbounds for k in 1:size(X, 2), l in 1:size(res, 2)
        index += 1
        @simd for i in 1:nobs
            temp[i, index] = X[i, k]* res[i, l]
        end
    end
    S = At_mul_B(temp, temp)
    scale!(S, nobs / x.df_residual)
    return S
end


