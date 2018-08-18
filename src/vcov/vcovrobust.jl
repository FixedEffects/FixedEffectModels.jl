VcovFormula(::Type{Val{:robust}}) = VcovRobustFormula()

struct VcovRobustFormula <: AbstractVcovFormula  end


struct VcovRobustMethod <: AbstractVcovMethod end
VcovMethod(::AbstractDataFrame, ::VcovRobustFormula) = VcovRobustMethod()

function vcov!(v::VcovRobustMethod, x::VcovData) 
    S = shat!(v, x)
    return sandwich(x.crossmatrix, S) 
end


# S_{(l-1) * K + k, (l'-1)*K + k'} = \sum_i X[i, k] res[i, l] X[i, k'] res[i, l']
function shat!(::VcovRobustMethod, x::VcovData{T, N}) where {T, N}
    dim = size(x.regressors, 2) * size(x.residuals, 2)
    X2 = fill(zero(Float64), size(x.regressors, 1), dim)
    index = 0
    for k in 1:size(x.residuals, 2)
        for j in 1:size(x.regressors, 2)
            index += 1
            @inbounds @simd for i in 1:size(x.regressors, 1)
                X2[i, index] = x.regressors[i, j] * x.residuals[i, k]
            end
        end
    end
    S2 = X2' * X2
    rmul!(S2, size(x.regressors, 1) / x.dof_residual)
    return S2
end


