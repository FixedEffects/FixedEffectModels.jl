Vcov(::Type{Val{:robust}}) = VcovRobust()

struct VcovRobust <: AbstractVcov  end

struct VcovRobustMethod <: AbstractVcovMethod end
VcovMethod(::AbstractDataFrame, ::VcovRobust) = VcovRobustMethod()

function vcov!(v::VcovRobustMethod, x::VcovData) 
    S = shat!(v, x)
    invcrossmatrix = inv(x.crossmatrix)
    return pinvertible(Symmetric(invcrossmatrix * S * invcrossmatrix))
end

# S_{(l-1) * K + k, (l'-1)*K + k'} = \sum_i X[i, k] res[i, l] X[i, k'] res[i, l']
function shat!(::VcovRobustMethod, x::VcovData{T, N}) where {T, N}
    X2 = zeros(size(x.regressors, 1), size(x.regressors, 2) * size(x.residuals, 2))
    index = 0
    for k in 1:size(x.residuals, 2)
        for j in 1:size(x.regressors, 2)
            index += 1
            for i in 1:size(x.regressors, 1)
                X2[i, index] = x.regressors[i, j] * x.residuals[i, k]
            end
        end
    end
    S2 = X2' * X2
    Symmetric(rmul!(S2, size(x.regressors, 1) / x.dof_residual))
end
