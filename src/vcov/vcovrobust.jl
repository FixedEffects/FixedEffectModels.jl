struct Robust <: AbstractVcov  end
robust() = Robust()

struct RobustMethod <: AbstractVcovMethod end
VcovMethod(::AbstractDataFrame, ::Robust) = RobustMethod()

function vcov!(v::RobustMethod, x::VcovData) 
    S = shat!(v, x)
    invcrossmatrix = inv(crossmatrix(x))
    return pinvertible(Symmetric(invcrossmatrix * S * invcrossmatrix))
end

# S_{(l-1) * K + k, (l'-1)*K + k'} = \sum_i X[i, k] res[i, l] X[i, k'] res[i, l']
function shat!(::RobustMethod, x::VcovData{T, N}) where {T, N}
    m = modelmatrix(x)
    r = residuals(x)
    X2 = zeros(size(m, 1), size(m, 2) * size(r, 2))
    index = 0
    for k in 1:size(r, 2)
        for j in 1:size(m, 2)
            index += 1
            for i in 1:size(m, 1)
                X2[i, index] = m[i, j] * r[i, k]
            end
        end
    end
    S2 = X2' * X2
    Symmetric(rmul!(S2, size(m, 1) / dof_residual(x)))
end
