struct RobustCovariance <: CovarianceEstimator end

robust() = RobustCovariance()

function shat!(x::RegressionModel, ::RobustCovariance)
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

function StatsBase.vcov(x::RegressionModel, v::RobustCovariance)
    S = shat!(x, v)
    invcrossmodelmatrix = inv(crossmodelmatrix(x))
    pinvertible(Symmetric(invcrossmodelmatrix * S * invcrossmodelmatrix))
end