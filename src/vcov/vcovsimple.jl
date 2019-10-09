struct SimpleCovariance <: CovarianceEstimator end

simple() = SimpleCovariance()

function shat!(x::RegressionModel, ::SimpleCovariance)
	rmul!(crossmodelmatrix(x), sum(abs2, residuals(x)))
end

function StatsBase.vcov(x::RegressionModel, ::SimpleCovariance)
    invcrossmodelmatrix = Matrix(inv(crossmodelmatrix(x)))
    rmul!(invcrossmodelmatrix, sum(abs2, residuals(x)) /  dof_residual(x))
    Symmetric(invcrossmodelmatrix)
end

