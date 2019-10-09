module Vcov

using StatsBase
using DataFrames
using LinearAlgebra
using FixedEffects
using Combinatorics


##############################################################################
##
## Any Method to Compute Standard Errors must Define two Types
##
##############################################################################
crossmodelmatrix(x::RegressionModel) = cholesky!(modelmatrix(x)' * modelmatrix(x))
materialize(x::CovarianceEstimator, df::AbstractDataFrame) = x
DataFrames.completecases(df::AbstractDataFrame, x::CovarianceEstimator) = trues(size(df, 1))
shat!(x::RegressionModel, ::CovarianceEstimator) = error("shat! not defined for this type")
df_FStat(x::RegressionModel, ::CovarianceEstimator, hasintercept::Bool) = dof_residual(x) - hasintercept

include("vcovsimple.jl")
include("vcovrobust.jl")
include("vcovcluster.jl")
include("utils.jl")

end