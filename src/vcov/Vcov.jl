module Vcov

using StatsBase
using DataFrames
using LinearAlgebra
using FixedEffects
using Combinatorics


crossmodelmatrix(x::RegressionModel) = cholesky!(modelmatrix(x)' * modelmatrix(x))
##############################################################################
##
## Any type used for standard errors must define the following methods:
##
##############################################################################
materialize(df::AbstractDataFrame, v::CovarianceEstimator) = v
completecases(df::AbstractDataFrame, ::CovarianceEstimator) = trues(size(df, 1))
shat!(x::RegressionModel, ::CovarianceEstimator) = error("shat! not defined for this type")
df_FStat(x::RegressionModel, ::CovarianceEstimator, hasintercept::Bool) = dof_residual(x) - hasintercept

include("utils.jl")
include("vcovsimple.jl")
include("vcovrobust.jl")
include("vcovcluster.jl")

end