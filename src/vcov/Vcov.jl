module Vcov
using LinearAlgebra
using StatsBase
using Combinatorics
using FixedEffects
using DataFrames

##############################################################################
##
## Any DataType must define the following methods:
##
##############################################################################
struct VcovData{T, N} <: RegressionModel
    modelmatrix::Matrix{Float64}       # X
    crossmodelmatrix::T                    # X'X in the simplest case. Can be Matrix but preferably Factorization
    residuals::Array{Float64, N}      # vector or matrix of residuals (matrix in the case of IV, residuals of Xendo on (Z, Xexo))
    dof_residual::Int
end
StatsBase.modelmatrix(x::VcovData) = x.modelmatrix
StatsBase.crossmodelmatrix(x::VcovData) = x.crossmodelmatrix
StatsBase.residuals(x::VcovData) = x.residuals
StatsBase.dof_residual(x::VcovData) = x.dof_residual

##############################################################################
##
## Any type used for standard errors must define the following methods:
##
##############################################################################
materialize(df::AbstractDataFrame, v::CovarianceEstimator) = v
completecases(df::AbstractDataFrame, ::CovarianceEstimator) = trues(size(df, 1))
S_hat(x::RegressionModel, ::CovarianceEstimator) = error("S_hat not defined for this type")
df_FStat(x::RegressionModel, ::CovarianceEstimator, hasintercept::Bool) = dof_residual(x) - hasintercept



include("utils.jl")
include("vcovsimple.jl")
include("vcovrobust.jl")
include("vcovcluster.jl")

end