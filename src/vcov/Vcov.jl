module Vcov

using StatsBase
using DataFrames
using LinearAlgebra
using FixedEffects
using Combinatorics
##############################################################################
##
## VcovData stores data you need to compute errors
##
##############################################################################

struct VcovData{T, N} 
    modelmatrix::Matrix{Float64}       # X
    crossmatrix::T                    # X'X in the simplest case. Can be Matrix but preferably Factorization
    residuals::Array{Float64, N}      # vector or matrix of residuals (matrix in the case of IV, residuals of Xendo on (Z, Xexo))
    dof_residual::Int
end

VcovDataVector{T} = VcovData{T, 1} 
VcovDataMatrix{T} = VcovData{T, 2} 
StatsBase.modelmatrix(x::VcovData) = x.modelmatrix
crossmatrix(x::VcovData) = x.crossmatrix
StatsBase.residuals(x::VcovData) = x.residuals
StatsBase.dof_residual(x::VcovData) = x.dof_residual




##############################################################################
##
## Any Method to COmpute Standard Errors must Define two Types
##
##############################################################################


abstract type AbstractVcov end
VcovMethod(df::AbstractDataFrame, x::AbstractVcov) = error("VcovMethod not defined for this type")
DataFrames.completecases(df::AbstractDataFrame, x::AbstractVcov) = trues(size(df, 1))


abstract type AbstractVcovMethod end
vcov!(::AbstractVcovMethod, x::VcovData) = error("vcov! not defined for this type")
shat!(::AbstractVcovMethod, x::VcovData) = error("shat! not defined for this type")
df_FStat(::AbstractVcovMethod, x::VcovData, hasintercept::Bool) = dof_residual(x) - hasintercept


include("vcovsimple.jl")
include("vcovrobust.jl")
include("vcovcluster.jl")
include("utils.jl")

end