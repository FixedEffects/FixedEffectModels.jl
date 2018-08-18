##############################################################################
##
## VcovData stores data you need to compute errors
##
##############################################################################

struct VcovData{T, N} 
    regressors::Matrix{Float64}       # X
    crossmatrix::T                    # X'X in the simplest case. Can be Matrix but preferably Factorization
    residuals::Array{Float64, N}      # vector or matrix of residuals (matrix in the case of IV, residuals of Xendo on (Z, Xexo))
    dof_residual::Int
end

VcovDataVector{T} = VcovData{T, 1} 
VcovDataMatrix{T} = VcovData{T, 2} 




##############################################################################
##
## Any Method to COmpute Standard Errors must Define two Types
##
##############################################################################


abstract type AbstractVcovFormula end
# this type is used for syntax
# the argument vcov = v(x) calls the function VcovFormula(::Type{Val{:v}}, x) which must result into a type that inherits from AbstractVcovFormula
# Moreover, this type must define 
# (i) a function allvars that gives all variables needed to compute standard errors and a Vcovmethod
allvars(x::AbstractVcovFormula) = Symbol[]
# (ii) a VcovMethod that transform a dataframe and a object <: ABstractVcovFormula into a AbstractVcovMethod


# this type must implement
# vcov!(::VcovSimpleMethod, x::VcovData)
# shat!(::VcovSimpleMethod, x::VcovData)
abstract type AbstractVcovMethod end
sandwich(H, S::Matrix{Float64}) = H \ S * inv(H)
df_FStat(::AbstractVcovMethod, x::VcovData, hasintercept::Bool) = x.dof_residual - hasintercept
