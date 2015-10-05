##############################################################################
##
## VcovData stores data you need to compute errors
##
##############################################################################

type VcovData{T, N} 
    regressors::Matrix{Float64}       # X
    crossmatrix::T                    # X'X in the simplest case. Can be Matrix but preferably Factorization
    residuals::Array{Float64, N}      # vector or matrix of residuals (matrix in the case of IV, residuals of Xendo on (Z, Xexo))
    df_residual::Int
end

typealias VcovDataVector{T} VcovData{T, 1} 
typealias VcovDataMatrix{T} VcovData{T, 2} 

##############################################################################
##
## Each method to compute standard errors should define two types
##
## 1. A type that inherits from AbstractVcovMethod (used for the construction)
## It must implement two methods:
## - allvars that specify variables needed
## - VcovMethodData(x, df::AbstractDataFrame) that construct a VcocMethodData object
##   Typicially, it uses a list of symbols and a dataframe to obtain vectors
##
## 2. A type that inherits from AbstractVcovMethodData (used for the computation)
## It must implement two methods: 
## - shat! returns a S hat matrix. It may change regressors in place
## - vcov! returns a covariance matrix. It may change regressors in matrix
##
##############################################################################


##############################################################################
##
## VcovMethod stores data you need to compute errors
##
##############################################################################

abstract AbstractVcovMethod
allvars(::AbstractVcovMethod) = Symbol[]

##############################################################################
##
## VcovDataMethod stores data you need to compute errors
##
##############################################################################

abstract AbstractVcovMethodData

# utilities

sandwich(H, S::Matrix{Float64}) = H \ S * inv(H)
df_FStat(::AbstractVcovMethodData, x::VcovData, hasintercept::Bool) = x.df_residual - hasintercept
