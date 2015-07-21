module FixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################
import Distributions: TDist, ccdf, FDist, Chisq
import Distances: chebyshev
import DataArrays: RefArray, PooledDataArray, PooledDataVector, DataArray, DataVector, compact, NAtype
import DataFrames: DataFrame, AbstractDataFrame, ModelMatrix, ModelFrame, Terms, coefnames, Formula, complete_cases, names!
import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var, RegressionModel, model_response, stderr, confint, fit, CoefTable
import GLM: df_residual, LinearModel

##############################################################################
##
## Exported methods and types 
##
##############################################################################

export group, 
reg,
partial_out,
demean!,
allvars,
FixedEffect,

RegressionResult,

AbstractFixedEffect,
FixedEffectIntercept, 
FixedEffectSlope,
VcovData,
AbstractVcovMethod,
AbstractVcovMethodData, 
VcovSimple, 
VcovWhite, 
VcovCluster,
VcovData
##############################################################################
##
## Load files
##
##############################################################################
include("utils.jl")
include("demean.jl")
include("vcov.jl")
include("RegressionResult.jl")
include("reg.jl")
include("partial_out.jl")

end  # module FixedEffectModels