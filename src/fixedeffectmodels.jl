module FixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################
import Distributions: TDist, ccdf, FDist
import Distances: sqeuclidean
import DataArrays: PooledDataArray, DataArray, DataVector
import DataFrames: DataFrame, AbstractDataFrame, ModelMatrix, ModelFrame, Terms, coefnames, Formula, complete_cases, names!
import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var, RegressionModel, model_response, stderr
import GLM: df_residual, LinearModel

##############################################################################
##
## Exported methods and types 
##
##############################################################################

export group, 
reg,

RegressionResult,

allvars,
AbstractVcov, 
VcovSimple, 
VcovWhite, 
VcovHac, 
VcovCluster,
AbstractVcovData, 
VcovData, 
VcovDataHat

##############################################################################
##
## Load files
##
##############################################################################
include("utils.jl")
include("vcov.jl")
include("RegressionResult.jl")
include("reg.jl")

end  # module FixedEffectModels