VERSION >= v"0.4.0-dev+6521" &&  __precompile__(true)

module FixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################
using Compat
import Distributions: TDist, ccdf, FDist, Chisq, AliasTable
import DataArrays: RefArray, PooledDataArray, PooledDataVector, DataArray, DataVector, compact, NAtype
import DataFrames: @~, DataFrame, AbstractDataFrame, ModelMatrix, ModelFrame, Terms, coefnames, Formula, complete_cases, names!
import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var, RegressionModel, model_response, stderr, confint, fit, CoefTable, df_residual
##############################################################################
##
## Exported methods and types 
##
##############################################################################

export group, 
reg,
partial_out,
demean!,
getfe,
decompose!,
allvars,

Ones,
FixedEffect,

AbstractRegressionResult,
RegressionResult,
RegressionResultIV,
RegressionResultFE,
RegressionResultFEIV,
title,
top,

AbstractVcovMethod,
AbstractVcovMethodData, 
VcovMethodData,
VcovData,
VcovSimple, 
VcovWhite, 
VcovCluster,
vcov!,
shat!

##############################################################################
##
## Load files
##
##############################################################################
include("utils/weight.jl")
include("utils/group.jl")
include("utils/formula.jl")
include("utils/compute_tss.jl")
include("utils/chebyshev.jl")
include("demean.jl")
include("vcov.jl")
include("RegressionResult.jl")
include("getfe.jl")
include("reg.jl")
include("partial_out.jl")

if VERSION >= v"0.4.0-dev+6521"
	include("precompile.jl")
end

end  # module FixedEffectModels