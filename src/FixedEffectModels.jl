VERSION >= v"0.4.0-dev+6521" &&  __precompile__(true)

module FixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################
using Compat
using Base.BLAS
import Base: A_mul_B!, Ac_mul_B!, size, copy!
import Distributions: TDist, ccdf, FDist, Chisq, AliasTable, Categorical
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
residualize!,
solvefe!,
decompose!,
allvars,

Ones,
FixedEffect,
FixedEffectProblem,

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
include("utils/group.jl")
include("utils/chebyshev.jl")
include("utils/formula.jl")
include("utils/cgls.jl")

include("Ones.jl")
include("RegressionResult.jl")

include("fixedeffect/types.jl")
include("fixedeffect/solvefe.jl")
include("fixedeffect/residualize.jl")

include("vcov/types.jl")
include("vcov/vcovsimple.jl")
include("vcov/vcovwhite.jl")
include("vcov/vcovcluster.jl")
include("vcov/ranktest.jl")

include("reg.jl")
include("partial_out.jl")

if VERSION >= v"0.4.0-dev+6521"
	include("precompile.jl")
end


end  # module FixedEffectModels