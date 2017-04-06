__precompile__(true)

module FixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################
using Compat
import Base.BLAS: axpy!
import Base: A_mul_B!, Ac_mul_B!, size, sumabs2, copy!, getindex, length, fill!, norm, scale!, eltype, length, view, start, next, done
import Distributions: TDist, ccdf, FDist, Chisq, AliasTable, Categorical
import DataArrays: RefArray, PooledDataArray, PooledDataVector, DataArray, DataVector, compact, NAtype
import DataFrames: @~, DataFrame, AbstractDataFrame, ModelMatrix, ModelFrame, Terms, coefnames, Formula, completecases, names!, pool
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
getfe!,
decompose_iv!,
allvars,

WeightFormula,
Ones,
FixedEffectFormula,
FixedEffect,
FixedEffectProblem,

AbstractRegressionResult,
title,
top,
RegressionResult,
RegressionResultIV,
RegressionResultFE,
RegressionResultFEIV,

AbstractVcovFormula, 
VcovSimpleFormula, 
VcovWhiteFormula, 
VcovClusterFormula,

AbstractVcovMethod, 
VcovMethod,
VcovSimpleMethod, 
VcovWhiteMethod, 
VcovClusterMethod,

vcov!,
shat!,
VcovData,



@fe,
@vcov,
@vcovrobust,
@vcovcluster,
@weight


##############################################################################
##
## Load files
##
##############################################################################
include("utils/group.jl")
include("utils/formula.jl")
include("utils/lsmr.jl")
include("utils/basecol.jl")
include("utils/combinations.jl")

include("weight/Ones.jl")
include("weight/weight.jl")



include("fixedeffect/FixedEffect.jl")
include("fixedeffect/FixedEffectProblem.jl")
include("fixedeffect/FixedEffectProblem_LSMR.jl")
include("fixedeffect/FixedEffectProblem_Factorization.jl")

include("RegressionResult.jl")

include("vcov/types.jl")
include("vcov/vcovsimple.jl")
include("vcov/vcovrobust.jl")
include("vcov/vcovcluster.jl")
include("vcov/ranktest.jl")

include("reg.jl")
include("partial_out.jl")



end  # module FixedEffectModels