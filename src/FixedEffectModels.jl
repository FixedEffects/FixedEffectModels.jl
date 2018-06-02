
module FixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################
import Base.BLAS: axpy!
import Base: A_mul_B!, Ac_mul_B!, size, copy!, getindex, length, fill!, norm, scale!, eltype, length, view, start, next, done
import Distributions: ccdf, TDist, FDist, Chisq
import Missings: Missing
import DataArrays: DataArray
import CategoricalArrays: CategoricalArray, CategoricalVector, compress, categorical, CategoricalPool, levels, droplevels!
import DataFrames: DataFrame, AbstractDataFrame, completecases, names!, ismissing
using Reexport
import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var, RegressionModel, model_response, stderror, confint, fit, CoefTable, df_residual, r2, r2adjr
@reexport using StatsBase
import StatsModels: @formula,  Formula, ModelFrame, ModelMatrix, Terms, coefnames
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
fes,

WeightFormula,
Ones,
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
VcovRobustFormula, 
VcovClusterFormula,
VcovFormula,

AbstractVcovMethod, 
VcovMethod,
VcovSimpleMethod, 
VcovWhiteMethod, 
VcovClusterMethod,

vcov!,
shat!,
VcovData,

Model,
@model

##############################################################################
##
## Load files
##
##############################################################################
include("utils/group.jl")
include("utils/isnested.jl")
include("utils/formula.jl")
include("utils/model.jl")
include("utils/lsmr.jl")
include("utils/basecol.jl")
include("utils/combinations.jl")

include("weight/Ones.jl")
include("weight/weight.jl")



include("fixedeffect/FixedEffect.jl")
include("fixedeffect/FixedEffectProblem.jl")
include("fixedeffect/FixedEffectProblem_LSMR.jl")
if isdefined(Base.SparseArrays, :CHOLMOD)
    include("fixedeffect/FixedEffectProblem_Factorization.jl")
end

include("RegressionResult.jl")

include("vcov/types.jl")
include("vcov/vcovsimple.jl")
include("vcov/vcovrobust.jl")
include("vcov/vcovcluster.jl")
include("vcov/ranktest.jl")

include("reg.jl")
include("partial_out.jl")



end  # module FixedEffectModels