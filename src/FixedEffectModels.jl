
module FixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################
using Compat
import Base.BLAS: axpy!
import Base.Broadcast: broadcast!
import Base: A_mul_B!, Ac_mul_B!, size, sumabs2, copy!, scale!, getindex, length, fill!, dot
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
getfe!,
decompose!,
allvars,

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


AbstractVcovMethod,
AbstractVcovMethodData, 
vcov!,
shat!,
VcovMethodData,
VcovData,
VcovSimple, 
VcovWhite, 
VcovCluster


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
include("fixedeffect/solve.jl")


include("vcov/types.jl")
include("vcov/vcovsimple.jl")
include("vcov/vcovwhite.jl")
include("vcov/vcovcluster.jl")
include("vcov/ranktest.jl")

include("reg.jl")
include("partial_out.jl")



end  # module FixedEffectModels